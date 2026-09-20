#include "ter_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>

#ifdef WITH_GPU

__device__ inline bool check_and_add_ternary_gpu(
    const TerEntry& u, const TerEntry& v, const TerEntry& w,
    TerEntry& out
) {
    uint64_t bp_lo = u.pos_lo & v.pos_lo;
    uint64_t bn_lo = u.neg_lo & v.neg_lo;
    uint64_t ep_lo = (u.pos_lo | v.pos_lo) & ~(u.neg_lo | v.neg_lo);
    uint64_t en_lo = (u.neg_lo | v.neg_lo) & ~(u.pos_lo | v.pos_lo);

    if ((bp_lo & ~w.neg_lo) || (bn_lo & ~w.pos_lo) ||
        (ep_lo & w.pos_lo)   || (en_lo & w.neg_lo)) {
        return false;
    }

    uint64_t bp_hi = u.pos_hi & v.pos_hi;
    uint64_t bn_hi = u.neg_hi & v.neg_hi;
    uint64_t ep_hi = (u.pos_hi | v.pos_hi) & ~(u.neg_hi | v.neg_hi);
    uint64_t en_hi = (u.neg_hi | v.neg_hi) & ~(u.pos_hi | v.pos_hi);

    if ((bp_hi & ~w.neg_hi) || (bn_hi & ~w.pos_hi) ||
        (ep_hi & w.pos_hi)   || (en_hi & w.neg_hi)) {
        return false;
    }

    uint64_t neither_lo = ~(bp_lo | ep_lo | bn_lo | en_lo);
    out.pos_lo = (bp_lo & w.neg_lo) | (ep_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.pos_lo);
    out.neg_lo = (bn_lo & w.pos_lo) | (en_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.neg_lo);

    uint64_t neither_hi = ~(bp_hi | ep_hi | bn_hi | en_hi);
    out.pos_hi = (bp_hi & w.neg_hi) | (ep_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.pos_hi);
    out.neg_hi = (bn_hi & w.pos_hi) | (en_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.neg_hi);

    out.psum = u.psum + v.psum + w.psum;
    return true;
}

// ── Kernel lama (binary search penuh di d_C) — tidak diubah ──
__global__ void level1_merge_kernel(
    const TerEntry* __restrict__ d_A, size_t size_A,
    const TerEntry* __restrict__ d_B, size_t size_B,
    const TerEntry* __restrict__ d_C, size_t size_C,
    uint64_t target_mod,
    uint64_t mask_m1,
    TerEntry* __restrict__ d_out,
    unsigned long long* __restrict__ d_out_count,
    size_t max_out_capacity
) {
    size_t total_pairs = size_A * size_B;
    size_t pair_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    size_t i_a = pair_idx / size_B;
    size_t i_b = pair_idx % size_B;

    TerEntry u = d_A[i_a];
    TerEntry v = d_B[i_b];

    uint64_t req_rem = (target_mod - ((uint64_t)(u.psum + v.psum) & mask_m1)) & mask_m1;

    size_t low = 0, high = size_C;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        uint64_t val = (uint64_t)d_C[mid].psum & mask_m1;
        if (val < req_rem) low = mid + 1;
        else high = mid;
    }

    for (size_t k = low; k < size_C; ++k) {
        uint64_t val = (uint64_t)d_C[k].psum & mask_m1;
        if (val != req_rem) break;

        TerEntry w = d_C[k];
        TerEntry combined;
        if (check_and_add_ternary_gpu(u, v, w, combined)) {
            unsigned long long out_pos = atomicAdd(d_out_count, 1ULL);
            if (out_pos < max_out_capacity) {
                d_out[out_pos] = combined;
            }
        }
    }
}

// ── Kernel bucket (langkah 4c) ──
// Per pasangan: baca a_ps[i_a] (broadcast) dan b_ps[i_b] (coalesced), cari di bucket table,
// dan baru muat TerEntry u/v/w bila kunci cocok. Himpunan hasil sama dengan kernel lama
// (pemeriksaan kecocokan akhir tetap check_and_add_ternary_gpu).
__global__ void level1_merge_kernel_bucket(
    const TerEntry* __restrict__ d_A, size_t size_A,
    const TerEntry* __restrict__ d_B, size_t size_B,
    const TerEntry* __restrict__ d_C,
    const uint64_t* __restrict__ d_aps,
    const uint64_t* __restrict__ d_bps,
    const uint64_t* __restrict__ d_ckeys,
    const uint32_t* __restrict__ d_start,
    int shift,
    uint64_t target_mod,
    uint64_t mask_m1,
    TerEntry* __restrict__ d_out,
    unsigned long long* __restrict__ d_out_count,
    size_t max_out_capacity
) {
    size_t total_pairs = size_A * size_B;
    size_t pair_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    size_t i_a = pair_idx / size_B;
    size_t i_b = pair_idx % size_B;

    uint64_t req = (target_mod - d_aps[i_a] - d_bps[i_b]) & mask_m1;

    uint32_t bucket = (uint32_t)(req >> shift);
    uint32_t end = d_start[bucket + 1];
    uint32_t lo = d_start[bucket];
    uint32_t hi = end;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (d_ckeys[mid] < req) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= end || d_ckeys[lo] != req) return;

    TerEntry u = d_A[i_a];
    TerEntry v = d_B[i_b];
    for (uint32_t k = lo; k < end && d_ckeys[k] == req; ++k) {
        TerEntry w = d_C[k];
        TerEntry combined;
        if (check_and_add_ternary_gpu(u, v, w, combined)) {
            unsigned long long out_pos = atomicAdd(d_out_count, 1ULL);
            if (out_pos < max_out_capacity) {
                d_out[out_pos] = combined;
            }
        }
    }
}

namespace {

GpuMergeStats g_stats;

using clk = std::chrono::high_resolution_clock;
inline double ms_since(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Buffer device dengan pelepasan otomatis (semua jalur return aman dari kebocoran).
struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { release(); }
    void release() { if (p) { cudaFree(p); p = nullptr; } }
};

bool dev_alloc(DevBuf& b, size_t bytes, const char* what) {
    cudaError_t e = cudaMalloc(&b.p, bytes ? bytes : 1);
    if (e != cudaSuccess) {
        std::cerr << "[GPU] cudaMalloc(" << what << ", " << bytes << " B) gagal: "
                  << cudaGetErrorString(e) << "\n";
        b.p = nullptr;
        return false;
    }
    return true;
}

bool cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "[GPU] " << what << " gagal: " << cudaGetErrorString(e) << "\n";
        return false;
    }
    return true;
}

struct EventSet {
    cudaEvent_t ev[4];
    bool ok = true;
    EventSet() { for (auto& e : ev) { if (cudaEventCreate(&e) != cudaSuccess) ok = false; } }
    ~EventSet() { for (auto& e : ev) cudaEventDestroy(e); }
};

} // namespace

GpuMergeStats gpu_merge_stats_get() { return g_stats; }
void gpu_merge_stats_reset() { g_stats = GpuMergeStats{}; }

bool run_level1_merge_gpu(
    const std::vector<TerEntry>& A,
    const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C,
    uint64_t target_mod,
    uint64_t mask_m1,
    size_t max_cap,
    std::vector<TerEntry>& L1_out,
    const Level1Sidecar* sidecar
) {
    if (A.empty() || B.empty() || sorted_C.empty()) return false;

    const auto t_call0 = clk::now();
    const bool first_call = (g_stats.calls == 0);

    // Guard ukuran grid (gridDim.x <= 2^31-1) dan overflow A*B.
    if (A.size() > SIZE_MAX / B.size()) {
        std::cerr << "[GPU] |A|*|B| overflow size_t\n";
        return false;
    }
    const size_t total_pairs = A.size() * B.size();
    const size_t block_size = 256;
    const size_t grid_size = (total_pairs + block_size - 1) / block_size;
    if (grid_size > 2147483647ULL) {
        std::cerr << "[GPU] grid terlalu besar (" << grid_size << " blok); butuh pemecahan pasangan\n";
        return false;
    }

    const bool use_bucket = (sidecar != nullptr && sidecar->valid &&
                             sidecar->a_ps.size() == A.size() && sidecar->b_ps.size() == B.size() &&
                             sidecar->c_keys.size() == sorted_C.size() && !sidecar->start.empty());

    const size_t size_A_bytes = A.size() * sizeof(TerEntry);
    const size_t size_B_bytes = B.size() * sizeof(TerEntry);
    const size_t size_C_bytes = sorted_C.size() * sizeof(TerEntry);
    const size_t size_out_bytes = max_cap * sizeof(TerEntry);

    DevBuf d_A, d_B, d_C, d_out, d_out_count;
    DevBuf d_aps, d_bps, d_ckeys, d_start;

    const auto t_alloc0 = clk::now();
    if (!dev_alloc(d_A, size_A_bytes, "A")) return false;
    if (!dev_alloc(d_B, size_B_bytes, "B")) return false;
    if (!dev_alloc(d_C, size_C_bytes, "C")) return false;
    if (!dev_alloc(d_out, size_out_bytes, "out")) return false;
    if (!dev_alloc(d_out_count, sizeof(unsigned long long), "out_count")) return false;
    if (use_bucket) {
        if (!dev_alloc(d_aps, A.size() * sizeof(uint64_t), "a_ps")) return false;
        if (!dev_alloc(d_bps, B.size() * sizeof(uint64_t), "b_ps")) return false;
        if (!dev_alloc(d_ckeys, sidecar->c_keys.size() * sizeof(uint64_t), "c_keys")) return false;
        if (!dev_alloc(d_start, sidecar->start.size() * sizeof(uint32_t), "start")) return false;
    }
    const double alloc_ms = ms_since(t_alloc0, clk::now());

    EventSet ev;
    if (ev.ok) cudaEventRecord(ev.ev[0]);

    if (!cuda_ok(cudaMemcpy(d_A.p, A.data(), size_A_bytes, cudaMemcpyHostToDevice), "memcpy A")) return false;
    if (!cuda_ok(cudaMemcpy(d_B.p, B.data(), size_B_bytes, cudaMemcpyHostToDevice), "memcpy B")) return false;
    if (!cuda_ok(cudaMemcpy(d_C.p, sorted_C.data(), size_C_bytes, cudaMemcpyHostToDevice), "memcpy C")) return false;
    if (!cuda_ok(cudaMemset(d_out_count.p, 0, sizeof(unsigned long long)), "memset out_count")) return false;
    if (use_bucket) {
        if (!cuda_ok(cudaMemcpy(d_aps.p, sidecar->a_ps.data(), A.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "memcpy a_ps")) return false;
        if (!cuda_ok(cudaMemcpy(d_bps.p, sidecar->b_ps.data(), B.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "memcpy b_ps")) return false;
        if (!cuda_ok(cudaMemcpy(d_ckeys.p, sidecar->c_keys.data(), sidecar->c_keys.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "memcpy c_keys")) return false;
        if (!cuda_ok(cudaMemcpy(d_start.p, sidecar->start.data(), sidecar->start.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "memcpy start")) return false;
    }
    if (ev.ok) cudaEventRecord(ev.ev[1]);

    if (use_bucket) {
        level1_merge_kernel_bucket<<<(unsigned int)grid_size, (unsigned int)block_size>>>(
            static_cast<const TerEntry*>(d_A.p), A.size(),
            static_cast<const TerEntry*>(d_B.p), B.size(),
            static_cast<const TerEntry*>(d_C.p),
            static_cast<const uint64_t*>(d_aps.p), static_cast<const uint64_t*>(d_bps.p),
            static_cast<const uint64_t*>(d_ckeys.p), static_cast<const uint32_t*>(d_start.p),
            sidecar->shift, target_mod, mask_m1,
            static_cast<TerEntry*>(d_out.p), static_cast<unsigned long long*>(d_out_count.p), max_cap
        );
    } else {
        level1_merge_kernel<<<(unsigned int)grid_size, (unsigned int)block_size>>>(
            static_cast<const TerEntry*>(d_A.p), A.size(),
            static_cast<const TerEntry*>(d_B.p), B.size(),
            static_cast<const TerEntry*>(d_C.p), sorted_C.size(),
            target_mod, mask_m1,
            static_cast<TerEntry*>(d_out.p), static_cast<unsigned long long*>(d_out_count.p), max_cap
        );
    }
    if (!cuda_ok(cudaGetLastError(), "kernel launch")) return false;
    if (ev.ok) cudaEventRecord(ev.ev[2]);
    if (!cuda_ok(cudaDeviceSynchronize(), "kernel sync")) return false;

    unsigned long long h_count = 0;
    if (!cuda_ok(cudaMemcpy(&h_count, d_out_count.p, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "memcpy count")) return false;
    const size_t actual_out = std::min((size_t)h_count, max_cap);

    L1_out.resize(actual_out);
    if (actual_out > 0) {
        if (!cuda_ok(cudaMemcpy(L1_out.data(), d_out.p, actual_out * sizeof(TerEntry), cudaMemcpyDeviceToHost), "memcpy out")) return false;
    }
    if (ev.ok) cudaEventRecord(ev.ev[3]);

    double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
    if (ev.ok && cudaEventSynchronize(ev.ev[3]) == cudaSuccess) {
        float f = 0.f;
        if (cudaEventElapsedTime(&f, ev.ev[0], ev.ev[1]) == cudaSuccess) h2d_ms = f;
        if (cudaEventElapsedTime(&f, ev.ev[1], ev.ev[2]) == cudaSuccess) kernel_ms = f;  // event distempel GPU setelah kernel selesai, jadi = durasi kernel
        if (cudaEventElapsedTime(&f, ev.ev[2], ev.ev[3]) == cudaSuccess) d2h_ms = f;
    }

    const auto t_free0 = clk::now();
    d_A.release(); d_B.release(); d_C.release(); d_out.release(); d_out_count.release();
    d_aps.release(); d_bps.release(); d_ckeys.release(); d_start.release();
    const double free_ms = ms_since(t_free0, clk::now());

    ++g_stats.calls;
    g_stats.alloc_ms += alloc_ms;
    g_stats.h2d_ms += h2d_ms;
    g_stats.kernel_ms += kernel_ms;
    g_stats.d2h_ms += d2h_ms;
    g_stats.free_ms += free_ms;
    if (first_call) g_stats.first_call_ms = ms_since(t_call0, clk::now());

    return true;
}

#endif
