#include "ter_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>

#ifdef WITH_GPU

__device__ inline bool check_and_add_ternary_gpu(const TerEntry& u, const TerEntry& v, const TerEntry& w, TerEntry& out) {
    uint64_t bp_lo = u.pos_lo & v.pos_lo;
    uint64_t bn_lo = u.neg_lo & v.neg_lo;
    uint64_t ep_lo = (u.pos_lo | v.pos_lo) & ~(u.neg_lo | v.neg_lo);
    uint64_t en_lo = (u.neg_lo | v.neg_lo) & ~(u.pos_lo | v.pos_lo);
    if ((bp_lo & ~w.neg_lo) || (bn_lo & ~w.pos_lo) || (ep_lo & w.pos_lo) || (en_lo & w.neg_lo)) return false;

    uint64_t bp_hi = u.pos_hi & v.pos_hi;
    uint64_t bn_hi = u.neg_hi & v.neg_hi;
    uint64_t ep_hi = (u.pos_hi | v.pos_hi) & ~(u.neg_hi | v.neg_hi);
    uint64_t en_hi = (u.neg_hi | v.neg_hi) & ~(u.pos_hi | v.pos_hi);
    if ((bp_hi & ~w.neg_hi) || (bn_hi & ~w.pos_hi) || (ep_hi & w.pos_hi) || (en_hi & w.neg_hi)) return false;

    uint64_t neither_lo = ~(bp_lo | ep_lo | bn_lo | en_lo);
    out.pos_lo = (bp_lo & w.neg_lo) | (ep_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.pos_lo);
    out.neg_lo = (bn_lo & w.pos_lo) | (en_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.neg_lo);
    uint64_t neither_hi = ~(bp_hi | ep_hi | bn_hi | en_hi);
    out.pos_hi = (bp_hi & w.neg_hi) | (ep_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.pos_hi);
    out.neg_hi = (bn_hi & w.pos_hi) | (en_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.neg_hi);
    out.psum = u.psum + v.psum + w.psum;
    return true;
}

__global__ void level1_merge_kernel(
    const TerEntry* __restrict__ d_A, size_t size_A, const TerEntry* __restrict__ d_B, size_t size_B,
    const TerEntry* __restrict__ d_C, size_t size_C, uint64_t target_mod, uint64_t mask_m1,
    TerEntry* __restrict__ d_out, unsigned long long* __restrict__ d_out_count, size_t max_out_capacity)
{
    // 2D tiled loop: tiap blok menangani baris-baris A (i_a, stride gridDim.x), tiap
    // thread di dalam blok menyapu kolom B (i_b, stride blockDim.x). Grid tetap di-cap di
    // gpu_l1_launch, jadi tidak ada ketergantungan pada gridDim.x <= 2^31-1 relatif
    // terhadap total_pairs = |A|*|B| (yang dulu membuat launch gagal untuk n=96, pair
    // ~10^12+). Yang PALING penting: tidak ada pembagian/modulo 64-bit per pasangan —
    // pair_idx / size_B dan pair_idx % size_B lebih lambat ~200+ cycle di Turing
    // (T4 tidak punya integer-division 64-bit native), dan itu dieksekusi per pair.
    for (size_t i_a = (size_t)blockIdx.x; i_a < size_A; i_a += (size_t)gridDim.x) {
        const TerEntry u = d_A[i_a];
        for (size_t i_b = (size_t)threadIdx.x; i_b < size_B; i_b += (size_t)blockDim.x) {
            const TerEntry v = d_B[i_b];
            uint64_t req_rem = (target_mod - ((uint64_t)(u.psum + v.psum) & mask_m1)) & mask_m1;
            size_t low = 0, high = size_C;
            while (low < high) {
                size_t mid = low + (high - low) / 2;
                uint64_t val = (uint64_t)d_C[mid].psum & mask_m1;
                if (val < req_rem) low = mid + 1; else high = mid;
            }
            for (size_t k = low; k < size_C; ++k) {
                uint64_t val = (uint64_t)d_C[k].psum & mask_m1;
                if (val != req_rem) break;
                TerEntry w = d_C[k];
                TerEntry combined;
                if (check_and_add_ternary_gpu(u, v, w, combined)) {
                    unsigned long long out_pos = atomicAdd(d_out_count, 1ULL);
                    if (out_pos < max_out_capacity) d_out[out_pos] = combined;
                }
            }
        }
    }
}

__global__ void level1_merge_kernel_bucket(
    const TerEntry* __restrict__ d_A, size_t size_A, const TerEntry* __restrict__ d_B, size_t size_B,
    const TerEntry* __restrict__ d_C, const uint64_t* __restrict__ d_aps, const uint64_t* __restrict__ d_bps,
    const uint64_t* __restrict__ d_ckeys, const uint32_t* __restrict__ d_start, int shift,
    uint64_t target_mod, uint64_t mask_m1, TerEntry* __restrict__ d_out,
    unsigned long long* __restrict__ d_out_count, size_t max_out_capacity)
{
    // 2D tiled loop: sama seperti level1_merge_kernel — tanpa pembagian/modulo 64-bit
    // per pasangan (lihat catatan di kernel di atas), dan grid tetap di-cap oleh caller.
    for (size_t i_a = (size_t)blockIdx.x; i_a < size_A; i_a += (size_t)gridDim.x) {
        const uint64_t a_ps = d_aps[i_a];
        const TerEntry u = d_A[i_a];
        for (size_t i_b = (size_t)threadIdx.x; i_b < size_B; i_b += (size_t)blockDim.x) {
            const uint64_t req = (target_mod - a_ps - d_bps[i_b]) & mask_m1;
            uint32_t bucket = (uint32_t)(req >> shift);
            uint32_t end = d_start[bucket + 1];
            uint32_t lo = d_start[bucket];
            uint32_t hi = end;
            while (lo < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                if (d_ckeys[mid] < req) lo = mid + 1; else hi = mid;
            }
            if (lo >= end || d_ckeys[lo] != req) continue;

            const TerEntry v = d_B[i_b];
            for (uint32_t k = lo; k < end && d_ckeys[k] == req; ++k) {
                const TerEntry w = d_C[k];
                TerEntry combined;
                if (check_and_add_ternary_gpu(u, v, w, combined)) {
                    unsigned long long out_pos = atomicAdd(d_out_count, 1ULL);
                    if (out_pos < max_out_capacity) d_out[out_pos] = combined;
                }
            }
        }
    }
}

namespace {

using clk = std::chrono::high_resolution_clock;
inline double ms_since(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

bool cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::cerr << "[GPU] " << what << " gagal: " << cudaGetErrorString(e) << "\n"; return false; }
    return true;
}

struct DevBuf {
    void* p = nullptr; size_t cap = 0;
    ~DevBuf() { release(); }
    void release() { if (p) cudaFree(p); p = nullptr; cap = 0; }
    bool ensure(size_t bytes, const char* what) {
        if (bytes <= cap) return true;
        release();
        size_t want = bytes + bytes / 4;
        if (!cuda_ok(cudaMalloc(&p, want), what)) { p = nullptr; return false; }
        cap = want; return true;
    }
};

struct PinBuf {
    void* p = nullptr; size_t cap = 0;
    ~PinBuf() { release(); }
    void release() { if (p) cudaFreeHost(p); p = nullptr; cap = 0; }
    bool ensure(size_t bytes, const char* what) {
        if (bytes <= cap) return true;
        release();
        size_t want = bytes + bytes / 4;
        if (!cuda_ok(cudaMallocHost(&p, want), what)) { p = nullptr; return false; }
        cap = want; return true;
    }
};

struct Session {
    DevBuf dA, dB, dC, dOut, dCnt, dAps, dBps, dCk, dSt;
    PinBuf hA, hB, hC, hAps, hBps, hCk, hSt, hCnt;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    bool inited = false, pending = false;
    size_t pend_cap = 0;
    clk::time_point t_launch;
    GpuMergeStats stats;
    bool first_done = false;
    double first_ms = 0.0;
};
Session g;

bool session_init() {
    if (g.inited) return true;
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (!cuda_ok(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "stream")) return false;
    for (auto& e : g.ev) if (!cuda_ok(cudaEventCreate(&e), "event")) return false;
    g.inited = true;
    return true;
}

} // namespace

GpuMergeStats gpu_merge_stats_get() { GpuMergeStats s = g.stats; s.first_call_ms = g.first_ms; return s; }
void gpu_merge_stats_reset() { g.stats = GpuMergeStats{}; g.first_done = false; g.first_ms = 0.0; }

void gpu_l1_release() {
    if (g.pending && g.stream) cudaStreamSynchronize(g.stream);
    g.pending = false;
    g.dA.release(); g.dB.release(); g.dC.release(); g.dOut.release(); g.dCnt.release();
    g.dAps.release(); g.dBps.release(); g.dCk.release(); g.dSt.release();
    g.hA.release(); g.hB.release(); g.hC.release(); g.hAps.release(); g.hBps.release(); g.hCk.release(); g.hSt.release(); g.hCnt.release();
    for (auto& e : g.ev) { if (e) cudaEventDestroy(e); e = nullptr; }
    if (g.stream) cudaStreamDestroy(g.stream);
    g.stream = nullptr; g.inited = false;
}

void gpu_l1_abort() {
    if (g.pending) { cudaStreamSynchronize(g.stream); g.pending = false; }
}

bool gpu_l1_launch(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                   const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                   size_t max_cap, const Level1Sidecar* sidecar, int block_size, size_t a_count)
{
    if (A.empty() || B.empty() || sorted_C.empty()) return false;
    if (a_count == (size_t)-1 || a_count > A.size()) a_count = A.size();
    if (a_count == 0) return false;
    if (g.pending) gpu_l1_abort();
    const auto t0 = clk::now();
    if (!session_init()) return false;

    // NOTE (Bug fix): hanya A[0..a_count) yang dikirim ke GPU -- ini HARUS sama
    // dengan `gpu_rows` yang dipakai caller (merge_level1 di ter_solver.cpp) untuk
    // menghitung jatah baris CPU (A[a_count..A.size())), atau baris yang sama akan
    // diproses dobel oleh CPU dan GPU sekaligus. Sebelumnya fungsi ini selalu memakai
    // A.size() penuh, sehingga total_pairs/grid dihitung dari SELURUH A meski hanya
    // sebagian yang seharusnya jadi jatah GPU -- untuk n besar (mis. n=96, |A|~1.6M)
    // ini membuat grid meledak jauh di atas batas CUDA (2^31-1) dan launch selalu
    // gagal, sehingga GPU tidak pernah benar-benar terpakai.
    if (a_count > SIZE_MAX / B.size()) { std::cerr << "[GPU] |A|*|B| overflow\n"; return false; }
    if (block_size < 32 || block_size > 1024 || (block_size % 32) != 0) block_size = 256;

    // Mapping 2D: blockIdx.x menunjuk baris A (i_a), threadIdx.x menyapu kolom B (i_b),
    // jadi grid TIDAK perlu sebesar total_pairs/block_size. Cukup satu blok per baris A
    // (plus stride gridDim.x di kernel untuk baris sisa), di-kap ke kMaxGridBlocks agar
    // tidak melebihi batas gridDim.x (2^31-1) dan tidak ada overhead scheduling blok
    // sia-sia untuk grid raksasa. Sebelumnya grid dihitung dari total_pairs yang bisa
    // ~10^12+ untuk n=96 → di luar batas grid CUDA dan launch gagal.
    constexpr size_t kMaxGridBlocks = 131072;
    size_t grid = std::min(a_count, kMaxGridBlocks);
    if (grid == 0) grid = 1;

    const bool bucket = (sidecar && sidecar->valid && sidecar->a_ps.size() == A.size() &&
                         sidecar->b_ps.size() == B.size() && sidecar->c_keys.size() == sorted_C.size() &&
                         !sidecar->start.empty());

    const size_t bA = a_count * sizeof(TerEntry), bB = B.size() * sizeof(TerEntry), bC = sorted_C.size() * sizeof(TerEntry);
    const size_t bOut = max_cap * sizeof(TerEntry);
    const size_t bAps = a_count * 8, bBps = B.size() * 8, bCk = sorted_C.size() * 8;
    const size_t bSt = bucket ? sidecar->start.size() * 4 : 0;

    const auto ta = clk::now();
    if (!g.dA.ensure(bA, "dA") || !g.dB.ensure(bB, "dB") || !g.dC.ensure(bC, "dC") || !g.dOut.ensure(bOut, "dOut") ||
        !g.dCnt.ensure(8, "dCnt") || !g.hA.ensure(bA, "hA") || !g.hB.ensure(bB, "hB") || !g.hC.ensure(bC, "hC") ||
        !g.hCnt.ensure(8, "hCnt")) return false;
    if (bucket) {
        if (!g.dAps.ensure(bAps, "dAps") || !g.dBps.ensure(bBps, "dBps") || !g.dCk.ensure(bCk, "dCk") || !g.dSt.ensure(bSt, "dSt") ||
            !g.hAps.ensure(bAps, "hAps") || !g.hBps.ensure(bBps, "hBps") || !g.hCk.ensure(bCk, "hCk") || !g.hSt.ensure(bSt, "hSt")) return false;
    }
    g.stats.alloc_ms += ms_since(ta, clk::now());

    // A dan a_ps dipotong ke a_count baris pertama (jatah GPU); B dan sorted_C tetap utuh
    // karena keduanya dipakai penuh baik oleh GPU maupun jatah CPU (lihat merge_level1).
    std::memcpy(g.hA.p, A.data(), bA);
    std::memcpy(g.hB.p, B.data(), bB);
    std::memcpy(g.hC.p, sorted_C.data(), bC);
    if (bucket) {
        std::memcpy(g.hAps.p, sidecar->a_ps.data(), bAps);
        std::memcpy(g.hBps.p, sidecar->b_ps.data(), bBps);
        std::memcpy(g.hCk.p, sidecar->c_keys.data(), bCk);
        std::memcpy(g.hSt.p, sidecar->start.data(), bSt);
    }

    cudaStream_t s = g.stream;
    cudaEventRecord(g.ev[0], s);
    if (!cuda_ok(cudaMemcpyAsync(g.dA.p, g.hA.p, bA, cudaMemcpyHostToDevice, s), "h2d A")) return false;
    if (!cuda_ok(cudaMemcpyAsync(g.dB.p, g.hB.p, bB, cudaMemcpyHostToDevice, s), "h2d B")) return false;
    if (!cuda_ok(cudaMemcpyAsync(g.dC.p, g.hC.p, bC, cudaMemcpyHostToDevice, s), "h2d C")) return false;
    if (!cuda_ok(cudaMemsetAsync(g.dCnt.p, 0, 8, s), "memset count")) return false;
    if (bucket) {
        if (!cuda_ok(cudaMemcpyAsync(g.dAps.p, g.hAps.p, bAps, cudaMemcpyHostToDevice, s), "h2d aps")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.dBps.p, g.hBps.p, bBps, cudaMemcpyHostToDevice, s), "h2d bps")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.dCk.p, g.hCk.p, bCk, cudaMemcpyHostToDevice, s), "h2d ckeys")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.dSt.p, g.hSt.p, bSt, cudaMemcpyHostToDevice, s), "h2d start")) return false;
    }
    cudaEventRecord(g.ev[1], s);

    auto* dOut = static_cast<TerEntry*>(g.dOut.p);
    auto* dCnt = static_cast<unsigned long long*>(g.dCnt.p);
    if (bucket) {
        level1_merge_kernel_bucket<<<(unsigned)grid, (unsigned)block_size, 0, s>>>(
            (const TerEntry*)g.dA.p, a_count, (const TerEntry*)g.dB.p, B.size(), (const TerEntry*)g.dC.p,
            (const uint64_t*)g.dAps.p, (const uint64_t*)g.dBps.p, (const uint64_t*)g.dCk.p, (const uint32_t*)g.dSt.p,
            sidecar->shift, target_mod, mask_m1, dOut, dCnt, max_cap);
    } else {
        level1_merge_kernel<<<(unsigned)grid, (unsigned)block_size, 0, s>>>(
            (const TerEntry*)g.dA.p, a_count, (const TerEntry*)g.dB.p, B.size(), (const TerEntry*)g.dC.p, sorted_C.size(),
            target_mod, mask_m1, dOut, dCnt, max_cap);
    }
    if (!cuda_ok(cudaGetLastError(), "kernel launch")) return false;
    cudaEventRecord(g.ev[2], s);
    if (!cuda_ok(cudaMemcpyAsync(g.hCnt.p, g.dCnt.p, 8, cudaMemcpyDeviceToHost, s), "d2h count")) return false;
    cudaEventRecord(g.ev[3], s);

    g.pending = true;
    g.pend_cap = max_cap;
    g.t_launch = t0;
    return true;
}

bool gpu_l1_finish(std::vector<TerEntry>& L1_out) {
    if (!g.pending) return false;
    const auto tw = clk::now();
    cudaError_t e = cudaStreamSynchronize(g.stream);
    g.stats.wait_ms += ms_since(tw, clk::now());
    g.pending = false;
    if (!cuda_ok(e, "stream sync")) return false;

    unsigned long long h_count = *static_cast<unsigned long long*>(g.hCnt.p);
    const size_t actual = std::min((size_t)h_count, g.pend_cap);
    L1_out.resize(actual);
    if (actual > 0) {
        if (!cuda_ok(cudaMemcpyAsync(L1_out.data(), g.dOut.p, actual * sizeof(TerEntry), cudaMemcpyDeviceToHost, g.stream), "d2h out")) return false;
        if (!cuda_ok(cudaStreamSynchronize(g.stream), "sync out")) return false;
    }

    float f = 0.f;
    if (cudaEventElapsedTime(&f, g.ev[0], g.ev[1]) == cudaSuccess) g.stats.h2d_ms += f;
    if (cudaEventElapsedTime(&f, g.ev[1], g.ev[2]) == cudaSuccess) g.stats.kernel_ms += f;
    if (cudaEventElapsedTime(&f, g.ev[2], g.ev[3]) == cudaSuccess) g.stats.d2h_ms += f;
    ++g.stats.calls;
    if (!g.first_done) { g.first_done = true; g.first_ms = ms_since(g.t_launch, clk::now()); }
    return true;
}

bool run_level1_merge_gpu(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                          const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                          size_t max_cap, std::vector<TerEntry>& L1_out, const Level1Sidecar* sidecar, int block_size)
{
    if (!gpu_l1_launch(A, B, sorted_C, target_mod, mask_m1, max_cap, sidecar, block_size)) return false;
    return gpu_l1_finish(L1_out);
}

#endif
