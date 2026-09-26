#include "ter_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

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

__device__ inline bool check_root_sum_gpu(const TerEntry& u, const TerEntry& v, const TerEntry& w, TerEntry& out) {
    if (!check_and_add_ternary_gpu(u, v, w, out)) return false;
    return (out.neg_lo == 0 && out.neg_hi == 0);
}

__global__ void root_merge_kernel_bucket(
    const TerEntry* __restrict__ d_A, size_t size_A, const TerEntry* __restrict__ d_B, size_t size_B,
    const TerEntry* __restrict__ d_C, const uint64_t* __restrict__ d_aps, const uint64_t* __restrict__ d_bps,
    const uint64_t* __restrict__ d_ckeys, const uint32_t* __restrict__ d_start, int shift,
    uint64_t target_exact, TerEntry* __restrict__ d_out,
    unsigned long long* __restrict__ d_out_count, size_t max_out_capacity, int* __restrict__ d_found_flag)
{
    for (size_t i_a = (size_t)blockIdx.x; i_a < size_A; i_a += (size_t)gridDim.x) {
        if (*d_found_flag) return;
        const uint64_t a_ps = d_aps[i_a];
        const TerEntry u = d_A[i_a];
        for (size_t i_b = (size_t)threadIdx.x; i_b < size_B; i_b += (size_t)blockDim.x) {
            const uint64_t req = target_exact - a_ps - d_bps[i_b];
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
                if (check_root_sum_gpu(u, v, w, combined)) {
                    unsigned long long out_pos = atomicAdd(d_out_count, 1ULL);
                    if (out_pos < max_out_capacity) d_out[out_pos] = combined;
                    atomicExch(d_found_flag, 1);
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

    DevBuf pdA, pdB, pdC, pdOut, pdCnt, pdAps, pdBps, pdCk, pdSt;
    PinBuf phA, phB, phC, phAps, phBps, phCk, phSt, phCnt;
    cudaStream_t pref_stream = nullptr;
    cudaEvent_t pref_ev[2] = {nullptr, nullptr};
    cudaEvent_t pref_done = nullptr;
    bool pref_ready = false;
    bool pref_used = false;

    struct PrefetchMeta {
        size_t a_count = 0, b_count = 0, c_count = 0, max_cap = 0;
        uint64_t target_mod = 0, mask_m1 = 0;
        int block_size = 256, shift = 0;
        bool bucket = false;
    } pref_meta;

    bool inited = false, pending = false;
    size_t pend_cap = 0;
    clk::time_point t_launch;
    GpuMergeStats stats;
    bool first_done = false;
    double first_ms = 0.0;
};
Session g;

template <class Buf>
void swap_buf(Buf& a, Buf& b) { std::swap(a.p, b.p); std::swap(a.cap, b.cap); }

void session_swap_banks(Session& s) {
    swap_buf(s.dA, s.pdA); swap_buf(s.dB, s.pdB); swap_buf(s.dC, s.pdC);
    swap_buf(s.dOut, s.pdOut); swap_buf(s.dCnt, s.pdCnt);
    swap_buf(s.dAps, s.pdAps); swap_buf(s.dBps, s.pdBps); swap_buf(s.dCk, s.pdCk); swap_buf(s.dSt, s.pdSt);
    swap_buf(s.hA, s.phA); swap_buf(s.hB, s.phB); swap_buf(s.hC, s.phC);
    swap_buf(s.hAps, s.phAps); swap_buf(s.hBps, s.phBps); swap_buf(s.hCk, s.phCk); swap_buf(s.hSt, s.phSt);
    swap_buf(s.hCnt, s.phCnt);
}

bool session_init() {
    if (g.inited) return true;
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (!cuda_ok(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking), "stream")) return false;
    if (!cuda_ok(cudaStreamCreateWithFlags(&g.pref_stream, cudaStreamNonBlocking), "pref stream")) return false;
    for (auto& e : g.ev) if (!cuda_ok(cudaEventCreate(&e), "event")) return false;
    for (auto& e : g.pref_ev) if (!cuda_ok(cudaEventCreate(&e), "pref event")) return false;
    if (!cuda_ok(cudaEventCreate(&g.pref_done), "pref done event")) return false;
    g.inited = true;
    return true;
}

void session_release_all(Session& s) {
    if (s.pending && s.stream) cudaStreamSynchronize(s.stream);
    if (s.pref_stream) cudaStreamSynchronize(s.pref_stream);
    s.pending = false; s.pref_ready = false;
    s.dA.release(); s.dB.release(); s.dC.release(); s.dOut.release(); s.dCnt.release();
    s.dAps.release(); s.dBps.release(); s.dCk.release(); s.dSt.release();
    s.hA.release(); s.hB.release(); s.hC.release(); s.hAps.release(); s.hBps.release(); s.hCk.release(); s.hSt.release(); s.hCnt.release();
    s.pdA.release(); s.pdB.release(); s.pdC.release(); s.pdOut.release(); s.pdCnt.release();
    s.pdAps.release(); s.pdBps.release(); s.pdCk.release(); s.pdSt.release();
    s.phA.release(); s.phB.release(); s.phC.release(); s.phAps.release(); s.phBps.release(); s.phCk.release(); s.phSt.release(); s.phCnt.release();
    for (auto& e : s.ev) { if (e) cudaEventDestroy(e); e = nullptr; }
    for (auto& e : s.pref_ev) { if (e) cudaEventDestroy(e); e = nullptr; }
    if (s.pref_done) { cudaEventDestroy(s.pref_done); s.pref_done = nullptr; }
    if (s.stream) cudaStreamDestroy(s.stream);
    if (s.pref_stream) cudaStreamDestroy(s.pref_stream);
    s.stream = nullptr; s.pref_stream = nullptr; s.inited = false;
}

}

GpuMergeStats gpu_merge_stats_get() { GpuMergeStats s = g.stats; s.first_call_ms = g.first_ms; return s; }
void gpu_merge_stats_reset() { g.stats = GpuMergeStats{}; g.first_done = false; g.first_ms = 0.0; }

void gpu_l1_release() { session_release_all(g); }

void gpu_l1_abort() {
    if (g.pending) { cudaStreamSynchronize(g.stream); g.pending = false; }
}

bool gpu_l1_prefetch(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                     const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                     size_t max_cap, const Level1Sidecar* sidecar, int block_size, size_t a_count)
{
    if (A.empty() || B.empty() || sorted_C.empty()) return false;
    if (a_count == (size_t)-1 || a_count > A.size()) a_count = A.size();
    if (a_count == 0) return false;
    if (!session_init()) return false;
    if (g.pref_ready) return false;
    if (a_count > SIZE_MAX / B.size()) { std::cerr << "[GPU] |A|*|B| overflow (prefetch)\n"; return false; }
    if (block_size < 32 || block_size > 1024 || (block_size % 32) != 0) block_size = 256;

    const bool bucket = (sidecar && sidecar->valid && sidecar->a_ps.size() == A.size() &&
                         sidecar->b_ps.size() == B.size() && sidecar->c_keys.size() == sorted_C.size() &&
                         !sidecar->start.empty());

    const size_t bA = a_count * sizeof(TerEntry), bB = B.size() * sizeof(TerEntry), bC = sorted_C.size() * sizeof(TerEntry);
    const size_t bOut = max_cap * sizeof(TerEntry);
    const size_t bAps = a_count * 8, bBps = B.size() * 8, bCk = sorted_C.size() * 8;
    const size_t bSt = bucket ? sidecar->start.size() * 4 : 0;

    if (!g.pdA.ensure(bA, "pdA") || !g.pdB.ensure(bB, "pdB") || !g.pdC.ensure(bC, "pdC") || !g.pdOut.ensure(bOut, "pdOut") ||
        !g.pdCnt.ensure(8, "pdCnt") || !g.phA.ensure(bA, "phA") || !g.phB.ensure(bB, "phB") || !g.phC.ensure(bC, "phC") ||
        !g.phCnt.ensure(8, "phCnt")) return false;
    if (bucket) {
        if (!g.pdAps.ensure(bAps, "pdAps") || !g.pdBps.ensure(bBps, "pdBps") || !g.pdCk.ensure(bCk, "pdCk") || !g.pdSt.ensure(bSt, "pdSt") ||
            !g.phAps.ensure(bAps, "phAps") || !g.phBps.ensure(bBps, "phBps") || !g.phCk.ensure(bCk, "phCk") || !g.phSt.ensure(bSt, "phSt")) return false;
    }

    std::memcpy(g.phA.p, A.data(), bA);
    std::memcpy(g.phB.p, B.data(), bB);
    std::memcpy(g.phC.p, sorted_C.data(), bC);
    if (bucket) {
        std::memcpy(g.phAps.p, sidecar->a_ps.data(), bAps);
        std::memcpy(g.phBps.p, sidecar->b_ps.data(), bBps);
        std::memcpy(g.phCk.p, sidecar->c_keys.data(), bCk);
        std::memcpy(g.phSt.p, sidecar->start.data(), bSt);
    }

    cudaStream_t ps = g.pref_stream;
    cudaEventRecord(g.pref_ev[0], ps);
    if (!cuda_ok(cudaMemcpyAsync(g.pdA.p, g.phA.p, bA, cudaMemcpyHostToDevice, ps), "pref h2d A")) return false;
    if (!cuda_ok(cudaMemcpyAsync(g.pdB.p, g.phB.p, bB, cudaMemcpyHostToDevice, ps), "pref h2d B")) return false;
    if (!cuda_ok(cudaMemcpyAsync(g.pdC.p, g.phC.p, bC, cudaMemcpyHostToDevice, ps), "pref h2d C")) return false;
    if (!cuda_ok(cudaMemsetAsync(g.pdCnt.p, 0, 8, ps), "pref memset cnt")) return false;
    if (bucket) {
        if (!cuda_ok(cudaMemcpyAsync(g.pdAps.p, g.phAps.p, bAps, cudaMemcpyHostToDevice, ps), "pref h2d aps")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.pdBps.p, g.phBps.p, bBps, cudaMemcpyHostToDevice, ps), "pref h2d bps")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.pdCk.p, g.phCk.p, bCk, cudaMemcpyHostToDevice, ps), "pref h2d ckeys")) return false;
        if (!cuda_ok(cudaMemcpyAsync(g.pdSt.p, g.phSt.p, bSt, cudaMemcpyHostToDevice, ps), "pref h2d start")) return false;
    }
    cudaEventRecord(g.pref_ev[1], ps);
    cudaEventRecord(g.pref_done, ps);

    g.pref_meta = {a_count, B.size(), sorted_C.size(), max_cap, target_mod, mask_m1,
                   block_size, bucket ? sidecar->shift : 0, bucket};
    g.pref_ready = true;
    return true;
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

    if (a_count > SIZE_MAX / B.size()) { std::cerr << "[GPU] |A|*|B| overflow\n"; return false; }
    if (block_size < 32 || block_size > 1024 || (block_size % 32) != 0) block_size = 256;

    const bool bucket = (sidecar && sidecar->valid && sidecar->a_ps.size() == A.size() &&
                         sidecar->b_ps.size() == B.size() && sidecar->c_keys.size() == sorted_C.size() &&
                         !sidecar->start.empty());

    bool used_prefetch = false;
    if (g.pref_ready) {
        const auto& m = g.pref_meta;
        if (m.a_count == a_count && m.b_count == B.size() && m.c_count == sorted_C.size() &&
            m.max_cap == max_cap && m.target_mod == target_mod && m.mask_m1 == mask_m1 &&
            m.block_size == block_size && m.bucket == bucket) {
            session_swap_banks(g);
            g.pref_ready = false;
            if (!cuda_ok(cudaStreamWaitEvent(g.stream, g.pref_done, 0), "stream wait pref")) return false;
            float pf = 0.f;
            if (cudaEventElapsedTime(&pf, g.pref_ev[0], g.pref_ev[1]) == cudaSuccess) g.stats.h2d_ms += pf;
            ++g.stats.pref_calls;
            used_prefetch = true;
        } else {
            cudaStreamSynchronize(g.pref_stream);
            g.pref_ready = false;
        }
    }

    constexpr size_t kMaxGridBlocks = 131072;
    size_t grid = std::min(a_count, kMaxGridBlocks);
    if (grid == 0) grid = 1;

    const size_t bA = a_count * sizeof(TerEntry), bB = B.size() * sizeof(TerEntry), bC = sorted_C.size() * sizeof(TerEntry);
    const size_t bOut = max_cap * sizeof(TerEntry);
    const size_t bAps = a_count * 8, bBps = B.size() * 8, bCk = sorted_C.size() * 8;
    const size_t bSt = bucket ? sidecar->start.size() * 4 : 0;

    const auto ta = clk::now();
    if (!used_prefetch) {
        if (!g.dA.ensure(bA, "dA") || !g.dB.ensure(bB, "dB") || !g.dC.ensure(bC, "dC") || !g.dOut.ensure(bOut, "dOut") ||
            !g.dCnt.ensure(8, "dCnt") || !g.hA.ensure(bA, "hA") || !g.hB.ensure(bB, "hB") || !g.hC.ensure(bC, "hC") ||
            !g.hCnt.ensure(8, "hCnt")) return false;
        if (bucket) {
            if (!g.dAps.ensure(bAps, "dAps") || !g.dBps.ensure(bBps, "dBps") || !g.dCk.ensure(bCk, "dCk") || !g.dSt.ensure(bSt, "dSt") ||
                !g.hAps.ensure(bAps, "hAps") || !g.hBps.ensure(bBps, "hBps") || !g.hCk.ensure(bCk, "hCk") || !g.hSt.ensure(bSt, "hSt")) return false;
        }
        g.stats.alloc_ms += ms_since(ta, clk::now());

        std::memcpy(g.hA.p, A.data(), bA);
        std::memcpy(g.hB.p, B.data(), bB);
        std::memcpy(g.hC.p, sorted_C.data(), bC);
        if (bucket) {
            std::memcpy(g.hAps.p, sidecar->a_ps.data(), bAps);
            std::memcpy(g.hBps.p, sidecar->b_ps.data(), bBps);
            std::memcpy(g.hCk.p, sidecar->c_keys.data(), bCk);
            std::memcpy(g.hSt.p, sidecar->start.data(), bSt);
        }
    }

    cudaStream_t s = g.stream;
    if (!used_prefetch) {
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
    } else {
        cudaEventRecord(g.ev[0], s);
        cudaEventRecord(g.ev[1], s);
    }

    auto* dOut = static_cast<TerEntry*>(g.dOut.p);
    auto* dCnt = static_cast<unsigned long long*>(g.dCnt.p);
    const int shift = used_prefetch ? g.pref_meta.shift : (bucket ? sidecar->shift : 0);
    if (bucket) {
        level1_merge_kernel_bucket<<<(unsigned)grid, (unsigned)block_size, 0, s>>>(
            (const TerEntry*)g.dA.p, a_count, (const TerEntry*)g.dB.p, B.size(), (const TerEntry*)g.dC.p,
            (const uint64_t*)g.dAps.p, (const uint64_t*)g.dBps.p, (const uint64_t*)g.dCk.p, (const uint32_t*)g.dSt.p,
            shift, target_mod, mask_m1, dOut, dCnt, max_cap);
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

namespace {
struct RootSession {
    DevBuf dA, dB, dC, dOut, dCnt, dAps, dBps, dCk, dSt, dFound;
    bool inited = false;
};
RootSession gr;

bool root_session_init() {
    if (gr.inited) return true;
    gr.inited = true;
    return true;
}
}

bool run_root_merge_gpu(
    const std::vector<TerEntry>& A, const std::vector<TerEntry>& B, const std::vector<TerEntry>& sorted_C,
    uint64_t target_exact, size_t max_cap, const Level1Sidecar& sidecar,
    std::vector<TerEntry>& out, int block_size)
{
    if (A.empty() || B.empty() || sorted_C.empty()) return false;
    if (!sidecar.valid || sidecar.a_ps.size() != A.size() || sidecar.b_ps.size() != B.size() ||
        sidecar.c_keys.size() != sorted_C.size() || sidecar.start.empty()) return false;
    if (!root_session_init()) return false;
    if (block_size < 32 || block_size > 1024 || (block_size % 32) != 0) block_size = 256;
    if (A.size() > SIZE_MAX / std::max<size_t>(B.size(), 1)) { std::cerr << "[GPU-root] |A|*|B| overflow\n"; return false; }

    const size_t bA = A.size() * sizeof(TerEntry), bB = B.size() * sizeof(TerEntry), bC = sorted_C.size() * sizeof(TerEntry);
    const size_t bOut = max_cap * sizeof(TerEntry);
    const size_t bAps = A.size() * 8, bBps = B.size() * 8, bCk = sorted_C.size() * 8;
    const size_t bSt = sidecar.start.size() * 4;

    if (!gr.dA.ensure(bA, "root dA") || !gr.dB.ensure(bB, "root dB") || !gr.dC.ensure(bC, "root dC") ||
        !gr.dOut.ensure(bOut, "root dOut") || !gr.dCnt.ensure(8, "root dCnt") || !gr.dFound.ensure(4, "root dFound") ||
        !gr.dAps.ensure(bAps, "root dAps") || !gr.dBps.ensure(bBps, "root dBps") ||
        !gr.dCk.ensure(bCk, "root dCk") || !gr.dSt.ensure(bSt, "root dSt")) return false;

    if (!cuda_ok(cudaMemcpy(gr.dA.p, A.data(), bA, cudaMemcpyHostToDevice), "root h2d A")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dB.p, B.data(), bB, cudaMemcpyHostToDevice), "root h2d B")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dC.p, sorted_C.data(), bC, cudaMemcpyHostToDevice), "root h2d C")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dAps.p, sidecar.a_ps.data(), bAps, cudaMemcpyHostToDevice), "root h2d aps")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dBps.p, sidecar.b_ps.data(), bBps, cudaMemcpyHostToDevice), "root h2d bps")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dCk.p, sidecar.c_keys.data(), bCk, cudaMemcpyHostToDevice), "root h2d ckeys")) return false;
    if (!cuda_ok(cudaMemcpy(gr.dSt.p, sidecar.start.data(), bSt, cudaMemcpyHostToDevice), "root h2d start")) return false;
    if (!cuda_ok(cudaMemset(gr.dCnt.p, 0, 8), "root memset cnt")) return false;
    if (!cuda_ok(cudaMemset(gr.dFound.p, 0, 4), "root memset found")) return false;

    constexpr size_t kMaxGridBlocks = 131072;
    size_t grid = std::min(A.size(), kMaxGridBlocks);
    if (grid == 0) grid = 1;

    root_merge_kernel_bucket<<<(unsigned)grid, (unsigned)block_size>>>(
        (const TerEntry*)gr.dA.p, A.size(), (const TerEntry*)gr.dB.p, B.size(), (const TerEntry*)gr.dC.p,
        (const uint64_t*)gr.dAps.p, (const uint64_t*)gr.dBps.p, (const uint64_t*)gr.dCk.p, (const uint32_t*)gr.dSt.p,
        sidecar.shift, target_exact, (TerEntry*)gr.dOut.p, (unsigned long long*)gr.dCnt.p, max_cap, (int*)gr.dFound.p);
    if (!cuda_ok(cudaGetLastError(), "root kernel launch")) return false;
    if (!cuda_ok(cudaDeviceSynchronize(), "root sync")) return false;

    unsigned long long h_count = 0;
    if (!cuda_ok(cudaMemcpy(&h_count, gr.dCnt.p, 8, cudaMemcpyDeviceToHost), "root d2h count")) return false;
    const size_t actual = std::min((size_t)h_count, max_cap);
    out.resize(actual);
    if (actual > 0) {
        if (!cuda_ok(cudaMemcpy(out.data(), gr.dOut.p, actual * sizeof(TerEntry), cudaMemcpyDeviceToHost), "root d2h out")) return false;
    }
    return true;
}

#endif