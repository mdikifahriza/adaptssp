// ter_kernel.cu — GPU Kernel for Level 1 Merge in TER Subset Sum Solver
// Target Platform: NVIDIA Tesla T4 (sm_75) and modern CUDA architectures.

#include "ter_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
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

bool run_level1_merge_gpu(
    const std::vector<TerEntry>& A,
    const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C,
    uint64_t target_mod,
    uint64_t mask_m1,
    size_t max_cap,
    std::vector<TerEntry>& L1_out
) {
    if (A.empty() || B.empty() || sorted_C.empty()) return false;

    TerEntry* d_A = nullptr;
    TerEntry* d_B = nullptr;
    TerEntry* d_C = nullptr;
    TerEntry* d_out = nullptr;
    unsigned long long* d_out_count = nullptr;

    size_t size_A_bytes = A.size() * sizeof(TerEntry);
    size_t size_B_bytes = B.size() * sizeof(TerEntry);
    size_t size_C_bytes = sorted_C.size() * sizeof(TerEntry);
    size_t size_out_bytes = max_cap * sizeof(TerEntry);

    if (cudaMalloc(&d_A, size_A_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_B, size_B_bytes) != cudaSuccess) { cudaFree(d_A); return false; }
    if (cudaMalloc(&d_C, size_C_bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); return false; }
    if (cudaMalloc(&d_out, size_out_bytes) != cudaSuccess) { cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return false; }
    if (cudaMalloc(&d_out_count, sizeof(unsigned long long)) != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_out); return false;
    }

    cudaMemcpy(d_A, A.data(), size_A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_B_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, sorted_C.data(), size_C_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out_count, 0, sizeof(unsigned long long));

    size_t total_pairs = A.size() * B.size();
    int block_size = 256;
    int grid_size = (int)((total_pairs + block_size - 1) / block_size);

    level1_merge_kernel<<<grid_size, block_size>>>(
        d_A, A.size(), d_B, B.size(), d_C, sorted_C.size(),
        target_mod, mask_m1, d_out, d_out_count, max_cap
    );
    cudaDeviceSynchronize();

    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    size_t actual_out = std::min((size_t)h_count, max_cap);

    L1_out.resize(actual_out);
    if (actual_out > 0) {
        cudaMemcpy(L1_out.data(), d_out, actual_out * sizeof(TerEntry), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_out);
    cudaFree(d_out_count);

    return true;
}

#endif
