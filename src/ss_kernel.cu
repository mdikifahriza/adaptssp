#include "ss_kernel.cuh"
#ifdef WITH_GPU
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

namespace {
bool cuda_ok2(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::cerr << "[GPU-SS] " << what << " gagal: " << cudaGetErrorString(e) << "\n"; return false; }
    return true;
}
}

bool gpu_sort_weights_with_payload(std::vector<uint64_t>& weights, std::vector<size_t>& subsets, bool ascending) {
    const size_t n = weights.size();
    if (n == 0 || n != subsets.size()) return n == subsets.size();
    if (n > (1ULL << 32)) { std::cerr << "[GPU-SS] n terlalu besar untuk sort GPU\n"; return false; }

    std::vector<uint64_t> payload(n);
    for (size_t i = 0; i < n; ++i) payload[i] = (uint64_t)subsets[i];

    uint64_t *d_keys_in = nullptr, *d_keys_out = nullptr, *d_vals_in = nullptr, *d_vals_out = nullptr;
    void* d_temp = nullptr; size_t temp_bytes = 0;
    bool ok = true;
    ok = ok && cuda_ok2(cudaMalloc(&d_keys_in, n * 8), "malloc keys_in");
    ok = ok && cuda_ok2(cudaMalloc(&d_keys_out, n * 8), "malloc keys_out");
    ok = ok && cuda_ok2(cudaMalloc(&d_vals_in, n * 8), "malloc vals_in");
    ok = ok && cuda_ok2(cudaMalloc(&d_vals_out, n * 8), "malloc vals_out");
    if (!ok) { cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); return false; }

    cudaMemcpy(d_keys_in, weights.data(), n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals_in, payload.data(), n * 8, cudaMemcpyHostToDevice);

    cudaError_t e;
    if (ascending) e = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, (int)n);
    else           e = cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, (int)n);
    if (!cuda_ok2(e, "cub sizing")) { cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); return false; }

    if (!cuda_ok2(cudaMalloc(&d_temp, temp_bytes), "malloc temp")) {
        cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); return false;
    }
    if (ascending) e = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, (int)n);
    else           e = cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, (int)n);
    ok = cuda_ok2(e, "cub sort") && cuda_ok2(cudaDeviceSynchronize(), "sync sort");

    if (ok) {
        cudaMemcpy(weights.data(), d_keys_out, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(payload.data(), d_vals_out, n * 8, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; ++i) subsets[i] = (size_t)payload[i];
    }
    cudaFree(d_keys_in); cudaFree(d_keys_out); cudaFree(d_vals_in); cudaFree(d_vals_out); cudaFree(d_temp);
    return ok;
}
#endif
