#pragma once
#include "ter_solver.cuh"
#include <cstddef>
#include <vector>

#ifdef WITH_GPU

struct GpuMergeStats {
    size_t calls = 0;
    size_t pref_calls = 0;      // job yang di-launch dari prefetch double-buffer (P3)
    double first_call_ms = 0.0;
    double alloc_ms = 0.0, h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
    double wait_ms = 0.0;   // waktu host terblokir menunggu GPU di gpu_l1_finish
};
GpuMergeStats gpu_merge_stats_get();
void gpu_merge_stats_reset();

// Async: launch mengembalikan segera; finish menunggu dan menyalin hasil. Satu job in-flight.
// a_count: jumlah baris A[0..a_count) yang diproses GPU (bagian awal A); sisanya
// A[a_count..A.size()) adalah jatah CPU (lihat merge_level1 di ter_solver.cpp). Wajib
// <= A.size(); default A.size() untuk kompatibilitas caller yang memproses seluruh A.
bool gpu_l1_launch(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                   const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                   size_t max_cap, const Level1Sidecar* sidecar, int block_size,
                   size_t a_count = (size_t)-1);
// P3: stage H2D untuk JOB BERIKUTNYA ke set buffer cadangan sambil kernel saat ini
// jalan. gpu_l1_launch berikutnya akan memakai data ini tanpa menunggu copy saat itu.
// Param identik dengan gpu_l1_launch. Return false bila input tak valid.
bool gpu_l1_prefetch(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                     const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                     size_t max_cap, const Level1Sidecar* sidecar, int block_size,
                     size_t a_count = (size_t)-1);
bool gpu_l1_finish(std::vector<TerEntry>& L1_out);
void gpu_l1_abort();
void gpu_l1_release();

bool run_level1_merge_gpu(const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
                          const std::vector<TerEntry>& sorted_C, uint64_t target_mod, uint64_t mask_m1,
                          size_t max_cap, std::vector<TerEntry>& L1_out,
                          const Level1Sidecar* sidecar = nullptr, int block_size = 256);
#endif
