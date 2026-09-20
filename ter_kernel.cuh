#pragma once
#include "ter_solver.cuh"
#include <cstddef>
#include <vector>

#ifdef WITH_GPU

// Statistik waktu GPU kumulatif (langkah 4b), diakumulasi oleh run_level1_merge_gpu.
// Dibaca ter_solve() untuk ringkasan profil; di-reset di awal tiap ter_solve().
struct GpuMergeStats {
    size_t calls = 0;            // panggilan yang sukses
    double first_call_ms = 0.0;  // seluruh panggilan pertama (termasuk inisialisasi konteks CUDA)
    double alloc_ms = 0.0;       // cudaMalloc (jam dinding)
    double h2d_ms = 0.0;         // salinan host->device (cudaEvent)
    double kernel_ms = 0.0;      // kernel (cudaEvent)
    double d2h_ms = 0.0;         // salinan device->host (cudaEvent)
    double free_ms = 0.0;        // cudaFree (jam dinding)
};
GpuMergeStats gpu_merge_stats_get();
void gpu_merge_stats_reset();

// Level 1 merge di GPU. Mengembalikan false bila gagal (alokasi, launch, atau sync);
// penyebabnya dicetak ke stderr dengan awalan "[GPU]" dan pemanggil jatuh ke CPU.
// `sidecar` == nullptr -> kernel lama (binary search di d_C).
// `sidecar` != nullptr dan valid -> kernel bucket (hasil sama, akses memori lebih ringan).
bool run_level1_merge_gpu(
    const std::vector<TerEntry>& A,
    const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C,
    uint64_t target_mod,
    uint64_t mask_m1,
    size_t max_cap,
    std::vector<TerEntry>& L1_out,
    const Level1Sidecar* sidecar = nullptr
);
#endif
