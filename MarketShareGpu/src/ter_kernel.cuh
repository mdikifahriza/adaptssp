#pragma once
#include "ter_solver.cuh"
#include <vector>

#ifdef WITH_GPU
bool run_level1_merge_gpu(
    const std::vector<TerEntry>& A,
    const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C,
    uint64_t target_mod,
    uint64_t mask_m1,
    size_t max_cap,
    std::vector<TerEntry>& L1_out
);
#endif
