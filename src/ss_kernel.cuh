#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef WITH_GPU
bool gpu_sort_weights_with_payload(std::vector<uint64_t>& weights, std::vector<size_t>& subsets, bool ascending);
#endif