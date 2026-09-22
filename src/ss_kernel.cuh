#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef WITH_GPU
// Langkah 6: hanya porsi yang cocok GPU (sort besar, murni data-paralel), bukan heap
// k-way merge (branchy/sequential, dibiarkan di CPU). weights[i] terurut ascending atau
// descending; subsets dipermutasi mengikuti weights. Payload uint64 (indeks subset < 2^62).
bool gpu_sort_weights_with_payload(std::vector<uint64_t>& weights, std::vector<size_t>& subsets, bool ascending);
#endif
