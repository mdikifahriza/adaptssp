#pragma once

#include <vector>

#include "markshare.hpp"
#include "pairs_tuple.hpp"

class GpuData
{
public:
    GpuData() = default;
    GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores);

    ~GpuData();

    int64_t n_bytes_alloc{};

    /* Required data for any computations on GPU. Is considered constant throughout the algorithm. */
    size_t *set1_scores{};
    size_t *set2_scores{};
    size_t *set3_scores{};
    size_t *set4_scores{};

    size_t *matrix{};
    size_t *rhs{};

    size_t m_rows{};
    size_t n_cols{};

    /* GPU buffers. Get resized depending on the problem. */
    size_t *required_buffer{};
    size_t len_required_buffer{}; /* Size of above buffers. */
    size_t n_required;

    size_t *search_buffer{};
    size_t len_search_buffer{};
    bool *results_search_buffer{};
    size_t len_results_buffer{};
    size_t n_search{};

    /* For new tuples approach. */
    size_t *tuples_buffer{};
    size_t len_tuples_buffer{};
    size_t n_tuples{};

    double get_gb_allocated() const
    {
        return (double)n_bytes_alloc / (1000000000);
    };

    template <typename T>
    void resize_buffer(T **buffer, size_t &buffer_size, size_t n_elems_required);

    void copy_tuples(const PairsTuple* tuples, size_t n_tuples);
    void copy_pairs_search(const std::vector<std::pair<size_t, size_t>> &pairs);
    void copy_pairs_required(const std::vector<std::pair<size_t, size_t>> &pairs);
};

void sort_required_gpu(GpuData &gpu_data);
std::vector<size_t> find_equal_hashes(GpuData &gpu_data, bool sort_required = true);
std::vector<std::pair<size_t, size_t>> find_hash_positions_gpu(GpuData &gpu_data, const std::vector<size_t>& hashes, size_t n_p1, size_t n_p2, bool encode_first_as_required = false);

void combine_and_encode_tuples_required_gpu(GpuData &gpu_data, const PairsTuple *tuples, size_t n_tuples, size_t n_pairs, const size_t *scores1, const size_t *scores2, size_t row_offset);
void combine_and_encode_tuples_search_gpu(GpuData &gpu_data, const PairsTuple *tuples, size_t n_tuples, size_t n_pairs, const size_t *scores1, const size_t *scores2, size_t row_offset);
void combine_and_encode_tuples_gpu(GpuData &gpu_data, const PairsTuple* tuples1, const PairsTuple* tuples2, size_t n_tuples1, size_t n_tuples2, size_t n_pairs1, size_t n_pairs2, size_t row_offset);
