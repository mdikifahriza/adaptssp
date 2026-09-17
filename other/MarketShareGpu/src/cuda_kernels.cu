#include "cuda_kernels.cuh"

#include "profiler.hpp"

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <math.h>
#include <iostream>

template <typename T>
T *copy_to_device(const std::vector<T> &host_vec, int64_t &n_bytes_alloc_total)
{
    T *device_ptr;
    const auto n_bytes_alloc = host_vec.size() * sizeof(T);
    cudaMalloc(&device_ptr, n_bytes_alloc);
    cudaMemcpy(device_ptr, host_vec.data(), n_bytes_alloc, cudaMemcpyHostToDevice);

    n_bytes_alloc_total += n_bytes_alloc;
    return device_ptr;
}

GpuData::GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores) : m_rows(ms_inst.m()), n_cols(ms_inst.n())
{
    this->matrix = copy_to_device(ms_inst.A(), n_bytes_alloc);
    this->rhs = copy_to_device(ms_inst.b(), n_bytes_alloc);

    this->set1_scores = copy_to_device(set1_scores, n_bytes_alloc);
    this->set2_scores = copy_to_device(set2_scores, n_bytes_alloc);
    this->set3_scores = copy_to_device(set3_scores, n_bytes_alloc);
    this->set4_scores = copy_to_device(set4_scores, n_bytes_alloc);
}

GpuData::~GpuData()
{
    cudaFree(set1_scores);
    cudaFree(set2_scores);
    cudaFree(set3_scores);
    cudaFree(set4_scores);

    cudaFree(matrix);
    cudaFree(rhs);

    cudaFree(required_buffer);

    cudaFree(search_buffer);
    cudaFree(results_search_buffer);
}

template <typename T>
void GpuData::resize_buffer(T **buffer, size_t &buffer_size, size_t n_elems_required)
{
    if (n_elems_required > buffer_size)
    {
        size_t new_buffer_size = static_cast<size_t>(n_elems_required * 1.1 + 1);
        assert(new_buffer_size > n_elems_required);

        cudaFree(*buffer);

        n_bytes_alloc += sizeof(T) * (new_buffer_size - buffer_size);
        cudaMalloc(buffer, new_buffer_size * sizeof(T));

        buffer_size = new_buffer_size;
    }
}

void GpuData::copy_pairs_required(const std::vector<std::pair<size_t, size_t>> &pairs)
{
    size_t len_needed = pairs.size();

    resize_buffer(&required_buffer, len_required_buffer, len_needed);

    size_t *pairs_required = (size_t *)(required_buffer);
    cudaMemcpy(pairs_required, pairs.data(), pairs.size() * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    n_required = len_needed;
}

void GpuData::copy_pairs_search(const std::vector<std::pair<size_t, size_t>> &pairs)
{
    size_t len_needed = pairs.size();
    resize_buffer(&search_buffer, len_search_buffer, len_needed);
    resize_buffer(&results_search_buffer, len_results_buffer, len_needed);

    size_t *pairs_search = (size_t *)(search_buffer);
    cudaMemcpy(pairs_search, pairs.data(), pairs.size() * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
    n_search = len_needed;
}

void GpuData::copy_tuples(const PairsTuple *tuples, size_t n_tuples)
{
    size_t len_needed = n_tuples * 4;
    resize_buffer(&tuples_buffer, len_tuples_buffer, len_needed);

    cudaError_t err = cudaMemcpy(tuples_buffer, tuples, len_needed * sizeof(size_t), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    this->n_tuples = n_tuples;
}

/* Converts tuples into pairs. */
__global__ void flatten_tuples(const size_t *tuples, size_t n_tuples, size_t *pairs)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_tuples)
        return;

    const size_t *tuple = tuples + 4 * idx;
    size_t pair_first = tuple[0];
    size_t pair_second_beg = tuple[1];
    size_t pair_second_end = tuple[2] + pair_second_beg;
    size_t pairs_offset = tuple[3];

    for (size_t second = pair_second_beg; second < pair_second_end; ++second)
    {
        pairs[2 * pairs_offset] = pair_first;
        pairs[2 * pairs_offset + 1] = second;
        ++pairs_offset;
    }
}

__device__ size_t custom_hash(size_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* Converts tuples into pairs. */
template <bool ENCODE_REQUIRED>
__global__ void flatten_and_encode_tuples(const size_t *tuples, size_t n_tuples, const size_t *__restrict__ scores1, const size_t *__restrict__ scores2, const size_t *__restrict__ rhs, size_t *__restrict__ hashes, size_t m_rows, size_t row_offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t m_rows_left = m_rows - row_offset;

    if (idx >= n_tuples)
        return;

    const size_t *tuple = tuples + 4 * idx;
    size_t first = tuple[0];
    size_t pair_second_beg = tuple[1];
    size_t pair_second_end = tuple[2] + pair_second_beg;
    size_t pairs_offset = tuple[3];

    for (size_t second = pair_second_beg; second < pair_second_end; ++second)
    {
        size_t key = 0;

        /* Compute the hash of this tuple. */
        for (size_t i_row = 0; i_row < m_rows_left; ++i_row)
        {
            /* Compute the pair's score of this row and add it (encoded) to key. */
            size_t row_score = scores1[first * m_rows_left + i_row] + scores2[second * m_rows_left + i_row];

            if (ENCODE_REQUIRED)
                row_score = rhs[i_row + row_offset] - row_score;

            key ^= custom_hash(row_score) + 0x9e3779b9 + (key << 6) + (key >> 2);
        }

        hashes[pairs_offset] = key;
        ++pairs_offset;
    }
}

void combine_and_encode_tuples_required_gpu(GpuData &gpu_data, const PairsTuple *tuples, size_t n_tuples, size_t n_pairs, const size_t *scores1, const size_t *scores2, size_t row_offset)
{
    const size_t m_rows = gpu_data.m_rows;
    /* Each tuple is treated by one warp. */
    const int n_threads = 256;

    gpu_data.n_required = n_pairs;

    /* Reserve space for hashes. */
    gpu_data.resize_buffer(&gpu_data.required_buffer, gpu_data.len_required_buffer, gpu_data.n_required);

    /* Copy and flatten the tuples. */
    gpu_data.copy_tuples(tuples, n_tuples);
    int n_blocks = (n_tuples + n_threads - 1) / n_threads;
    assert(n_blocks > 0);
    flatten_and_encode_tuples<true><<<n_blocks, n_threads>>>(gpu_data.tuples_buffer, n_tuples, scores1, scores2, gpu_data.rhs, gpu_data.required_buffer, m_rows, row_offset);
}

void combine_and_encode_tuples_search_gpu(GpuData &gpu_data, const PairsTuple *tuples, size_t n_tuples, size_t n_pairs, const size_t *scores1, const size_t *scores2, size_t row_offset)
{
    const int n_threads = 256;
    /* Each tuple is treated by one warp. */
    const size_t m_rows = gpu_data.m_rows;

    gpu_data.n_search = n_pairs;

    /* Reserve space for hashes. */
    gpu_data.resize_buffer(&gpu_data.search_buffer, gpu_data.len_search_buffer, gpu_data.n_search);
    gpu_data.resize_buffer(&gpu_data.results_search_buffer, gpu_data.len_results_buffer, gpu_data.n_search);

    gpu_data.copy_tuples(tuples, n_tuples);
    int n_blocks = (n_tuples + n_threads - 1) / n_threads;
    assert(n_blocks > 0);
    flatten_and_encode_tuples<false><<<n_blocks, n_threads>>>(gpu_data.tuples_buffer, n_tuples, scores1, scores2, gpu_data.rhs, gpu_data.search_buffer, m_rows, row_offset);
}

void combine_and_encode_tuples_gpu(GpuData &gpu_data, const PairsTuple *tuples1, const PairsTuple *tuples2, size_t n_tuples1, size_t n_tuples2, size_t n_pairs1, size_t n_pairs2, size_t row_offset)
{
    auto profiler = std::make_unique<ScopedProfiler>("Eval GPU: combine + encode  ");

    /* The shorter array will be encoded as required. */
    const bool encode_first_as_required = (n_pairs1 < n_pairs2);

    if (encode_first_as_required)
    {
        combine_and_encode_tuples_required_gpu(gpu_data, tuples1, n_tuples1, n_pairs1, gpu_data.set1_scores, gpu_data.set2_scores, row_offset);
        combine_and_encode_tuples_search_gpu(gpu_data, tuples2, n_tuples2, n_pairs2, gpu_data.set3_scores, gpu_data.set4_scores, row_offset);
    }
    else
    {
        combine_and_encode_tuples_required_gpu(gpu_data, tuples2, n_tuples2, n_pairs2, gpu_data.set3_scores, gpu_data.set4_scores, row_offset);
        combine_and_encode_tuples_search_gpu(gpu_data, tuples1, n_tuples1, n_pairs1, gpu_data.set1_scores, gpu_data.set2_scores, row_offset);
    }

    profiler.reset();
}

void sort_required_gpu(GpuData &gpu_data)
{
    auto profiler = std::make_unique<ScopedProfiler>("Eval GPU: sort required     ");

    const size_t n_required = gpu_data.n_required;
    size_t *required = gpu_data.required_buffer;

    /* Sort the array of required keys. */
    thrust::sort(thrust::device, required, required + n_required);

    profiler.reset();
}

std::vector<size_t> find_equal_hashes(GpuData &gpu_data, bool sort_required)
{
    /* The shorter array will be encoded as required and will be sorted. */
    const size_t n_required = gpu_data.n_required;
    const size_t n_search = gpu_data.n_search;

    size_t *required = gpu_data.required_buffer;
    size_t *search = gpu_data.search_buffer;
    bool *result = gpu_data.results_search_buffer;

    /* Compute hashes of required vectors. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval GPU: sort required     ");
    if (sort_required)
    {
        sort_required_gpu(gpu_data);
    }

    profiler = std::make_unique<ScopedProfiler>("Eval GPU: binary search     ");

    thrust::binary_search(thrust::device, required, required + n_required, search, search + n_search, result);

    profiler = std::make_unique<ScopedProfiler>("Eval GPU: check results     ");

    std::vector<size_t> hashes;
    auto iter = thrust::find(thrust::device, result, result + n_search, true);

    while (iter != result + n_search)
    {
        /* Retrieve all potential matches! If we have duplicates in our hash, we might skip hashes here.. */
        /* Get the position of the found element and copy back its (unsorted) search value. */
        size_t i_search = thrust::distance(result, iter);

        size_t val = 0;
        cudaMemcpy(&val, search + i_search, sizeof(size_t), cudaMemcpyDeviceToHost);
        hashes.push_back(val);

        iter = thrust::find(thrust::device, iter + 1, result + n_search, true);
    }
    profiler.reset();
    return hashes;
}

std::vector<std::pair<size_t, size_t>> find_hash_positions_gpu(GpuData &gpu_data, const std::vector<size_t> &hashes, size_t n_p1, size_t n_p2, bool encode_first_as_required)
{
    encode_first_as_required = encode_first_as_required || (n_p1 < n_p2);

    std::vector<std::pair<size_t, size_t>> solution_candidates;
    solution_candidates.reserve(hashes.size());

    size_t *required = gpu_data.required_buffer;
    size_t *search = gpu_data.search_buffer;

    const size_t n_required = gpu_data.n_required;
    const size_t n_search = gpu_data.n_search;

    /* Retrieve all potential matches! If we have duplicates in our hash, we might skip hashes here.. */
    for (const auto hash : hashes)
    {
        auto iter_req = thrust::find(thrust::device, required, required + n_required, hash);
        auto iter_search = thrust::find(thrust::device, search, search + n_search, hash);

        assert(iter_req != required + n_required);
        assert(iter_search != search + n_search);

        auto pos_req = thrust::distance(required, iter_req);
        auto pos_search = thrust::distance(search, iter_search);

        if (encode_first_as_required)
            solution_candidates.emplace_back(pos_req, pos_search);
        else
            solution_candidates.emplace_back(pos_search, pos_req);
    }

    return solution_candidates;
}
