#include "argparse.hpp"

#include "markshare.hpp"
#include "pairs_tuple.hpp"
#include "profiler.hpp"
#include "ter_solver.cuh"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits> // For CHAR_BIT
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <execution>
#include <fstream>
#include <future>
#include <iterator>
#include <iostream>
#include <numeric>
#include <limits>
#include <queue>
#include <utility>
#include <string>
#include <unordered_map>
#include <cmath>
#include <thread>
#include <type_traits>

#include <omp.h>

/* [OPT] SSE4.2: _mm_popcnt_u64, SSE2: _mm_add_epi64 / _mm_loadu_si128 */
#include <nmmintrin.h>  /* SSE4.2 — _mm_popcnt_u64 */
#include <emmintrin.h>  /* SSE2   — _mm_add_epi64, _mm_loadu/storeu_si128 */

#ifdef WITH_GPU
#include "cuda_kernels.cuh"
#endif

/* 3000000000 ~= 56 GB of active storage requirement. 4000000000 goes OOM on H200. 3500000000 works and goes up to 63.6 GB. 3900000000 also works and is about ~= 70.11 */
/* Adapted for 8GB RAM (Intel Celeron N5100). Safe limit for CPU memory. */
size_t max_pairs_per_chunk = 20000000;

typedef std::chrono::high_resolution_clock::time_point TimeVar;
using u128 = unsigned __int128;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&...args)
{
    TimeVar t1 = timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow() - t1);
}

template <typename T>
void print_vector(const std::vector<T> vec)
{
    std::cout << "[";
    for (auto &e : vec)
        std::cout << " " << e;
    std::cout << "]\n";
}

inline std::string u128_to_string(u128 value)
{
    if (value == 0)
        return "0";

    std::string result;
    while (value > 0)
    {
        result.push_back(static_cast<char>('0' + static_cast<int>(value % 10)));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

inline std::ostream &operator<<(std::ostream &os, u128 value)
{
    return os << u128_to_string(value);
}

inline u128 parse_u128_decimal(const std::string &str)
{
    u128 value = 0;
    for (char c : str)
    {
        if (std::isdigit(static_cast<unsigned char>(c)))
            value = value * 10 + static_cast<unsigned>(c - '0');
    }
    return value;
}

template <typename T>
std::string number_to_string(T value)
{
    if constexpr (std::is_same_v<T, u128>)
        return u128_to_string(value);
    else
        return std::to_string(value);
}

std::vector<u128> parse_u128_tokens(const std::string &text)
{
    std::vector<u128> values;
    std::string cur;

    for (char c : text)
    {
        if (std::isdigit(static_cast<unsigned char>(c)))
        {
            cur.push_back(c);
        }
        else if (!cur.empty())
        {
            values.push_back(parse_u128_decimal(cur));
            cur.clear();
        }
    }

    if (!cur.empty())
        values.push_back(parse_u128_decimal(cur));

    return values;
}

struct SubsetSum1D128
{
    std::vector<u128> values;
    u128 target = 0;
    std::vector<size_t> planted_indices;
    std::string source_format;

    bool has_planted_solution() const
    {
        return !planted_indices.empty();
    }

    bool verify_planted_solution() const
    {
        u128 sum = 0;
        for (size_t idx : planted_indices)
        {
            if (idx >= values.size())
                return false;
            sum += values[idx];
        }
        return sum == target;
    }
};

bool load_hgj_txt_instance(const std::string &path, SubsetSum1D128 &instance)
{
    std::ifstream file(path);
    if (!file.is_open())
        return false;

    instance = {};
    instance.source_format = "txt";

    std::string line;
    bool saw_target = false;
    bool reading_planted = false;
    std::string planted_text;

    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        const size_t target_pos = line.find("Target T");
        if (target_pos != std::string::npos)
        {
            const size_t colon_pos = line.find(':', target_pos);
            if (colon_pos != std::string::npos)
            {
                instance.target = parse_u128_decimal(line.substr(colon_pos + 1));
                saw_target = true;
            }
            continue;
        }

        if (line.find("planted solution indices") != std::string::npos)
        {
            reading_planted = true;
            planted_text += line;
            planted_text += ' ';
            if (line.find(']') != std::string::npos)
                reading_planted = false;
            continue;
        }

        if (reading_planted)
        {
            planted_text += line;
            planted_text += ' ';
            if (line.find(']') != std::string::npos)
                reading_planted = false;
            continue;
        }

        const size_t comment_pos = std::min(line.find('#'), line.find("//"));
        const std::string data_part = (comment_pos == std::string::npos) ? line : line.substr(0, comment_pos);
        auto values = parse_u128_tokens(data_part);
        instance.values.insert(instance.values.end(), values.begin(), values.end());
    }

    if (!planted_text.empty())
    {
        const size_t open = planted_text.find('[');
        const size_t close = planted_text.find(']');
        if (open != std::string::npos && close != std::string::npos && close > open)
        {
            for (u128 value : parse_u128_tokens(planted_text.substr(open + 1, close - open - 1)))
                instance.planted_indices.push_back(static_cast<size_t>(value));
        }
    }

    return saw_target && !instance.values.empty();
}

bool load_prb_1d128_instance(const std::string &path, SubsetSum1D128 &instance)
{
    std::ifstream file(path);
    if (!file.is_open())
        return false;

    std::string raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto tokens = parse_u128_tokens(raw);
    if (tokens.size() < 4)
        return false;

    const size_t m = static_cast<size_t>(tokens[0]);
    const size_t n = static_cast<size_t>(tokens[1]);
    if (m != 1 || tokens.size() < 2 + n + 1)
        return false;

    instance = {};
    instance.source_format = "prb";
    instance.values.reserve(n);
    for (size_t i = 0; i < n; ++i)
        instance.values.push_back(tokens[2 + i]);
    instance.target = tokens[2 + n];
    return true;
}

bool can_fit_size_t(const SubsetSum1D128 &instance)
{
    const u128 max_size_t = static_cast<u128>(std::numeric_limits<size_t>::max());
    if (instance.target > max_size_t)
        return false;
    for (u128 value : instance.values)
    {
        if (value > max_size_t)
            return false;
    }
    return true;
}

void write_1d_prb(const SubsetSum1D128 &instance, const std::string &path)
{
    std::ofstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Could not open output prb file");

    file << "1 " << instance.values.size() << "\n";
    for (u128 value : instance.values)
        file << value << " ";
    file << instance.target << "\n";
}

size_t highestSetBit(size_t value)
{
    if (value == 0)
        return -1; // No bits are set
#if defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 8
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clzll(value);
#elif defined(__SIZEOF_SIZE_T__) && __SIZEOF_SIZE_T__ == 4
    return (sizeof(size_t) * CHAR_BIT - 1) - __builtin_clz(value);
#else
#error Unsupported size_t size
#endif
}

size_t countSetBits(size_t num)
{
    return __builtin_popcount(num);
}

void print_bits(size_t value)
{
    if (value == 0)
    {
        std::cout << "0"; // Special case: value is 0
        return;
    }

    // Determine the position of the highest set bit
    size_t msb = 0;
    for (size_t i = sizeof(size_t) * 8; i > 0; --i)
    {
        if (value & (1ULL << (i - 1)))
        {
            msb = i - 1;
            break;
        }
    }

    // Print bits from the highest set bit down to 0
    for (size_t i = msb + 1; i > 0; --i)
    {
        std::cout << ((value & (1ULL << (i - 1))) ? '1' : '0');
    }
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T> &vec,
    const std::vector<std::size_t> &p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
                   [&](std::size_t i)
                   { return vec[i]; });
    return sorted_vec;
}

template <typename T>
std::vector<T> generate_subsets(const std::vector<T> &weights)
{
    size_t n = weights.size();
    size_t total_subsets = 1ULL << n; /* Total subsets is 2^n. */

    printf("Generating %zu possible subset weights for set of size %zu.\n", total_subsets, n);
    std::vector<T> set_weights(total_subsets, 0);

    size_t generated = 1;
    for (size_t pass = 0; pass < n; ++pass)
    {
        const T weight = weights[pass];

        /* [OPT] SSE2 _mm_add_epi64: proses 2 elemen uint64_t per iterasi SIMD.
         * Untuk n/4=16: loop ini dijalankan 16 kali dengan generated hingga 32768.
         * u128 tidak bisa pakai SSE karena lebar 128-bit melebihi register. */
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            const __m128i wvec = _mm_set1_epi64x(static_cast<int64_t>(weight));
            size_t i = 0;
            /* Loop SIMD: 2 elemen per iterasi */
            for (; i + 2 <= generated; i += 2)
            {
                __m128i v = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(set_weights.data() + i));
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>(set_weights.data() + generated + i),
                    _mm_add_epi64(v, wvec));
            }
            /* Tail: sisa 0 atau 1 elemen */
            for (; i < generated; ++i)
                set_weights[generated + i] = set_weights[i] + weight;
        }
        else
        {
            /* u128 atau tipe lain: pakai OMP paralel scalar */
#pragma omp parallel for
            for (size_t i = 0; i < generated; ++i)
                set_weights[generated + i] = set_weights[i] + weight;
        }

        generated <<= 1;
    }

    return set_weights;
}

// Function to sort an array and obtain sorted indices
template <typename T>
std::vector<size_t> sort_indices(const std::vector<T> &arr, bool ascending)
{
    size_t n = arr.size();

    // Create indices list from 0 to n-1 using std::iota
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on corresponding values in the array
    if (ascending)
    {
        std::sort(indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] < arr[i2]; });
    }
    else
    {
        std::sort(indices.begin(), indices.end(), [&arr](size_t i1, size_t i2)
                  { return arr[i1] > arr[i2]; });
    }

    return indices;
}

static inline std::vector<size_t> unpack_mask(size_t mask)
{
    std::vector<size_t> indices;
    while (mask > 0)
    {
        int bit = __builtin_ctzll(mask);
        indices.push_back((size_t)bit);
        mask &= mask - 1;
    }
    return indices;
}

/* [OPT] Variant yang menulis ke buffer yang sudah ada (hindari alokasi heap di hot path). */
static inline void unpack_mask_into(size_t mask, std::vector<size_t> &buf)
{
    buf.clear();
    while (mask > 0)
    {
        buf.push_back((size_t)__builtin_ctzll(mask));
        mask &= mask - 1;
    }
}

size_t print_subset_and_compute_sum(const std::vector<size_t> &numbers, size_t index)
{
    std::cout << "Subset for index " << index << " (binary " << std::bitset<64>(index) << "): ";
    size_t sum = 0;

    bool hasElements = false;
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
        {
            if (hasElements)
            {
                std::cout << ", ";
            }
            std::cout << numbers[i];
            sum += numbers[i];
            hasElements = true;
        }
    }

    if (!hasElements)
    {
        std::cout << "Empty";
    }

    std::cout << std::endl;
    return sum;
}

std::vector<size_t> extract_subset(const std::vector<size_t> &numbers, size_t index)
{
    std::vector<size_t> indices;

    size_t position = 0;
    while (index > 0)
    {
        if (index & 1)
            indices.push_back(numbers[position]);
        index >>= 1;
        ++position;
    }
    return indices;
}

void concat_vectors(std::vector<size_t> &concat_vec, size_t &concat_len, const std::vector<const std::vector<size_t> *> &vectors, const std::vector<size_t> offsets)
{
    assert(vectors.size() == offsets.size());

    concat_len = 0;
    for (const auto &vec : vectors)
    {
        concat_len += vec->size();
    }

    assert(concat_len <= concat_vec.size());

    size_t pos = 0;
    for (size_t ivec = 0; ivec < vectors.size(); ++ivec)
    {
        const auto &vec = *vectors[ivec];
        const auto offset = offsets[ivec];

        for (size_t num : vec)
        {
            concat_vec[pos] = (num + offset);
            ++pos;
        }
    }

    assert(pos == concat_len);
}

void print_four_list_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4)
{
    auto sum1 = print_subset_and_compute_sum(list1, index_list1);
    auto sum2 = print_subset_and_compute_sum(list2, index_list2);
    auto sum3 = print_subset_and_compute_sum(list3, index_list3);
    auto sum4 = print_subset_and_compute_sum(list4, index_list4);

    std::cout << "The sum is " << sum1 << " + " << sum2 << " + " << sum3 << " + " << sum4 << " = " << sum1 + sum2 + sum3 + sum4 << std::endl;
}

void append_solution_to_file(std::ofstream &sol_file, const std::vector<size_t> &numbers, size_t index)
{
    for (size_t i = 0; i < numbers.size(); ++i)
    {
        // Check if the i-th bit in the index is set
        if (index & (1ULL << i))
            sol_file << 1;
        else
            sol_file << 0;
    }
}

void write_four_list_solution_to_file(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4, const std::vector<size_t> &list1, const std::vector<size_t> &list2, const std::vector<size_t> &list3, const std::vector<size_t> &list4, const std::string &instance_name)
{
    const std::string sol_name = instance_name + ".sol";
    printf("Writing solution to %s\n", sol_name.c_str());
    std::ofstream sol_file(sol_name);

    append_solution_to_file(sol_file, list1, index_list1);
    append_solution_to_file(sol_file, list2, index_list2);
    append_solution_to_file(sol_file, list3, index_list3);
    append_solution_to_file(sol_file, list4, index_list4);

    sol_file << std::endl;
}

size_t custom_hash_cpu(size_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

template <const bool ENCODE_REQUIRED>
std::vector<size_t> flatten_and_encode_tuples_cpu(const std::vector<PairsTuple>& tuples, size_t n_tuples, const std::vector<size_t> &scores1, const std::vector<size_t> &scores2, const MarkShareFeas &ms_inst, size_t row_offset)
{
    const size_t m_rows_left = ms_inst.m() - row_offset;
    std::vector<size_t> hashes (n_tuples);

#pragma omp parallel for
    for (size_t i_tuple = 0; i_tuple < tuples.size(); ++i_tuple)
    {
        auto first = tuples[i_tuple].pairs_first;
        auto pair_second_beg = tuples[i_tuple].pairs_second_beg;
        auto pair_second_end = pair_second_beg + tuples[i_tuple].pairs_n_second;
        auto pair_offset = tuples[i_tuple].pairs_offset;

        for (size_t second = pair_second_beg; second < pair_second_end; ++second)
        {
            size_t key = 0;

            /* Compute the hash of this tuple. */
            for (size_t i_row = 0; i_row < m_rows_left; ++i_row) {
                /* Compute the pair's score of this row and add it (encoded) to key. */
                size_t row_score = scores1[first * m_rows_left + i_row] + scores2[second * m_rows_left + i_row];

                if (ENCODE_REQUIRED)
                    row_score = ms_inst.b()[i_row + row_offset] - row_score;

                key ^= custom_hash_cpu(row_score) + 0x9e3779b9 + (key << 6) + (key >> 2);
            }

            hashes[pair_offset] = key;
            ++pair_offset;
        }
    }

    return hashes;
}

void compute_scores_cpu(const MarkShareFeas &ms_inst, std::vector<size_t> &scores, const std::vector<size_t> &subsets, size_t col_offset, size_t row_offset)
{
    const size_t m_rows_left = ms_inst.m() - row_offset;
    assert(ms_inst.m() >= row_offset);

    /* [OPT] Gunakan thread_local buffer untuk hindari alokasi heap berulang di inner loop. */
#pragma omp parallel for
    for (size_t i = 0; i < subsets.size(); ++i)
    {
        thread_local std::vector<size_t> local_indices;
        unpack_mask_into(subsets[i], local_indices);
        ms_inst.compute_value(local_indices, col_offset, row_offset, scores.data() + i * m_rows_left);
    }
}

/* For each tuple consisting of one element in set1_scores and a range of elements in set2_scores, compute its partial right hand side. */
void combine_scores_cpu(const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, size_t m_rows, const std::vector<PairsTuple> &tuples, std::vector<size_t> &scores_pairs, size_t row_offset)
{
    const size_t m_rows_left = m_rows - row_offset;

#pragma omp parallel for
    for (size_t i_tuple = 0; i_tuple < tuples.size(); ++i_tuple)
    {
        auto first = tuples[i_tuple].pairs_first;
        auto pair_second_beg = tuples[i_tuple].pairs_second_beg;
        auto pair_second_end = pair_second_beg + tuples[i_tuple].pairs_n_second;
        auto pair_offset = tuples[i_tuple].pairs_offset;

        for (size_t second = pair_second_beg; second < pair_second_end; ++second)
        {
            for (size_t i_row = 0; i_row < m_rows_left; ++i_row)
                scores_pairs[pair_offset * m_rows_left + i_row] = set1_scores[first * m_rows_left + i_row] + set2_scores[second * m_rows_left + i_row];
            ++pair_offset;
        }
    }
}

template <typename T>
void print_info_line(
#ifdef WITH_GPU
    const GpuData &gpu_data,
    bool run_on_gpu,
#endif
    size_t i_iter, double time, T score1, T score2, size_t n_q1, size_t n_q2)
{
    bool print = false;
    if (i_iter < 10)
        print = true;
    else if (i_iter < 100 && i_iter % 10 == 0)
        print = true;
    else if (i_iter < 10000 && i_iter % 100 == 0)
        print = true;
    else if (i_iter % 1000 == 0)
        print = true;

    if (print)
    {
#ifdef WITH_GPU
        if (run_on_gpu)
        {
            const double n_gb = gpu_data.get_gb_allocated();
            std::cout << std::setw(5) << i_iter << " "
                      << std::setw(8) << std::fixed << std::setprecision(2) << time << "s "
                      << "[" << std::setprecision(6) << n_gb << " GB]: "
                      << number_to_string(score1) << " + " << number_to_string(score2) << "; "
                      << n_q1 << " x " << n_q2 << " possible solutions\n";
        }
        else
#endif
        {
            std::cout << std::setw(5) << i_iter << " "
                      << std::setw(8) << std::fixed << std::setprecision(2) << time << "s: "
                      << number_to_string(score1) << " + " << number_to_string(score2) << "; "
                      << n_q1 << " x " << n_q2 << " possible solutions\n";
        }
    }
}

bool verify_solution(const std::pair<size_t, size_t> &solution, const PairsTuple *same_score_q1, size_t n_tuples_q1, const PairsTuple *same_score_q2, size_t n_tuples_q2, const std::vector<size_t> &set1_subsets, const std::vector<size_t> &set2_subsets_sorted_asc, const std::vector<size_t> &set3_subsets, const std::vector<size_t> &set4_subsets_sorted_desc, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, const MarkShareFeas &ms_inst)
{
    /* Print and verify the solution! */
    /* Get the correct pairs. */
    size_t pos_q1 = 0;
    size_t pos_q2 = 0;
    (void)n_tuples_q1;
    (void)n_tuples_q2;

    while (solution.first >= same_score_q1[pos_q1].pairs_offset + same_score_q1[pos_q1].pairs_n_second)
    {
        ++pos_q1;
        assert(pos_q1 < n_tuples_q1);
    }
    while (solution.second >= same_score_q2[pos_q2].pairs_offset + same_score_q2[pos_q2].pairs_n_second)
    {
        ++pos_q2;
        assert(pos_q2 < n_tuples_q2);
    }

    assert(solution.first - same_score_q1[pos_q1].pairs_offset < same_score_q1[pos_q1].pairs_n_second);
    assert(solution.second - same_score_q2[pos_q2].pairs_offset < same_score_q2[pos_q2].pairs_n_second);

    const size_t pair_q1_second = same_score_q1[pos_q1].pairs_second_beg + solution.first - same_score_q1[pos_q1].pairs_offset;
    const size_t pair_q2_second = same_score_q2[pos_q2].pairs_second_beg + solution.second - same_score_q2[pos_q2].pairs_offset;
    std::pair<size_t, size_t> pair_q1 = {same_score_q1[pos_q1].pairs_first, pair_q1_second};
    std::pair<size_t, size_t> pair_q2 = {same_score_q2[pos_q2].pairs_first, pair_q2_second};

    std::vector<size_t> solution_1d(subset_sum_1d.size());

    auto v1 = unpack_mask(set1_subsets[pair_q1.first]);
    auto v2 = unpack_mask(set2_subsets_sorted_asc[pair_q1.second]);
    auto v3 = unpack_mask(set3_subsets[pair_q2.first]);
    auto v4 = unpack_mask(set4_subsets_sorted_desc[pair_q2.second]);
    const std::vector<const std::vector<size_t> *> vectors = {&v1, &v2, &v3, &v4};

    size_t len;
    concat_vectors(solution_1d, len, vectors, offsets);

    /* We found a solution. Construct it, print it, and return. */
    if (!ms_inst.is_solution_feasible(solution_1d, len))
    {
#ifndef NDEBUG
        printf("Error, solution is not feasible!\n");
#endif
        return false;
    }

    return true;
}

template <typename T>
std::pair<size_t, T> max_encodable_dimension(size_t max_coeff, size_t n_cols)
{
    constexpr T max_index = std::numeric_limits<T>::max();

    static_assert(std::numeric_limits<T>::max() >= std::numeric_limits<size_t>::max());
    assert(max_coeff > 1);

    /* TODO: we need to properly check for an overflow here. */
    const T basis = static_cast<T>(n_cols) * static_cast<T>(max_coeff) + 1;

    /* Given a vector (x1, x2, .. ) we reduce dimensions as x1 * max_sum^0 + x2 * max_sum^1 + x3 * max_sum^2 ...
     * When encoding into T we need to guarantee that the highest dimension, d, still fits into our index type. Define the basis B as
     *   B := (max_sum + 1)
     * Then
     *   sum_0^d-1 (B - 1) (max_sum + 1)^k <= max_index
     * <=> geometric sum
     *   (B - 1) [B^d - 1] / [B - 1] = (B - 1) * [B^d - 1] / (B - 1) <= max_index
     * <=>
     *   B^d <= max_index + 1
     * <=>
     *   d_max = |_ log_(max_sum + 1) (max_index + 1) _|
     */
    size_t max_dim = static_cast<size_t>(std::floor(std::log(max_index) / std::log(basis)));

    printf("Max reducible dimension is %zu (encoded with basis %zu)\n", max_dim, static_cast<size_t>(basis));
    return {max_dim, basis};
}

template <bool ascending, typename T, typename F>
void extract_pairs_from_heap(std::vector<PairsTuple> &pairs_same_score, std::vector<std::pair<size_t, size_t>> &heap, size_t score_pair, std::vector<size_t> &chunks_beg, std::vector<size_t> &chunks_n_pairs, size_t &n_pairs_total, const std::vector<T> &first_weights, const std::vector<T> &second_weights, const std::vector<T> &second_weights_sorted_asc, F &&cmp)
{
    /* Counter for pairs stored in current chunk. */
    size_t n_pairs_chunk = 0;

    /* For each element a in the heap with score(a) == score_pair, collect all solutions. */
    while (!heap.empty() && first_weights[heap.front().first] + second_weights_sorted_asc[heap.front().second] == score_pair)
    {
        const auto pair1_same_score = heap.front();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.pop_back();

        const size_t pos_second_weights_beg = pair1_same_score.second;
        size_t pos_second_weights_end = pos_second_weights_beg;

        /* Iterate the second elements. */
        const auto pos2_val = second_weights_sorted_asc[pos_second_weights_end];

        while (pos_second_weights_end < second_weights.size() && pos2_val == second_weights_sorted_asc[pos_second_weights_end])
            ++pos_second_weights_end;

        size_t n_pairs = pos_second_weights_end - pos_second_weights_beg;
        assert(n_pairs <= max_pairs_per_chunk);

        if (n_pairs + n_pairs_chunk >= max_pairs_per_chunk)
        {
            chunks_beg.push_back(pairs_same_score.size());
            chunks_n_pairs.push_back(n_pairs_chunk);
            n_pairs_chunk = 0;
        }

        pairs_same_score.emplace_back(pair1_same_score.first, pos_second_weights_beg, n_pairs, n_pairs_chunk);
        n_pairs_chunk += n_pairs;
        n_pairs_total += n_pairs;

        if (pos_second_weights_end < second_weights.size())
        {
            if (ascending)
                assert(score_pair < first_weights[pair1_same_score.first] + second_weights_sorted_asc[pos_second_weights_end]);
            else
                assert(score_pair > first_weights[pair1_same_score.first] + second_weights_sorted_asc[pos_second_weights_end]);

            heap.emplace_back(pair1_same_score.first, pos_second_weights_end);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    chunks_n_pairs.push_back(n_pairs_chunk);
    chunks_beg.push_back(pairs_same_score.size());
}

std::vector<size_t> find_equal_hashes_cpu(std::vector<size_t>& hashes_required, const std::vector<size_t>& hashes_search, bool sort_required = true)
{
    /* The shorter array will be encoded as required and will be sorted. */
    const size_t n_search = hashes_search.size();

    /* Compute hashes of required vectors. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval CPU: sort required     ");
    if (sort_required)
    {
        std::sort(hashes_required.begin(), hashes_required.end());
    }

    profiler = std::make_unique<ScopedProfiler>("Eval CPU: binary search     ");

    std::vector<bool> result(n_search);

    /* Parallel binary search for each hash. */
#pragma omp parallel for
    for (size_t i = 0; i < n_search; i++)
    {
        result[i] = std::binary_search(hashes_required.begin(), hashes_required.end(), hashes_search[i]);
    }

    profiler = std::make_unique<ScopedProfiler>("Eval CPU: check results     ");

    std::vector<size_t> hashes;
    for (size_t i = 0; i < n_search; i++)
    {
        if (result[i])
        {
            hashes.push_back(hashes_search[i]);
        }
    }

    profiler.reset();
    return hashes;
}

std::vector<std::pair<size_t, size_t>> find_hash_positions_cpu(const std::vector<size_t>& hashes_required,
                                                                const std::vector<size_t>& hashes_search,
                                                                const std::vector<size_t>& matching_hashes,
                                                                bool encode_first_as_required)
{
    encode_first_as_required = encode_first_as_required || (hashes_required.size() < hashes_search.size());

    std::vector<std::pair<size_t, size_t>> solution_candidates;
    solution_candidates.reserve(matching_hashes.size());

    for (const auto hash : matching_hashes)
    {
        // Find position in required vector
        auto iter_req = std::find(hashes_required.begin(), hashes_required.end(), hash);
        // Find position in search vector
        auto iter_search = std::find(hashes_search.begin(), hashes_search.end(), hash);

        // Assertions (could be removed or replaced with error handling)
        assert(iter_req != hashes_required.end());
        assert(iter_search != hashes_search.end());

        auto pos_req = std::distance(hashes_required.begin(), iter_req);
        auto pos_search = std::distance(hashes_search.begin(), iter_search);

        if (encode_first_as_required)
            solution_candidates.emplace_back(pos_req, pos_search);
        else
            solution_candidates.emplace_back(pos_search, pos_req);
    }

    return solution_candidates;
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_cpu(const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores_sorted_asc, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores_sorted_desc, const std::vector<PairsTuple> &same_score_q1, size_t n_pairs_q1, const std::vector<PairsTuple> &same_score_q2, size_t n_pairs_q2, const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<size_t> &set1_subsets, const std::vector<size_t> &set2_subsets_sorted_asc, const std::vector<size_t> &set3_subsets, const std::vector<size_t> &set4_subsets_sorted_desc,
                                                                         const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets)
{
    /* Encode the 2 sets as 'required'. */
    auto profiler = std::make_unique<ScopedProfiler>("Eval CPU: combine + encode  ");
    auto required = flatten_and_encode_tuples_cpu<true>(same_score_q1, n_pairs_q1, set1_scores, set2_scores_sorted_asc, ms_inst, reduce_dim);
    const auto search = flatten_and_encode_tuples_cpu<false>(same_score_q2, n_pairs_q2, set3_scores, set4_scores_sorted_desc, ms_inst, reduce_dim);
    profiler.reset();

    auto hashes = find_equal_hashes_cpu(required, search);

    if (!hashes.empty())
    {
        required = flatten_and_encode_tuples_cpu<true>(same_score_q1, n_pairs_q1, set1_scores, set2_scores_sorted_asc, ms_inst, reduce_dim);
        const std::vector<std::pair<size_t, size_t>> candidates = find_hash_positions_cpu(required, search, hashes, true);

        /* Check all potential solutions. */
        for (const auto &solution_cand : candidates)
        {
            /* Offset each solution candidate by its chunk. */
            const auto feasible = verify_solution(solution_cand, same_score_q1.data(), n_pairs_q1, same_score_q2.data(), n_pairs_q2, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, ms_inst);

            if (feasible)
                return {true, solution_cand};
        }
    }

    return {false, {0, 0}};
}

#ifdef WITH_GPU
std::tuple<bool, size_t, size_t, std::pair<size_t, size_t>> evaluate_gpu(GpuData &gpu_data, const std::vector<PairsTuple> &same_score_q1, const std::vector<PairsTuple> &same_score_q2, const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<size_t> &chunks_q1_beg, const std::vector<size_t> &chunks_q1_n_pairs, const std::vector<size_t> &chunks_q2_beg, const std::vector<size_t> &chunks_q2_n_pairs, size_t n_q1_chunks, size_t n_q2_chunks, const std::vector<size_t> &set1_subsets, const std::vector<size_t> &set2_subsets_sorted_asc, const std::vector<size_t> &set3_subsets, const std::vector<size_t> &set4_subsets_sorted_desc,
                                                                         const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets
#ifndef NDEBUG
                                                                         ,
                                                                         size_t n_pairs_q1, size_t n_pairs_q2
#endif
)
{
    assert(n_q1_chunks > 0);
    assert(n_q2_chunks > 0);
    assert(chunks_q1_beg.size() == n_q1_chunks + 1);
    assert(chunks_q2_beg.size() == n_q2_chunks + 1);

    /* Do all this per chunk and quadratically. */
    size_t n_q1_pairs_offset = 0;

    for (size_t i_q1_chunk = 0; i_q1_chunk < n_q1_chunks; ++i_q1_chunk)
    {
        const size_t q1_chunk_beg = chunks_q1_beg[i_q1_chunk];
        const size_t n_pairs_q1_chunk = chunks_q1_n_pairs[i_q1_chunk];
        const size_t n_tuples_q1_chunk = chunks_q1_beg[i_q1_chunk + 1] - q1_chunk_beg;
        assert(q1_chunk_beg + n_tuples_q1_chunk <= same_score_q1.size());

        const PairsTuple *q1_chunk = same_score_q1.data() + q1_chunk_beg;
        size_t n_q2_pairs_offset = 0;

        combine_and_encode_tuples_required_gpu(gpu_data, q1_chunk, n_tuples_q1_chunk, n_pairs_q1_chunk, gpu_data.set1_scores, gpu_data.set2_scores, reduce_dim);
        sort_required_gpu(gpu_data);

        for (size_t i_q2_chunk = 0; i_q2_chunk < n_q2_chunks; ++i_q2_chunk)
        {
            const size_t q2_chunk_beg = chunks_q2_beg[i_q2_chunk];
            const size_t n_pairs_q2_chunk = chunks_q2_n_pairs[i_q2_chunk];
            const size_t n_tuples_q2_chunk = chunks_q2_beg[i_q2_chunk + 1] - q2_chunk_beg;
            assert(q2_chunk_beg + n_tuples_q2_chunk <= same_score_q2.size());

            const PairsTuple *q2_chunk = same_score_q2.data() + q2_chunk_beg;

            combine_and_encode_tuples_search_gpu(gpu_data, q2_chunk, n_tuples_q2_chunk, n_pairs_q2_chunk, gpu_data.set3_scores, gpu_data.set4_scores, reduce_dim);

            const std::vector<size_t> hashes = find_equal_hashes(gpu_data, false);

            if (!hashes.empty())
            {
                /* Retrieve the actual solution. We have to copy encode our arrays once more and look for the hash afterwards. */
                combine_and_encode_tuples_required_gpu(gpu_data, q1_chunk, n_tuples_q1_chunk, n_pairs_q1_chunk, gpu_data.set1_scores, gpu_data.set2_scores, reduce_dim);
                combine_and_encode_tuples_search_gpu(gpu_data, q2_chunk, n_tuples_q2_chunk, n_pairs_q2_chunk, gpu_data.set3_scores, gpu_data.set4_scores, reduce_dim);

                const std::vector<std::pair<size_t, size_t>> candidates = find_hash_positions_gpu(gpu_data, hashes, n_pairs_q1_chunk, n_pairs_q2_chunk, true);

                /* Check all potential solutions. */
                for (const auto &solution_cand : candidates)
                {
                    /* Offset each solution candidate by its chunk. */
                    const auto feasible = verify_solution(solution_cand, q1_chunk, n_tuples_q1_chunk, q2_chunk, n_tuples_q2_chunk, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, ms_inst);

                    if (feasible)
                        return {true, i_q1_chunk, i_q2_chunk, solution_cand};
                }
            }

            n_q2_pairs_offset += n_pairs_q2_chunk;
        }

        assert(n_q2_pairs_offset == n_pairs_q2);
        n_q1_pairs_offset += n_pairs_q1_chunk;
    }

    assert(n_q1_pairs_offset == n_pairs_q1);

    return {false, 0, 0, {0, 0}};
}
#endif

enum BufferState
{
    EMPTY,
    EXTRACTING,
    READY_FOR_EVAL,
    EVALUATING,
    EVALUATED
};

struct PipelineBuffer
{
    std::vector<PairsTuple> same_score_q1, same_score_q2;
    std::vector<size_t> chunks_q1_n_pairs, chunks_q1_beg;
    std::vector<size_t> chunks_q2_n_pairs, chunks_q2_beg;
    size_t n_pairs_q1 = 0, n_pairs_q2 = 0;
    BufferState state = EMPTY;
};

struct EvalResult
{
    bool found = false;
    std::pair<size_t, size_t> solution;
    size_t i_q1_chunk = 0;
    size_t i_q2_chunk = 0;
};

EvalResult evaluate_gpu_or_cpu(PipelineBuffer &buf,
#ifdef WITH_GPU
                               GpuData &gpu_data,
#endif
                               const MarkShareFeas &ms_inst, size_t reduce_dim, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores_sorted_asc, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores_sorted_desc, const std::vector<size_t> &set1_subsets, const std::vector<size_t> &set2_subsets_sorted_asc, const std::vector<size_t> &set3_subsets, const std::vector<size_t> &set4_subsets_sorted_desc, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, bool run_on_gpu)
{
    EvalResult res;

    const size_t n_q1_chunks = buf.chunks_q1_n_pairs.size();
    const size_t n_q2_chunks = buf.chunks_q2_n_pairs.size();

    if (run_on_gpu)
    {
#ifdef WITH_GPU
        auto profiler_evaluate = std::make_unique<ScopedProfiler>("Evaluate solutions GPU      ");

        auto [done, q1_chunk, q2_chunk, solution_indices] = evaluate_gpu(gpu_data, buf.same_score_q1, buf.same_score_q2, ms_inst, reduce_dim, buf.chunks_q1_beg, buf.chunks_q1_n_pairs, buf.chunks_q2_beg, buf.chunks_q2_n_pairs, n_q1_chunks, n_q2_chunks, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets
#ifndef NDEBUG
                                                                         ,
                                                                         buf.n_pairs_q1,
                                                                         buf.n_pairs_q2
#endif
        );

        res.found = done;
        res.solution = solution_indices;
        res.i_q1_chunk = q1_chunk;
        res.i_q2_chunk = q2_chunk;

        profiler_evaluate.reset();
#else
        (void)n_q1_chunks;
        (void)n_q2_chunks;
        (void)set1_subsets;
        (void)set2_subsets_sorted_asc;
        (void)set3_subsets;
        (void)set4_subsets_sorted_desc;
        (void)subset_sum_1d;
        (void)offsets;

        printf("Error: GPU mode not available!\n\nAborting!\n");
        exit(1);
#endif
    }
    else
    {
        assert(n_q1_chunks == 1);
        assert(n_q2_chunks == 1);
        auto profiler_evaluate = std::make_unique<ScopedProfiler>("Evaluate solutions CPU      ");

        auto [done, solution_indices] = evaluate_cpu(set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc, buf.same_score_q1, buf.n_pairs_q1, buf.same_score_q2, buf.n_pairs_q2, ms_inst, reduce_dim, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets);

        res.found = done;
        res.solution = solution_indices;
        res.i_q1_chunk = 0;
        res.i_q2_chunk = 0;

        profiler_evaluate.reset();
    }

    buf.state = EVALUATED;

    return res;
}

template <typename T>
bool print_and_verify_solution(const PipelineBuffer &buf, const EvalResult &res, const MarkShareFeas &ms_inst, const std::vector<size_t> &subset_sum_1d, const std::vector<size_t> &offsets, const std::vector<T> &asc_indices_set2_weights, const std::vector<T> &desc_indices_set4_weights, const std::vector<size_t> &set1_subsets, const std::vector<size_t> &set2_subsets_sorted_asc, const std::vector<size_t> &set3_subsets, const std::vector<size_t> &set4_subsets_sorted_desc, std::vector<T> &list1, std::vector<T> &list2, std::vector<T> &list3, std::vector<T> &list4, const std::string &instance_name

#ifndef NDEBUG
                               ,
                               size_t subset_sum_1d_rhs,
                               const std::vector<T> &set1_weights, const std::vector<T> &set2_weights_sorted_asc, const std::vector<T> &set3_weights, const std::vector<T> &set4_weights_sorted_desc
#endif
)
{
    (void)asc_indices_set2_weights;
    (void)desc_indices_set4_weights;

    /* Get the correct pairs. */
    size_t pos_q1 = buf.chunks_q1_beg[res.i_q1_chunk];
    size_t pos_q2 = buf.chunks_q2_beg[res.i_q2_chunk];

    while (res.solution.first >= buf.same_score_q1[pos_q1].pairs_offset + buf.same_score_q1[pos_q1].pairs_n_second)
        ++pos_q1;
    while (res.solution.second >= buf.same_score_q2[pos_q2].pairs_offset + buf.same_score_q2[pos_q2].pairs_n_second)
        ++pos_q2;

    assert(pos_q1 < buf.chunks_q1_beg[res.i_q1_chunk + 1]);
    assert(pos_q2 < buf.chunks_q2_beg[res.i_q2_chunk + 1]);
    assert(res.solution.first - buf.same_score_q1[pos_q1].pairs_offset < buf.same_score_q1[pos_q1].pairs_n_second);
    assert(res.solution.second - buf.same_score_q2[pos_q2].pairs_offset < buf.same_score_q2[pos_q2].pairs_n_second);

    const size_t pair_q1_second = buf.same_score_q1[pos_q1].pairs_second_beg + res.solution.first - buf.same_score_q1[pos_q1].pairs_offset;
    const size_t pair_q2_second = buf.same_score_q2[pos_q2].pairs_second_beg + res.solution.second - buf.same_score_q2[pos_q2].pairs_offset;
    std::pair<size_t, size_t> pair_q1 = {buf.same_score_q1[pos_q1].pairs_first, pair_q1_second};
    std::pair<size_t, size_t> pair_q2 = {buf.same_score_q2[pos_q2].pairs_first, pair_q2_second};

    std::vector<size_t> solution_1d(subset_sum_1d.size());

    assert(set1_weights[pair_q1.first] + set2_weights_sorted_asc[pair_q1.second] + set3_weights[pair_q2.first] + set4_weights_sorted_desc[pair_q2.second] == subset_sum_1d_rhs);

    auto v1 = unpack_mask(set1_subsets[pair_q1.first]);
    auto v2 = unpack_mask(set2_subsets_sorted_asc[pair_q1.second]);
    auto v3 = unpack_mask(set3_subsets[pair_q2.first]);
    auto v4 = unpack_mask(set4_subsets_sorted_desc[pair_q2.second]);
    const std::vector<const std::vector<size_t> *> vectors = {&v1, &v2, &v3, &v4};

    size_t len;
    concat_vectors(solution_1d, len, vectors, offsets);

    /* We found a solution. Construct it, print it, and return. */
    if (!ms_inst.is_solution_feasible(solution_1d, len))
    {
        printf("Error, solution is not feasible!\n");
        exit(1);
    }

    printf("Found market share solution from SS-Algorithm!\n");
    print_four_list_solution(set1_subsets[pair_q1.first], set2_subsets_sorted_asc[pair_q1.second], set3_subsets[pair_q2.first], set4_subsets_sorted_desc[pair_q2.second], list1, list2, list3, list4);

    if (!instance_name.empty())
        write_four_list_solution_to_file(set1_subsets[pair_q1.first], set2_subsets_sorted_asc[pair_q1.second], set3_subsets[pair_q2.first], set4_subsets_sorted_desc[pair_q2.second], list1, list2, list3, list4, instance_name);
    return true;
}

// Analog Grup A4: window kardinalitas [k_min,k_max] utk subset-sum 1D.
template <typename T>
std::pair<size_t, size_t> compute_k_window(const std::vector<T> &values, T target)
{
    auto sorted_vals = values;
    std::sort(sorted_vals.begin(), sorted_vals.end(), std::greater<T>());
    size_t n = sorted_vals.size();
    std::vector<T> P(n + 1, 0), S(n + 1, 0);
    for (size_t i = 0; i < n; ++i) P[i + 1] = P[i] + sorted_vals[i];
    for (size_t i = n; i-- > 0; ) S[i] = S[i + 1] + sorted_vals[i];

    size_t k_min = SIZE_MAX, k_max = 0; bool found = false;
    for (size_t k = 1; k <= n; ++k) {
        if (P[k] >= target && k_min == SIZE_MAX) k_min = k;
        if (S[n - k] <= target) { k_max = k; found = true; }
    }
    if (k_min == SIZE_MAX || !found || k_max < k_min) return {1, 0}; // window kosong
    return {k_min, k_max};
}

// Analog Grup F: k dengan estimasi share solusi tertinggi di dalam window.
template <typename T>
size_t compute_peak_k(const std::vector<T> &values, T target, size_t k_min, size_t k_max)
{
    (void)target;
    auto sorted_vals = values;
    std::sort(sorted_vals.begin(), sorted_vals.end(), std::greater<T>());
    size_t n = sorted_vals.size();
    std::vector<T> P(n + 1, 0), S(n + 1, 0);
    for (size_t i = 0; i < n; ++i) P[i + 1] = P[i] + sorted_vals[i];
    for (size_t i = n; i-- > 0; ) S[i] = S[i + 1] + sorted_vals[i];

    std::vector<double> logExp; double mx = -std::numeric_limits<double>::infinity();
    for (size_t k = k_min; k <= k_max && k <= n; ++k) {
        double range = double(P[k]) - double(S[n - k]) + 1.0;
        if (range < 1.0) range = 1.0;
        double logC = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
        logExp.push_back(logC - std::log(range));
        mx = std::max(mx, logExp.back());
    }
    double denom = 0; for (double v : logExp) denom += std::exp(v - mx);
    double best = -1; size_t peak_k = k_min; size_t idx = 0;
    for (size_t k = k_min; k <= k_max && k <= n; ++k, ++idx) {
        double share = std::exp(logExp[idx] - mx) / denom;
        if (share > best) { best = share; peak_k = k; }
    }
    return peak_k;
}

template <typename T>
std::vector<std::pair<size_t, double>> compute_k_priority_profile(const std::vector<T> &values, T target, size_t k_min, size_t k_max)
{
    (void)target;
    auto sorted_vals = values;
    std::sort(sorted_vals.begin(), sorted_vals.end(), std::greater<T>());
    const size_t n = sorted_vals.size();
    std::vector<T> P(n + 1, 0), S(n + 1, 0);
    for (size_t i = 0; i < n; ++i)
        P[i + 1] = P[i] + sorted_vals[i];
    for (size_t i = n; i-- > 0;)
        S[i] = S[i + 1] + sorted_vals[i];

    std::vector<std::pair<size_t, double>> log_profile;
    double mx = -std::numeric_limits<double>::infinity();
    for (size_t k = k_min; k <= k_max && k <= n; ++k)
    {
        double range = static_cast<double>(P[k]) - static_cast<double>(S[n - k]) + 1.0;
        if (range < 1.0)
            range = 1.0;
        const double logC = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
        const double logExp = logC - std::log(range);
        log_profile.emplace_back(k, logExp);
        mx = std::max(mx, logExp);
    }

    double denom = 0.0;
    for (const auto &entry : log_profile)
        denom += std::exp(entry.second - mx);

    std::vector<std::pair<size_t, double>> profile;
    profile.reserve(log_profile.size());
    for (const auto &entry : log_profile)
    {
        const double share = denom > 0.0 ? std::exp(entry.second - mx) / denom : 0.0;
        profile.emplace_back(entry.first, share);
    }

    std::sort(profile.begin(), profile.end(), [](const auto &a, const auto &b)
              {
                  if (a.second != b.second)
                      return a.second > b.second;
                  return a.first < b.first;
              });
    return profile;
}

// Berapa rentang kardinalitas quarter INI yg masih mungkin, mengingat
// 3 quarter lain bisa menyumbang antara 0..n_others_total.
inline std::pair<size_t, size_t> quarter_k_bounds(size_t n_this, size_t n_others_total,
                                                  size_t k_target_lo, size_t k_target_hi)
{
    size_t lo = (k_target_lo > n_others_total) ? (k_target_lo - n_others_total) : 0;
    size_t hi = std::min(n_this, k_target_hi);
    return {lo, hi};
}

template <typename T>
std::pair<std::vector<T>, std::vector<size_t>>
filter_by_cardinality(const std::vector<T> &weights, size_t k_lo, size_t k_hi)
{
    const size_t n = weights.size();

    /* [OPT] Pass 1: kardinalitas via _mm_popcnt_u64 (SSE4.2 eksplisit).
     * Hitung n_valid dahulu -> pre-alokasi tepat, zero realloc. */
    size_t n_valid = 0;
    std::vector<uint8_t> valid_mask(n);
    for (size_t i = 0; i < n; ++i)
    {
        const size_t card = static_cast<size_t>(
            _mm_popcnt_u64(static_cast<unsigned long long>(i)));
        const uint8_t ok = (card >= k_lo && card <= k_hi) ? 1u : 0u;
        valid_mask[i] = ok;
        n_valid += ok;
    }

    /* Pre-alokasi ukuran tepat */
    std::vector<T>      fw(n_valid);
    std::vector<size_t> fm(n_valid);

    /* [OPT] Pass 2: branchless scatter -> nol branch misprediction.
     * Tulis ke fw[out]/fm[out] selalu; 'out' maju hanya jika valid.
     * Branch-free -> compiler dapat auto-vektorisasi loop ini. */
    size_t out = 0;
    for (size_t i = 0; i < n; ++i)
    {
        fw[out] = weights[i];
        fm[out] = i;
        out += valid_mask[i];
    }

    return {fw, fm};
}

template <typename T>
bool print_and_write_1d_ss_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4,
                                    const std::vector<T> &list1, const std::vector<T> &list2,
                                    const std::vector<T> &list3, const std::vector<T> &list4,
                                    const std::string &instance_name, T target)
{
    const std::vector<const std::vector<T> *> lists = {&list1, &list2, &list3, &list4};
    const std::vector<size_t> masks = {index_list1, index_list2, index_list3, index_list4};

    std::string bitstring;
    std::vector<size_t> indices;
    T sum = 0;
    size_t offset = 0;

    for (size_t i_list = 0; i_list < lists.size(); ++i_list)
    {
        const auto &list = *lists[i_list];
        const size_t mask = masks[i_list];
        for (size_t i = 0; i < list.size(); ++i)
        {
            const bool selected = (mask & (1ULL << i)) != 0;
            bitstring.push_back(selected ? '1' : '0');
            if (selected)
            {
                indices.push_back(offset + i);
                sum += list[i];
            }
        }
        offset += list.size();
    }

    if (sum != target)
    {
        std::cout << "Error, reconstructed 1D SS solution sum " << sum
                  << " != target " << target << "\n";
        return false;
    }

    std::cout << "Found subset-sum solution from Schroeppel-Shamir!\n";
    std::cout << "Bitstring: " << bitstring << "\n";
    std::cout << "Indices (0-based):";
    for (size_t idx : indices)
        std::cout << " " << idx;
    std::cout << "\nThe sum is " << sum << "\n";

    if (!instance_name.empty())
    {
        const std::string sol_name = instance_name + ".sol";
        std::cout << "Writing solution to " << sol_name << "\n";
        std::ofstream sol_file(sol_name);
        sol_file << bitstring << std::endl;
    }

    return true;
}

template <typename T>
bool shroeppel_shamir_1d(const std::vector<T> &values, T target, const std::string &instance_name,
                         int k_radius = -1)
{
    std::cout << "Running 1D Schroeppel-Shamir" << std::endl;
    std::cout << "Running with " << omp_get_max_threads() << " threads" << std::endl;
    std::cout << "Target: " << target << "\n";

    if (values.empty())
        return false;

    const size_t split_index1 = values.size() / 4;
    const size_t split_index2 = values.size() / 2;
    const size_t split_index3 = 3 * values.size() / 4;
    const size_t max_quarter = std::max({split_index1, split_index2 - split_index1, split_index3 - split_index2, values.size() - split_index3});
    if (max_quarter >= sizeof(size_t) * CHAR_BIT)
    {
        std::cout << "Quarter size " << max_quarter << " is too large for mask-based subset generation.\n";
        return false;
    }

    auto [k_min, k_max] = compute_k_window(values, target);
    if (k_min > k_max)
    {
        std::cout << "UNSAT (cardinality window kosong)!\n";
        return false;
    }

    const auto k_priority = compute_k_priority_profile(values, target, k_min, k_max);
    if (k_priority.empty())
        return false;

    const size_t peak_k = k_priority.front().first;
    std::cout << "k-window=[" << k_min << "," << k_max << "], peak_k=" << peak_k << "\n";
    std::cout << "k priority:";
    for (const auto &entry : k_priority)
        std::cout << " " << entry.first << "(" << std::fixed << std::setprecision(2) << entry.second * 100.0 << "%)";
    std::cout << "\n";
    std::cout << "Splitting sets into [0, " << split_index1 - 1 << "]; ["
              << split_index1 << ", " << split_index2 - 1 << "]; ["
              << split_index2 << ", " << split_index3 - 1 << "]; ["
              << split_index3 << ", " << values.size() << "]\n";

    auto profiler = std::make_unique<ScopedProfiler>("Setup time                  ");
    auto profilerTotal = std::make_unique<ScopedProfiler>("Solution time               ");

    std::vector<T> list1(values.begin(), values.begin() + split_index1);
    std::vector<T> list2(values.begin() + split_index1, values.begin() + split_index2);
    std::vector<T> list3(values.begin() + split_index2, values.begin() + split_index3);
    std::vector<T> list4(values.begin() + split_index3, values.end());

    auto set1_weights_full = generate_subsets(list1);
    auto set2_weights_full = generate_subsets(list2);
    auto set3_weights_full = generate_subsets(list3);
    auto set4_weights_full = generate_subsets(list4);

    const size_t n1 = list1.size(), n2 = list2.size(), n3 = list3.size(), n4 = list4.size();
    /* [OPT] k_radius: -1 = semua k, 0 = hanya peak_k, r = peak_k ± r */
    for (const auto &k_entry : k_priority)
    {
    if (k_radius >= 0 && std::abs((int64_t)k_entry.first - (int64_t)peak_k) > (int64_t)k_radius)
        continue;
    const size_t k_focus_lo = k_entry.first;
    const size_t k_focus_hi = k_entry.first;
    std::cout << "\nTrying k=" << k_entry.first << " (priority share "
              << std::fixed << std::setprecision(2) << k_entry.second * 100.0 << "%)\n";
    auto b1 = quarter_k_bounds(n1, n2 + n3 + n4, k_focus_lo, k_focus_hi);
    auto b2 = quarter_k_bounds(n2, n1 + n3 + n4, k_focus_lo, k_focus_hi);
    auto b3 = quarter_k_bounds(n3, n1 + n2 + n4, k_focus_lo, k_focus_hi);
    auto b4 = quarter_k_bounds(n4, n1 + n2 + n3, k_focus_lo, k_focus_hi);

    auto filtered1 = filter_by_cardinality(set1_weights_full, b1.first, b1.second);
    auto filtered2 = filter_by_cardinality(set2_weights_full, b2.first, b2.second);
    auto filtered3 = filter_by_cardinality(set3_weights_full, b3.first, b3.second);
    auto filtered4 = filter_by_cardinality(set4_weights_full, b4.first, b4.second);

    auto &set1_weights = filtered1.first;
    auto &set1_subsets = filtered1.second;
    auto &set2_weights = filtered2.first;
    auto &set2_subsets = filtered2.second;
    auto &set3_weights = filtered3.first;
    auto &set3_subsets = filtered3.second;
    auto &set4_weights = filtered4.first;
    auto &set4_subsets = filtered4.second;

    auto asc_indices_set2_weights = sort_indices(set2_weights, true);
    auto desc_indices_set4_weights = sort_indices(set4_weights, false);
    const auto set2_weights_sorted_asc = apply_permutation(set2_weights, asc_indices_set2_weights);
    const auto set2_subsets_sorted_asc = apply_permutation(set2_subsets, asc_indices_set2_weights);
    const auto set4_weights_sorted_desc = apply_permutation(set4_weights, desc_indices_set4_weights);
    const auto set4_subsets_sorted_desc = apply_permutation(set4_subsets, desc_indices_set4_weights);

    if (set2_weights_sorted_asc.empty() || set4_weights_sorted_desc.empty())
        continue;

    auto min_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return set1_weights[a1.first] + set2_weights_sorted_asc[a1.second] > set1_weights[a2.first] + set2_weights_sorted_asc[a2.second];
    };

    auto max_cmp = [&](std::pair<size_t, size_t> a1, std::pair<size_t, size_t> a2) -> bool
    {
        return set3_weights[a1.first] + set4_weights_sorted_desc[a1.second] < set3_weights[a2.first] + set4_weights_sorted_desc[a2.second];
    };

    std::vector<std::pair<size_t, size_t>> heap1;
    heap1.reserve(set1_weights.size());
    std::vector<std::pair<size_t, size_t>> heap2;
    heap2.reserve(set3_weights.size());

    for (size_t i = 0; i < set1_weights.size(); ++i)
    {
        if (set1_weights[i] + set2_weights_sorted_asc[0] <= target)
            heap1.emplace_back(i, 0);
    }

    for (size_t i = 0; i < set3_weights.size(); ++i)
        heap2.emplace_back(i, 0);

    std::make_heap(heap1.begin(), heap1.end(), min_cmp);
    std::make_heap(heap2.begin(), heap2.end(), max_cmp);

    std::cout << "Running the search loop\n\n";
    profiler = std::make_unique<ScopedProfiler>("List traversal              ");

    size_t i_iter_checking = 0;
    while (!heap1.empty() && !heap2.empty())
    {
        const T score_pair1 = set1_weights[heap1.front().first] + set2_weights_sorted_asc[heap1.front().second];
        const T score_pair2 = set3_weights[heap2.front().first] + set4_weights_sorted_desc[heap2.front().second];
        const T score = score_pair1 + score_pair2;

        if (score == target)
        {
            std::vector<PairsTuple> same_score_q1, same_score_q2;
            std::vector<size_t> chunks_q1_beg{0}, chunks_q2_beg{0};
            std::vector<size_t> chunks_q1_n_pairs, chunks_q2_n_pairs;
            size_t n_pairs_q1 = 0, n_pairs_q2 = 0;

            extract_pairs_from_heap<true, T>(same_score_q1, heap1, score_pair1, chunks_q1_beg, chunks_q1_n_pairs, n_pairs_q1, set1_weights, set2_weights, set2_weights_sorted_asc, min_cmp);
            extract_pairs_from_heap<false, T>(same_score_q2, heap2, score_pair2, chunks_q2_beg, chunks_q2_n_pairs, n_pairs_q2, set3_weights, set4_weights, set4_weights_sorted_desc, max_cmp);

            ++i_iter_checking;
            print_info_line(i_iter_checking, profilerTotal->elapsed(), score_pair1, score_pair2, n_pairs_q1, n_pairs_q2);

            if (!same_score_q1.empty() && !same_score_q2.empty())
            {
                const auto &q1 = same_score_q1.front();
                const auto &q2 = same_score_q2.front();
                const size_t q1_second = q1.pairs_second_beg;
                const size_t q2_second = q2.pairs_second_beg;

                return print_and_write_1d_ss_solution(
                    set1_subsets[q1.pairs_first],
                    set2_subsets_sorted_asc[q1_second],
                    set3_subsets[q2.pairs_first],
                    set4_subsets_sorted_desc[q2_second],
                    list1, list2, list3, list4, instance_name, target);
            }
        }
        else if (score < target)
        {
            const auto pair1 = heap1.front();
            size_t pos_set2_weights = pair1.second;
            std::pop_heap(heap1.begin(), heap1.end(), min_cmp);
            heap1.pop_back();
            ++pos_set2_weights;

            while (pos_set2_weights + 1 < set2_weights.size() &&
                   (set2_weights_sorted_asc[pos_set2_weights] == set2_weights_sorted_asc[pair1.second] ||
                    (set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] + score_pair2) < target))
                ++pos_set2_weights;

            if (pos_set2_weights < set2_weights.size() && set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] <= target)
            {
                heap1.emplace_back(pair1.first, pos_set2_weights);
                std::push_heap(heap1.begin(), heap1.end(), min_cmp);
            }
        }
        else
        {
            const auto pair2 = heap2.front();
            size_t pos_set4_weights = pair2.second;
            std::pop_heap(heap2.begin(), heap2.end(), max_cmp);
            heap2.pop_back();
            ++pos_set4_weights;

            while (pos_set4_weights + 1 < set4_weights.size() &&
                   (set4_weights_sorted_desc[pos_set4_weights] == set4_weights_sorted_desc[pair2.second] ||
                    (score_pair1 + set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights]) > target))
                ++pos_set4_weights;

            if (pos_set4_weights < set4_weights.size() && set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights] <= target)
            {
                heap2.emplace_back(pair2.first, pos_set4_weights);
                std::push_heap(heap2.begin(), heap2.end(), max_cmp);
            }
        }
    }
    }

    return false;
}

template <typename T>
bool shroeppel_shamir_dim_reduced(const MarkShareFeas &ms_inst, bool run_on_gpu, const std::string &instance_name, size_t n_reduce_max,
                                  int k_radius = -1)
{
    std::cout << "Running reduced dim shroeppel shamir" << std::endl;
    std::cout << "Running with " << omp_get_max_threads() << " threads" << std::endl;

    /* First, attempt some dimensionality reduction/perfect hashing.
     * We know that 0 <= A[i][j] <= 200 and thus sum_j A[i][j] <= n * 200.
     *
     * So, reserving n * 200 + 1 intervals for each of the entries i of a vector A.j we have an overlap free combination.
     * The amount of dimension we can remove this way is limited by the maximum value of T, the index type we are using for the 1-dimensional subset sum problem.
     */
    constexpr size_t max_coeff = 200;
    const size_t n_cols = ms_inst.n();

    const auto [max_reduce_dim, basis] = max_encodable_dimension<T>(max_coeff, n_cols);
    const size_t reduce_dim = std::max(size_t(1), std::min(std::min(ms_inst.m(), max_reduce_dim), n_reduce_max));
    const size_t leftover_dim = ms_inst.m() - reduce_dim;

    printf("Reducing %zu dimensions for Shroeppel-Shamir - leaving %zu for verification\n", reduce_dim, leftover_dim);

    const auto reduced_problem = ms_inst.reduce_first_dimensions(reduce_dim, basis);
    const auto &subset_sum_1d = reduced_problem.first;
    const auto subset_sum_1d_rhs = reduced_problem.second;

    // >>> BARU: hitung window k & urutan prioritas Grup F utk instance 1D ini
    auto [k_min, k_max] = compute_k_window(subset_sum_1d, subset_sum_1d_rhs);
    if (k_min > k_max) { printf("UNSAT (cardinality window kosong)!\n"); return false; }
    const auto k_priority = compute_k_priority_profile(subset_sum_1d, subset_sum_1d_rhs, k_min, k_max);
    if (k_priority.empty())
        return false;
    size_t peak_k = k_priority.front().first;
    printf("k-window=[%zu,%zu], peak_k=%zu\n", k_min, k_max, peak_k);
    std::cout << "k priority:";
    for (const auto &entry : k_priority)
        std::cout << " " << entry.first << "(" << std::fixed << std::setprecision(2) << entry.second * 100.0 << "%)";
    std::cout << "\n";
    // <

    const size_t split_index1 = subset_sum_1d.size() / 4;
    const size_t split_index2 = subset_sum_1d.size() / 2;
    const size_t split_index3 = 3 * subset_sum_1d.size() / 4;
    printf("Splitting sets into [0, %zu]; [%zu, %zu]; [%zu, %zu]; [%zu, %zu]\n", split_index1 - 1, split_index1, split_index2 - 1, split_index2, split_index3 - 1, split_index3, subset_sum_1d.size());

    auto profiler = std::make_unique<ScopedProfiler>("Setup time                  ");
    auto profilerTotal = std::make_unique<ScopedProfiler>("Solution time               ");

    std::vector<T> list1(subset_sum_1d.begin(), subset_sum_1d.begin() + split_index1);
    std::vector<T> list2(subset_sum_1d.begin() + split_index1, subset_sum_1d.begin() + split_index2);
    std::vector<T> list3(subset_sum_1d.begin() + split_index2, subset_sum_1d.begin() + split_index3);
    std::vector<T> list4(subset_sum_1d.begin() + split_index3, subset_sum_1d.end());
    assert(list1.size() + list2.size() + list3.size() + list4.size() == subset_sum_1d.size());

    const std::vector<size_t> offsets = {0, list1.size(), list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};
    const std::vector<size_t> offsetsQ1 = {0, list1.size()};
    const std::vector<size_t> offsetsQ2 = {list1.size() + list2.size(), list1.size() + list2.size() + list3.size()};

    auto set1_weights_full = generate_subsets(list1);
    auto set2_weights_full = generate_subsets(list2);
    auto set3_weights_full = generate_subsets(list3);
    auto set4_weights_full = generate_subsets(list4);

    size_t n1 = list1.size(), n2 = list2.size(), n3 = list3.size(), n4 = list4.size();
    /* [OPT] k_radius: -1 = semua k, 0 = hanya peak_k, r = peak_k ± r */
    for (const auto &k_entry : k_priority)
    {
    if (k_radius >= 0 && std::abs((int64_t)k_entry.first - (int64_t)peak_k) > (int64_t)k_radius)
        continue;
    std::cout << "\nTrying k=" << k_entry.first << " (priority share "
              << std::fixed << std::setprecision(2) << k_entry.second * 100.0 << "%)\n";
    size_t k_focus_lo = k_entry.first, k_focus_hi = k_entry.first;

    auto b1 = quarter_k_bounds(n1, n2 + n3 + n4, k_focus_lo, k_focus_hi);
    auto b2 = quarter_k_bounds(n2, n1 + n3 + n4, k_focus_lo, k_focus_hi);
    auto b3 = quarter_k_bounds(n3, n1 + n2 + n4, k_focus_lo, k_focus_hi);
    auto b4 = quarter_k_bounds(n4, n1 + n2 + n3, k_focus_lo, k_focus_hi);

    auto filtered1 = filter_by_cardinality(set1_weights_full, b1.first, b1.second);
    auto filtered2 = filter_by_cardinality(set2_weights_full, b2.first, b2.second);
    auto filtered3 = filter_by_cardinality(set3_weights_full, b3.first, b3.second);
    auto filtered4 = filter_by_cardinality(set4_weights_full, b4.first, b4.second);

    auto &set1_weights = filtered1.first;
    auto &set1_subsets = filtered1.second;
    auto &set2_weights = filtered2.first;
    auto &set2_subsets = filtered2.second;
    auto &set3_weights = filtered3.first;
    auto &set3_subsets = filtered3.second;
    auto &set4_weights = filtered4.first;
    auto &set4_subsets = filtered4.second;
    // <

    if (set2_weights.empty() || set4_weights.empty())
        continue;

    /* Sort set2_weights ascending, set4_weights descending. */
    auto asc_indices_set2_weights = sort_indices(set2_weights, true);
    auto desc_indices_set4_weights = sort_indices(set4_weights, false);

    const auto set2_weights_sorted_asc = apply_permutation(set2_weights, asc_indices_set2_weights);
    const auto set2_subsets_sorted_asc = apply_permutation(set2_subsets, asc_indices_set2_weights);

    const auto set4_weights_sorted_desc = apply_permutation(set4_weights, desc_indices_set4_weights);
    const auto set4_subsets_sorted_desc = apply_permutation(set4_subsets, desc_indices_set4_weights);

    std::vector<size_t> set1_scores(set1_subsets.size() * leftover_dim);
    std::vector<size_t> set2_scores_sorted_asc(set2_subsets.size() * leftover_dim);
    std::vector<size_t> set3_scores(set3_subsets.size() * leftover_dim);
    std::vector<size_t> set4_scores_sorted_desc(set4_subsets.size() * leftover_dim);

    compute_scores_cpu(ms_inst, set1_scores, set1_subsets, offsets[0], reduce_dim);
    compute_scores_cpu(ms_inst, set2_scores_sorted_asc, set2_subsets_sorted_asc, offsets[1], reduce_dim);
    compute_scores_cpu(ms_inst, set3_scores, set3_subsets, offsets[2], reduce_dim);
    compute_scores_cpu(ms_inst, set4_scores_sorted_desc, set4_subsets_sorted_desc, offsets[3], reduce_dim);

#ifdef WITH_GPU
    GpuData gpu_data(ms_inst, set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc);
#endif

    /* Create the priority queues q1 consisting of pairs {(i, 0) | i \in set1_weights} and q2 consisting of {(i, 0) | i \in set3_weights}. The priority/score for a pair (i, j)
     * is given set1_weights[i] + set2_weights[j] if the pair is in q1 and set3_weights[i] + set4_weights[j] if the pair is in q2. */

    /* Compare returns true if the first argument comes BEFORE the second argument. Since however the priority queue outputs the largest element first,
     * we have to flip the > signs. */
    auto min_cmp = [&](std::pair<T, T> a1, std::pair<T, T> a2) -> bool
    {
        return set1_weights[a1.first] + set2_weights_sorted_asc[a1.second] > set1_weights[a2.first] + set2_weights_sorted_asc[a2.second];
    };

    auto max_cmp = [&](std::pair<T, T> a1, std::pair<T, T> a2) -> bool
    {
        return set3_weights[a1.first] + set4_weights_sorted_desc[a1.second] < set3_weights[a2.first] + set4_weights_sorted_desc[a2.second];
    };

    /* Vectors used to count the number of elements extracted from q1/q2 with the same solution value. */
    /* Each tuple {a, b, c, d} will describe the range of pairs <a, b> ... <a, b + c - 1>; d denotes the offset of the pairs within a list of all pairs. */
    std::vector<std::pair<size_t, size_t>> heap1;
    heap1.reserve(set1_weights.size());
    std::vector<std::pair<size_t, size_t>> heap2;
    heap2.reserve(set3_weights.size());

    // TODO: the initial insert can likely be improved by simple sorting.
    for (size_t i = 0; i < set1_weights.size(); ++i)
    {
        /* If already the sum of these 2 elements is greater than the right hand side we can skip them. Subsequent combinations (e.g. with higher pos_subset2)
         * will only be even larger. */
        if (set1_weights[i] + set2_weights_sorted_asc[0] <= subset_sum_1d_rhs)
            heap1.emplace_back(i, 0);
    }

    for (size_t i = 0; i < set3_weights.size(); ++i)
        heap2.emplace_back(i, 0);

    std::make_heap(heap1.begin(), heap1.end(), min_cmp);
    std::make_heap(heap2.begin(), heap2.end(), max_cmp);

    printf("Running the search loop\n\n");

    profiler = std::make_unique<ScopedProfiler>("List traversal              ");

    PipelineBuffer buffers[2];
    std::future<EvalResult> eval_future[2];

    buffers[0].same_score_q1.reserve(100000);
    buffers[0].same_score_q2.reserve(100000);
    buffers[1].same_score_q1.reserve(100000);
    buffers[1].same_score_q2.reserve(100000);

    buffers[0].state = EMPTY;
    buffers[1].state = EMPTY;

    size_t curr = 0;
    size_t next = 1;
    size_t i_iter_checking = 0;

    while (!heap1.empty() && !heap2.empty())
    {
        /* score_pair1 is the currently lowest score in {set1_weights, set2_weights} we are still considering */
        const T score_pair1 = set1_weights[heap1.front().first] + set2_weights_sorted_asc[heap1.front().second];
        /* score_pair2 is the currently highest score in {set3_weights, set4_weights} we are still considering */
        const T score_pair2 = set3_weights[heap2.front().first] + set4_weights_sorted_desc[heap2.front().second];

        const T score = score_pair1 + score_pair2;

        if (score == subset_sum_1d_rhs)
        {
            /* Extract all tuples from both lists with equal scores. Potentially, we subdivide extracted tuples into chunks to not overflow the GPU memory. We remember the start and number of pairs in each chunk. */
            auto &buf_curr = buffers[curr];
            auto &buf_next = buffers[next];
            assert(buf_curr.state == EMPTY);
            buf_curr.state = EXTRACTING;

            /* Reset the buffer. */
            /* Clear vectors but keep their old capacity. */
            buf_curr.same_score_q1.clear();
            buf_curr.same_score_q2.clear();

            buf_curr.chunks_q1_beg.clear();
            buf_curr.chunks_q2_beg.clear();

            buf_curr.chunks_q1_n_pairs.clear();
            buf_curr.chunks_q2_n_pairs.clear();

            buf_curr.chunks_q1_beg.push_back(0);
            buf_curr.chunks_q2_beg.push_back(0);

            ++i_iter_checking;

            buf_curr.n_pairs_q1 = 0;
            buf_curr.n_pairs_q2 = 0;

            auto profiler_cand_extraction = std::make_unique<ScopedProfiler>("Candidate extraction        ");

            /* In CPU parallel, extract equal tuples from each heap. */
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    extract_pairs_from_heap<true, T>(buf_curr.same_score_q1, heap1, score_pair1, buf_curr.chunks_q1_beg, buf_curr.chunks_q1_n_pairs, buf_curr.n_pairs_q1, set1_weights, set2_weights, set2_weights_sorted_asc, min_cmp);
                }
#pragma omp section
                {
                    extract_pairs_from_heap<false, T>(buf_curr.same_score_q2, heap2, score_pair2, buf_curr.chunks_q2_beg, buf_curr.chunks_q2_n_pairs, buf_curr.n_pairs_q2, set3_weights, set4_weights, set4_weights_sorted_desc, max_cmp);
                }
            }
            buf_curr.state = READY_FOR_EVAL;

            profiler_cand_extraction.reset();
            print_info_line(
#ifdef WITH_GPU
                gpu_data,
                run_on_gpu,
#endif
                i_iter_checking, profilerTotal->elapsed(), score_pair1, score_pair2, buf_curr.n_pairs_q1, buf_curr.n_pairs_q2);

            /* Before submitting the current buffer, wait until the last one is finished. */
            if (buf_next.state != EMPTY)
            {
                eval_future[next].wait();
                const auto &result = eval_future[next].get();
                assert(buf_next.state == EVALUATED);

                /* Check the result - break if we are done. */
                if (result.found)
                {
                    return print_and_verify_solution(buf_next, result, ms_inst, subset_sum_1d, offsets, asc_indices_set2_weights, desc_indices_set4_weights, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, list1, list2, list3, list4, instance_name
#ifndef NDEBUG
                                                     ,
                                                     subset_sum_1d_rhs,
                                                     set1_weights,
                                                     set2_weights_sorted_asc, set3_weights, set4_weights_sorted_desc
#endif
                    );
                }

                buf_next.state = EMPTY;
            }
            assert(buf_next.state == EMPTY);

            buf_curr.state = EVALUATING;
            /* Launch evaluation in background thread. */
            eval_future[curr] = std::async(std::launch::async, [&]()
                                           { return evaluate_gpu_or_cpu(buf_curr,
#ifdef WITH_GPU
                                                                        gpu_data,
#endif
                                                                        ms_inst, reduce_dim, set1_scores, set2_scores_sorted_asc, set3_scores, set4_scores_sorted_desc, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, subset_sum_1d, offsets, run_on_gpu); });

            /* Switch buffer. */
            curr = next;
            next = 1 - curr;
        }
        else if (score < subset_sum_1d_rhs)
        {
            const auto pair1 = heap1.front();
            size_t pos_set2_weights = pair1.second;

            std::pop_heap(heap1.begin(), heap1.end(), min_cmp);
            heap1.pop_back();

            ++pos_set2_weights;

            while (pos_set2_weights + 1 < set2_weights.size() && (set2_weights_sorted_asc[pos_set2_weights] == set2_weights_sorted_asc[pair1.second] || (set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] + score_pair2) < subset_sum_1d_rhs))
                ++pos_set2_weights;

            /* Again, the element in q1 can only increase (or stay equal). So ignore elements that are already too big. */
            if (pos_set2_weights < set2_weights.size() && set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights] <= subset_sum_1d_rhs)
            {
                heap1.emplace_back(pair1.first, pos_set2_weights);
                std::push_heap(heap1.begin(), heap1.end(), min_cmp);
            }
        }
        else if (score > subset_sum_1d_rhs)
        {
            const auto pair2 = heap2.front();
            size_t pos_set4_weights = pair2.second;

            std::pop_heap(heap2.begin(), heap2.end(), max_cmp);
            heap2.pop_back();

            ++pos_set4_weights;

            /* Skip all entries in set4_weights until we find a smaller one. */
            while (pos_set4_weights + 1 < set4_weights.size() && (set4_weights_sorted_desc[pos_set4_weights] == set4_weights_sorted_desc[pair2.second] || (score_pair1 + set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights]) > subset_sum_1d_rhs))
                ++pos_set4_weights;

            if (pos_set4_weights < set4_weights.size() && set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights] <= subset_sum_1d_rhs)
            {
                heap2.emplace_back(pair2.first, pos_set4_weights);
                std::push_heap(heap2.begin(), heap2.end(), max_cmp);
            }
        }
    }

    profiler.reset();

    for (size_t i_buf = 0; i_buf < 2; ++i_buf)
    {
        auto &buf = buffers[i_buf];
        if (buf.state == EVALUATING)
        {
            eval_future[i_buf].wait();
            const auto &result = eval_future[i_buf].get();
            assert(buf.state == EVALUATED);

            if (result.found)
            {
                return print_and_verify_solution(buf, result, ms_inst, subset_sum_1d, offsets, asc_indices_set2_weights, desc_indices_set4_weights, set1_subsets, set2_subsets_sorted_asc, set3_subsets, set4_subsets_sorted_desc, list1, list2, list3, list4, instance_name
#ifndef NDEBUG
                                                 ,
                                                 subset_sum_1d_rhs,
                                                 set1_weights,
                                                 set2_weights_sorted_asc, set3_weights, set4_weights_sorted_desc
#endif
                );
            }

            buf.state = EMPTY;
        }
    }
    }

    return false;
}

std::string get_filename_without_extension(const std::string &filePath)
{
    // Find the last path separator.
    size_t lastSlash = filePath.find_last_of("/\\");
    size_t start = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;

    // Find the last dot after the last slash
    size_t lastDot = filePath.find_last_of('.');
    size_t end = (lastDot == std::string::npos || lastDot < start) ? filePath.length() : lastDot;

    // Extract the filename without extension
    return filePath.substr(start, end - start);
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("markshare");

    std::string path = "";
    size_t n_iter = 1;
    size_t seed = 2025;
    size_t m = 5;
    size_t n = 0;
    size_t n_reduce = 0;
    size_t k = 100;
    size_t n_threads = 0;
    bool check_only = false;
    bool write_prb = false;
    int k_radius = -1; /* [OPT] -1 = semua k, 0 = peak_k saja, r = peak_k ± r */
    int runs = 0;
    double timeout_sec = 0.0;
    std::string solver = "ter";

    program.add_argument("-m", "--m")
        .store_into(m)
        .help("Number of rows of the markshare problem.");

    program.add_argument("-n", "--n")
        .store_into(n)
        .help("Number of columns of the markshare problem. Set to (m - 1) * 10 if not given. ");

    program.add_argument("-k", "--k")
        .store_into(k)
        .help("Coefficients are generated in the range [0, k).")
        .default_value(100);

    program.add_argument("--reduce")
        .store_into(n_reduce)
        .help("Number of rows (max) to be reduced. Only effective if --reduced is set. ")
        .default_value(0);

    program.add_argument("-s", "--seed")
        .store_into(seed)
        .help("Random seed for instance generation.")
        .default_value(2025);

    program.add_argument("-i", "--iter")
        .store_into(n_iter)
        .help("Number of problems to solve. Seed for problem of iteration i (starting from 0) is seed + i.")
        .default_value(1);

    program.add_argument("-t", "--threads")
        .store_into(n_threads)
        .help("Number of CPU threads. Defaults to all available logical CPU cores.")
        .default_value(0);

    program.add_argument("--gpu")
        .help("Run validation on GPU")
        .flag();

    program.add_argument("--check_only")
        .help("Only parse and verify input metadata/planted witness; do not solve.")
        .flag();

    program.add_argument("--write_prb")
        .help("Write parsed 1D txt/prb input back as MarketShareGpu-style .prb.")
        .flag();

    program.add_argument("-f", "--file")
        .store_into(path)
        .help("Supply instance path to read instance from. Overrides '-m', '-n', '-k', and '-i'");

    program.add_argument("--max_pairs")
        .store_into(max_pairs_per_chunk)
        .help("Maximum number of pairs to be evaluated on the GPU simultaneously. If GPU runs OOM, reduce this number.")
        .default_value(3500000000);

    /* [OPT] k_radius: batasi eksplorasi k di sekitar peak_k */
    program.add_argument("--k_radius")
        .store_into(k_radius)
        .help("Radius eksplorasi k di sekitar peak_k. -1 = semua k (default), 0 = peak_k saja (tercepat), r = peak_k +/- r.")
        .default_value(-1);

    /* [TER] CLI arguments */
    program.add_argument("--runs")
        .store_into(runs)
        .help("Number of runs/attempts for TER solver (default 0 = auto/1 run).")
        .default_value(0);

    program.add_argument("--autorestart")
        .help("Enable autorestart for TER solver until solution is found.")
        .flag();

    program.add_argument("--timeout")
        .store_into(timeout_sec)
        .help("Timeout in seconds for TER solver (0 = no timeout).")
        .default_value(0.0);

    program.add_argument("--solver")
        .store_into(solver)
        .help("Solver engine for 1D subset sum: 'ter' (Li et al. 2025 Ternary) or 'ss' (Schroeppel-Shamir). Default: 'ter'")
        .default_value(std::string("ter"));

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    /* Adjust n. */
    if (n == 0)
        n = (m - 1) * 10;

    check_only = (program["--check_only"] == true);
    write_prb = (program["--write_prb"] == true);

    const unsigned detected_threads = std::max(1u, std::thread::hardware_concurrency());
    const int threads_to_use = static_cast<int>(n_threads == 0 ? detected_threads : n_threads);
    omp_set_num_threads(threads_to_use);
    std::cout << "CPU threads: " << threads_to_use;
    if (n_threads == 0)
        std::cout << " (auto-detected)";
    std::cout << std::endl;

    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter)
    {
        std::string instance_name{};
        const size_t seed_iter = seed + i_iter;
        MarkShareFeas instance;

        /* Generate/read instance. For now, random instances. */
        if (!path.empty())
        {
            instance_name = get_filename_without_extension(path);
            printf("Reading instance from file %s; instance_name %s\n", path.c_str(), instance_name.c_str());

            SubsetSum1D128 instance_1d;
            if (load_hgj_txt_instance(path, instance_1d) || load_prb_1d128_instance(path, instance_1d))
            {
                std::cout << "Detected 1D subset-sum instance (" << instance_1d.source_format << "): n="
                          << instance_1d.values.size() << ", target=" << instance_1d.target << "\n";

                if (instance_1d.has_planted_solution())
                {
                    std::cout << "Planted witness: " << instance_1d.planted_indices.size()
                              << " indices, verify="
                              << (instance_1d.verify_planted_solution() ? "OK" : "FAILED") << "\n";
                }

                if (write_prb)
                {
                    const std::string prb_name = instance_name + ".prb";
                    write_1d_prb(instance_1d, prb_name);
                    std::cout << "Wrote PRB: " << prb_name << "\n";
                }

                if (check_only)
                    continue;

                /* [TER / SS Dispatch] */
                bool found = false;
                std::vector<uint64_t> vals64;
                vals64.reserve(instance_1d.values.size());
                for (const auto &v : instance_1d.values)
                    vals64.push_back(static_cast<uint64_t>(v));
                const uint64_t target64 = static_cast<uint64_t>(instance_1d.target);

                if (solver == "ter")
                {
                    std::cout << "[TER] Menggunakan TER Solver (Yang Li et al., 2025)\n";
                    TerParams params = ter_default_params((int)vals64.size());
                    params.fixed_runs = runs;
                    params.timeout_seconds = timeout_sec;
                    params.use_gpu = (program["--gpu"] == true);
                    if (program["--autorestart"] == true)
                    {
                        params.max_restarts = 0; // infinite until found or timeout
                        params.fixed_runs = 0;
                    }
                    else if (runs > 0)
                    {
                        params.fixed_runs = runs;
                    }
                    else
                    {
                        params.fixed_runs = 1; // single run if no autorestart and no runs specified
                    }

                    TerResult res = ter_solve(vals64, target64, params);
                    found = res.found;
                    if (found)
                    {
                        printf("Found feasible solution by TER in %.3f s (%d runs)!\n", res.elapsed_seconds, res.runs_attempted);
                        std::cout << "Solution indices (" << res.solution_indices.size() << "): ";
                        for (size_t idx : res.solution_indices) std::cout << idx << " ";
                        std::cout << "\n";

                        std::string sol_name = instance_name + ".sol";
                        std::string sol_str(vals64.size(), '0');
                        for (size_t idx : res.solution_indices) {
                            if (idx < sol_str.size()) sol_str[idx] = '1';
                        }
                        std::ofstream sol_file(sol_name);
                        if (sol_file.is_open()) {
                            sol_file << sol_str << "\n";
                            std::cout << "Saved solution to: " << sol_name << "\n";
                        }
                    }
                    else
                    {
                        printf("TER: No solution found within %d runs (%.3f s). Gunakan --autorestart untuk terus mencoba.\n", res.runs_attempted, res.elapsed_seconds);
                    }
                }
                else if (can_fit_size_t(instance_1d))
                {
                    std::cout << "[SS] Menggunakan Schroeppel-Shamir Solver (64-bit)\n";
                    found = shroeppel_shamir_1d<uint64_t>(vals64, target64, instance_name, k_radius);
                }
                else
                {
                    std::cout << "[SS] Menggunakan Schroeppel-Shamir Solver (u128)\n";
                    found = shroeppel_shamir_1d<u128>(instance_1d.values, instance_1d.target, instance_name, k_radius);
                }

                if (found)
                    printf("Found feasible solution!\n");
                else
                    printf("Instance was infeasible or not found by current solver .. \n");

                continue;
            }

            instance = MarkShareFeas(path);
        }
        else
        {
            instance = MarkShareFeas(m, n, k, seed_iter);
            instance_name = "markshare_m_" + std::to_string(m) + "_n_" + std::to_string(n) + "_seed_" + std::to_string(seed_iter);
            instance.write_as_prb(instance_name + ".prb");
        }

        printf("Running markshare: m=%zu, n=%zu, seed=%zu, iter=%zu, nthread=%d\n", instance.m(), instance.n(), seed_iter, i_iter, omp_get_max_threads());
        instance.print();

        /* Solve the instance using one of the available algorithms. */

        /* Create the one dimensional subset sum problem. */
        const bool on_gpu = (program["--gpu"] == true);
#ifndef WITH_GPU
        if (on_gpu)
            std::cout << "Warning: --gpu requested but compiled without CUDA (running on CPU with 4 threads)." << std::endl;
#endif
        const bool found = shroeppel_shamir_dim_reduced<uint64_t>(instance,
#ifdef WITH_GPU
                                                                        on_gpu,
#else
                                                                        false,
#endif
                                                                        instance_name, n_reduce, k_radius);

        if (found)
            printf("Found feasible solution!\n");
        else
            printf("Instance was infeasible .. \n");
    }

    ScopedProfiler::report();

    return 0;
}
