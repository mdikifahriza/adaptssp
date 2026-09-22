#include "argparse.hpp"

#include "pairs_tuple.hpp"
#include "profiler.hpp"
#include "ter_solver.cuh"
#include "zero_sum_swap.h"
#ifdef WITH_GPU
#include "ss_kernel.cuh"
#endif

#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <fstream>
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

#include <immintrin.h>

size_t max_pairs_per_chunk = 20000000;

constexpr size_t BYTES_PER_PAIR_HOST_ESTIMATE = 64;

size_t compute_default_max_pairs(size_t budget_bytes, double headroom_fraction = 0.6)
{
    const size_t usable_bytes = static_cast<size_t>(budget_bytes * headroom_fraction);
    size_t max_pairs = usable_bytes / BYTES_PER_PAIR_HOST_ESTIMATE;
    if (max_pairs < 1000)
        max_pairs = 1000;
    return max_pairs;
}

size_t detect_system_ram_bytes()
{
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open())
    {
        std::string label;
        size_t kb;
        std::string unit;
        while (meminfo >> label >> kb >> unit)
        {
            if (label == "MemTotal:")
                return kb * 1024ULL;
        }
    }
    constexpr size_t fallback_bytes = 8ULL * 1024 * 1024 * 1024;
    return fallback_bytes;
}

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

    // Bug 2 (rencana.md §4.3): checking only that each individual value
    // and the target fit in size_t is not enough. shroeppel_shamir_1d<uint64_t>
    // internally sums many values (P[k]/S[k] windows in compute_k_window /
    // compute_k_priority_profile, every subset-sum produced by
    // generate_subsets, pair scores placed in the heap, and the final
    // verification sum in print_and_write_1d_ss_solution). If the *total*
    // of all values exceeds 2^64, those sums wrap silently: score < target
    // comparisons can come out wrong (missing real solutions), and the
    // final "sum != target" check is itself only correct modulo 2^64 (so
    // it can also accept a false solution). Require the grand total,
    // computed here at full u128 precision, to also fit size_t before the
    // uint64_t path is used; otherwise callers must fall back to the
    // (slower, not yet GPU-accelerated) u128 path.
    u128 total = 0;
    for (u128 value : instance.values)
    {
        if (value > max_size_t)
            return false;
        total += value;
        if (total > max_size_t)
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
        return -1;
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
        std::cout << "0";
        return;
    }

    size_t msb = 0;
    for (size_t i = sizeof(size_t) * 8; i > 0; --i)
    {
        if (value & (1ULL << (i - 1)))
        {
            msb = i - 1;
            break;
        }
    }

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
    size_t total_subsets = 1ULL << n;

    printf("Generating %zu possible subset weights for set of size %zu.\n", total_subsets, n);
    std::vector<T> set_weights(total_subsets, 0);

    size_t generated = 1;
    for (size_t pass = 0; pass < n; ++pass)
    {
        const T weight = weights[pass];

        if constexpr (std::is_same_v<T, uint64_t>)
        {
            const __m256i wvec = _mm256_set1_epi64x(static_cast<int64_t>(weight));
            size_t i = 0;

            for (; i + 4 <= generated; i += 4)
            {
                __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(set_weights.data() + i));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(set_weights.data() + generated + i),
                    _mm256_add_epi64(v, wvec));
            }

            for (; i < generated; ++i)
                set_weights[generated + i] = set_weights[i] + weight;
        }
        else
        {
            
            if (generated >= 4096)
            {
#pragma omp parallel for
                for (size_t i = 0; i < generated; ++i)
                    set_weights[generated + i] = set_weights[i] + weight;
            }
            else
            {
                
                size_t i = 0;
                for (; i + 4 <= generated; i += 4)
                {
                    set_weights[generated + i]     = set_weights[i]     + weight;
                    set_weights[generated + i + 1] = set_weights[i + 1] + weight;
                    set_weights[generated + i + 2] = set_weights[i + 2] + weight;
                    set_weights[generated + i + 3] = set_weights[i + 3] + weight;
                }
                for (; i < generated; ++i)
                    set_weights[generated + i] = set_weights[i] + weight;
            }
        }

        generated <<= 1;
    }

    return set_weights;
}

template <typename T>
std::vector<size_t> sort_indices(const std::vector<T> &arr, bool ascending)
{
    size_t n = arr.size();

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

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

template <typename T>
void print_info_line(
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
        std::cout << std::setw(5) << i_iter << " "
                  << std::setw(8) << std::fixed << std::setprecision(2) << time << "s: "
                  << number_to_string(score1) << " + " << number_to_string(score2) << "; "
                  << n_q1 << " x " << n_q2 << " possible solutions\n";
    }
}

template <typename T>
struct HeapNode
{
    T score;
    size_t first;
    size_t second;
};

template <bool ascending, typename T, typename F>
void extract_pairs_from_heap(std::vector<PairsTuple> &pairs_same_score, std::vector<HeapNode<T>> &heap, T score_pair, std::vector<size_t> &chunks_beg, std::vector<size_t> &chunks_n_pairs, size_t &n_pairs_total, const std::vector<T> &first_weights, const std::vector<T> &second_weights, const std::vector<T> &second_weights_sorted_asc, F &&cmp)
{
    
    size_t n_pairs_chunk = 0;

    while (!heap.empty() && heap.front().score == score_pair)
    {
        const auto pair1_same_score = heap.front();
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.pop_back();

        const size_t pos_second_weights_beg = pair1_same_score.second;
        size_t pos_second_weights_end = pos_second_weights_beg;

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
            const T next_score = first_weights[pair1_same_score.first] + second_weights_sorted_asc[pos_second_weights_end];
            if (ascending)
                assert(score_pair < next_score);
            else
                assert(score_pair > next_score);

            heap.push_back({next_score, pair1_same_score.first, pos_second_weights_end});
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    chunks_n_pairs.push_back(n_pairs_chunk);
    chunks_beg.push_back(pairs_same_score.size());
}

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
    if (k_min == SIZE_MAX || !found || k_max < k_min) return {1, 0};
    return {k_min, k_max};
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

    // +1 slot scratch: kompaksi branchless di bawah selalu menulis fw[out]/fm[out],
    // termasuk untuk elemen tak valid setelah entri valid terakhir (out == n_valid).
    std::vector<T>      fw(n_valid + 1);
    std::vector<size_t> fm(n_valid + 1);

    size_t out = 0;
    for (size_t i = 0; i < n; ++i)
    {
        fw[out] = weights[i];
        fm[out] = i;
        out += valid_mask[i];
    }
    fw.resize(n_valid);
    fm.resize(n_valid);

    return {std::move(fw), std::move(fm)};
}

template <typename T>
bool print_and_write_1d_ss_solution(size_t index_list1, size_t index_list2, size_t index_list3, size_t index_list4,
                                    const std::vector<T> &list1, const std::vector<T> &list2,
                                    const std::vector<T> &list3, const std::vector<T> &list4,
                                    const std::string &instance_name, T target,
                                    std::vector<size_t> *solution_out = nullptr)
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

    if (solution_out)
        *solution_out = indices; // indeks asli (list1..list4 adalah potongan berurutan dari values)

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
                         int k_radius = -1, std::vector<size_t> *solution_out = nullptr, bool ss_gpu_sort = false)
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

    std::vector<T> set2_weights_sorted_asc, set4_weights_sorted_desc;
    std::vector<size_t> set2_subsets_sorted_asc, set4_subsets_sorted_desc;
    bool gpu_sorted2 = false, gpu_sorted4 = false;
#ifdef WITH_GPU
    if constexpr (std::is_same_v<T, uint64_t>) {
        if (ss_gpu_sort && set2_weights.size() > (1u << 16)) {
            set2_weights_sorted_asc = set2_weights;
            set2_subsets_sorted_asc = set2_subsets;
            gpu_sorted2 = gpu_sort_weights_with_payload(set2_weights_sorted_asc, set2_subsets_sorted_asc, true);
        }
        if (ss_gpu_sort && set4_weights.size() > (1u << 16)) {
            set4_weights_sorted_desc = set4_weights;
            set4_subsets_sorted_desc = set4_subsets;
            gpu_sorted4 = gpu_sort_weights_with_payload(set4_weights_sorted_desc, set4_subsets_sorted_desc, false);
        }
    }
#endif
    if (!gpu_sorted2) {
        auto asc_indices_set2_weights = sort_indices(set2_weights, true);
        set2_weights_sorted_asc = apply_permutation(set2_weights, asc_indices_set2_weights);
        set2_subsets_sorted_asc = apply_permutation(set2_subsets, asc_indices_set2_weights);
    }
    if (!gpu_sorted4) {
        auto desc_indices_set4_weights = sort_indices(set4_weights, false);
        set4_weights_sorted_desc = apply_permutation(set4_weights, desc_indices_set4_weights);
        set4_subsets_sorted_desc = apply_permutation(set4_subsets, desc_indices_set4_weights);
    }
    if (ss_gpu_sort && (set2_weights.size() > (1u << 16) || set4_weights.size() > (1u << 16)))
        std::cout << "[SS-GPU] sort quarter2=" << (gpu_sorted2 ? "GPU" : "CPU(fallback)")
                  << " quarter4=" << (gpu_sorted4 ? "GPU" : "CPU(fallback)") << "\n";

    if (set2_weights_sorted_asc.empty() || set4_weights_sorted_desc.empty())
        continue;

    auto min_cmp = [](const HeapNode<T> &a1, const HeapNode<T> &a2) noexcept -> bool
    {
        return a1.score > a2.score;
    };

    auto max_cmp = [](const HeapNode<T> &a1, const HeapNode<T> &a2) noexcept -> bool
    {
        return a1.score < a2.score;
    };

    std::vector<HeapNode<T>> heap1;
    heap1.reserve(set1_weights.size());
    std::vector<HeapNode<T>> heap2;
    heap2.reserve(set3_weights.size());

    for (size_t i = 0; i < set1_weights.size(); ++i)
    {
        const T initial_score = set1_weights[i] + set2_weights_sorted_asc[0];
        if (initial_score <= target)
            heap1.push_back({initial_score, i, 0});
    }

    for (size_t i = 0; i < set3_weights.size(); ++i)
    {
        const T initial_score = set3_weights[i] + set4_weights_sorted_desc[0];
        heap2.push_back({initial_score, i, 0});
    }

    std::make_heap(heap1.begin(), heap1.end(), min_cmp);
    std::make_heap(heap2.begin(), heap2.end(), max_cmp);

    std::cout << "Running the search loop\n\n";
    profiler = std::make_unique<ScopedProfiler>("List traversal              ");

    size_t i_iter_checking = 0;
    while (!heap1.empty() && !heap2.empty())
    {
        const T score_pair1 = heap1.front().score;
        const T score_pair2 = heap2.front().score;
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
            print_info_line(
                i_iter_checking, profilerTotal->elapsed(), score_pair1, score_pair2, n_pairs_q1, n_pairs_q2);

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
                    list1, list2, list3, list4, instance_name, target, solution_out);
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
            {
                size_t step = 1;
                while (pos_set2_weights + step + 1 < set2_weights.size() &&
                       (set2_weights_sorted_asc[pos_set2_weights + step] == set2_weights_sorted_asc[pair1.second] ||
                        (set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights + step] + score_pair2) < target))
                {
                    pos_set2_weights += step;
                    step <<= 1;
                }
                ++pos_set2_weights;
            }

            if (pos_set2_weights < set2_weights.size())
            {
                const T next_score = set1_weights[pair1.first] + set2_weights_sorted_asc[pos_set2_weights];
                if (next_score <= target)
                {
                    heap1.push_back({next_score, pair1.first, pos_set2_weights});
                    std::push_heap(heap1.begin(), heap1.end(), min_cmp);
                }
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
            {
                size_t step = 1;
                while (pos_set4_weights + step + 1 < set4_weights.size() &&
                       (set4_weights_sorted_desc[pos_set4_weights + step] == set4_weights_sorted_desc[pair2.second] ||
                        (score_pair1 + set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights + step]) > target))
                {
                    pos_set4_weights += step;
                    step <<= 1;
                }
                ++pos_set4_weights;
            }

            if (pos_set4_weights < set4_weights.size())
            {
                const T next_score = set3_weights[pair2.first] + set4_weights_sorted_desc[pos_set4_weights];
                if (next_score <= target)
                {
                    heap2.push_back({next_score, pair2.first, pos_set4_weights});
                    std::push_heap(heap2.begin(), heap2.end(), max_cmp);
                }
            }
        }
    }
    }

    return false;
}

std::string get_filename_without_extension(const std::string &filePath)
{
    
    size_t lastSlash = filePath.find_last_of("/\\");
    size_t start = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;

    size_t lastDot = filePath.find_last_of('.');
    size_t end = (lastDot == std::string::npos || lastDot < start) ? filePath.length() : lastDot;

    return filePath.substr(start, end - start);
}

// --extsol (rencana.md §9): setelah solver menemukan satu solusi, jelajahi solusi lain yang
// TERHUBUNG lewat swap nol-jumlah (zero_sum_swap.h). Hasil ditulis ke <instance>.extsol,
// satu bitstring per baris; baris pertama = solusi awal (sama dengan isi .sol).
static void run_extsol(const std::vector<u128> &values, u128 target, const std::vector<size_t> &initial,
                       const std::string &instance_name, int swap_size, size_t max_solutions)
{
    SwapExploreParams P;
    P.max_swap_size = swap_size;
    P.max_solutions = max_solutions; // 0 = tanpa batas
    P.verbose = true;

    ScopedProfiler prof("Extsol (zero-sum swap)     ");
    const SwapExploreResult R = explore_zero_sum_swaps(values, target, initial, P);

    if (!R.error.empty())
    {
        std::cout << "[extsol] GAGAL: " << R.error << "\n";
        return;
    }

    std::cout << "[extsol] selesai: " << R.all_solutions.size() << " solusi (termasuk solusi awal), "
              << R.tiers_run << " tier, berhenti karena: " << R.stop_reason << "\n";
    if (R.verify_failures > 0)
        std::cout << "[extsol] PERINGATAN: " << R.verify_failures
                  << " kandidat gagal verifikasi ulang (seharusnya 0)\n";
    if (R.capped)
        std::cout << "[extsol] hasil DIBATASI (capped): daftar di bawah belum tentu lengkap "
                     "bahkan untuk solusi yang terhubung.\n";
    else
        std::cout << "[extsol] closure: semua solusi yang terhubung dari solusi awal lewat swap <= "
                  << swap_size << " per sisi sudah ditemukan. Ini BUKAN bukti tidak ada solusi lain "
                     "di luar komponen tersebut.\n";

    const std::string ext_name = instance_name + ".extsol";
    std::ofstream ext_file(ext_name);
    if (!ext_file.is_open())
    {
        std::cout << "[extsol] tidak bisa menulis " << ext_name << "\n";
        return;
    }
    for (const SwapWitness &w : R.all_solutions)
    {
        std::string bits(values.size(), '0');
        for (size_t idx : w.indices)
            bits[idx] = '1';
        ext_file << bits << "\n";
    }
    std::cout << "[extsol] " << R.all_solutions.size() << " solusi ditulis ke " << ext_name << "\n";
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("markshare");

    std::string path = "";
    size_t n_threads = 0;
    bool check_only = false;
    bool write_prb = false;
    int k_radius = -1;
    int runs = 0;
    double timeout_sec = 0.0;
    std::string solver = "ter";
    double mem_budget_gb = 0.0;
    int extsol_swap_size = 4;
    size_t extsol_max_solutions = 1000;

    program.add_argument("-t", "--threads")
        .store_into(n_threads)
        .help("Number of CPU threads. Defaults to all available logical CPU cores.")
        .default_value(0);

    program.add_argument("--check_only")
        .help("Only parse and verify input metadata/planted witness; do not solve.")
        .flag();

    program.add_argument("--write_prb")
        .help("Write parsed 1D txt/prb input back as MarketShareGpu-style .prb.")
        .flag();

    program.add_argument("-f", "--file")
        .store_into(path)
        .help("Path to a 1D subset-sum instance (.txt in HGJ format, or .prb). Required.")
        .required();

    program.add_argument("--max_pairs")
        .store_into(max_pairs_per_chunk)
        .help("Maximum number of pairs to be evaluated on the GPU/CPU simultaneously. If it runs OOM, reduce this number. Default: dihitung otomatis dari --mem_budget_gb (lihat compute_default_max_pairs), BUKAN angka tetap.")
        .default_value(size_t(0));

    program.add_argument("--mem_budget_gb")
        .store_into(mem_budget_gb)
        .help("Budget memori (GB) yang boleh dipakai untuk buffer per-pair saat evaluasi. 0 = auto-detect dari RAM sistem (lihat /proc/meminfo).")
        .default_value(0.0);

    program.add_argument("--k_radius")
        .store_into(k_radius)
        .help("Radius eksplorasi k di sekitar peak_k. -1 = semua k (default), 0 = peak_k saja (tercepat), r = peak_k +/- r.")
        .default_value(-1);

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

    program.add_argument("--ter_bucket")
        .help("TER: pakai sidecar key + bucket lookup di merge_level1 (CPU dan GPU) menggantikan binary search. "
              "Hasil harus sama; default mati supaya bisa dibandingkan.")
        .flag();

    program.add_argument("--ter_stats")
        .help("TER: mode statistik. Tidak berhenti di solusi pertama; jalankan tepat --runs kali dan cetak "
              "success rate per run + profil per fase. Wajib dengan --runs N (N > 0), tidak bisa dengan --autorestart.")
        .flag();

    program.add_argument("--ss_gpu")
        .help("SS: radix-sort quarter2/quarter4 (>65536 elemen) di GPU via cub::DeviceRadixSort, "
              "bukan std::sort CPU. Logika heap k-way merge tetap di CPU. Jalur uint64_t saja.")
        .flag();

    program.add_argument("--extsol")
        .help("Setelah solusi ditemukan, jelajahi solusi lain yang terhubung lewat swap nol-jumlah "
              "(BFS bertingkat); hasil ditulis ke <instance>.extsol. Hanya menjelajahi komponen "
              "terhubung dari solusi awal, bukan semua solusi.")
        .flag();

    program.add_argument("--extsol_swap_size")
        .store_into(extsol_swap_size)
        .help("Ukuran maksimum subset per sisi swap untuk --extsol (1.." + std::to_string(kMaxSwapSize) + "). Default: 4.")
        .default_value(4);

    program.add_argument("--extsol_max_solutions")
        .store_into(extsol_max_solutions)
        .help("Batas jumlah solusi (termasuk solusi awal) untuk --extsol. 0 = tanpa batas. Default: 1000.")
        .default_value(size_t(1000));

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

    const bool extsol = (program["--extsol"] == true);
    if (extsol && (extsol_swap_size < 1 || extsol_swap_size > kMaxSwapSize))
    {
        std::cerr << "Error: --extsol_swap_size harus di antara 1 dan " << kMaxSwapSize << ".\n";
        std::exit(1);
    }

    check_only = (program["--check_only"] == true);
    write_prb = (program["--write_prb"] == true);

    if (!program.is_used("--max_pairs"))
    {
        const size_t budget_bytes = (mem_budget_gb > 0.0)
                                         ? static_cast<size_t>(mem_budget_gb * 1024.0 * 1024.0 * 1024.0)
                                         : detect_system_ram_bytes();
        max_pairs_per_chunk = compute_default_max_pairs(budget_bytes);
        std::cout << "max_pairs_per_chunk auto-calibrated: " << max_pairs_per_chunk
                  << " (from " << (budget_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB budget"
                  << (mem_budget_gb > 0.0 ? ", user-specified --mem_budget_gb)" : ", auto-detected RAM)") << "\n";
    }
    else if (max_pairs_per_chunk == 0)
    {
        std::cout << "Warning: --max_pairs 0 tidak valid, menggunakan fallback minimal 1000.\n";
        max_pairs_per_chunk = 1000;
    }

    const unsigned detected_threads = std::max(1u, std::thread::hardware_concurrency());
    const int threads_to_use = static_cast<int>(n_threads == 0 ? detected_threads : n_threads);
    omp_set_num_threads(threads_to_use);
    std::cout << "CPU threads: " << threads_to_use;
    if (n_threads == 0)
        std::cout << " (auto-detected)";
    std::cout << std::endl;

    // K1 (rencana.md §10): the old m x n MarkShareFeas pipeline is gone.
    // -f is required (enforced above by argparse) and must name a 1D
    // subset-sum instance; there is no more fallback to a matrix format.
    const std::string instance_name = get_filename_without_extension(path);
    printf("Reading instance from file %s; instance_name %s\n", path.c_str(), instance_name.c_str());

    SubsetSum1D128 instance_1d;
    if (!load_hgj_txt_instance(path, instance_1d) && !load_prb_1d128_instance(path, instance_1d))
    {
        std::cerr << "Error: '" << path << "' is not a recognized 1D subset-sum instance "
                     "(expected HGJ .txt format with a 'Target T' line, or a '1 n v1 v2 ... vn target' .prb file). "
                     "Matrix (m x n MarkShareFeas) input is no longer supported.\n";
        std::exit(1);
    }

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
    {
        ScopedProfiler::report();
        return 0;
    }

    bool found = false;
    std::vector<size_t> solution_indices; // solusi awal untuk --extsol (indeks asli 0-based)
    std::vector<uint64_t> vals64;
    vals64.reserve(instance_1d.values.size());
    for (const auto &v : instance_1d.values)
        vals64.push_back(static_cast<uint64_t>(v));
    const uint64_t target64 = static_cast<uint64_t>(instance_1d.target);

    if (solver == "ter")
    {
        std::cout << "[TER] Menggunakan TER Solver (Yang Li et al., 2025)\n";
        TerParams params = ter_default_params((int)instance_1d.values.size());
        params.fixed_runs = runs;
        params.timeout_seconds = timeout_sec;
        if (program["--autorestart"] == true)
        {
            params.max_restarts = 0;
            params.fixed_runs = 0;
        }
        else if (runs > 0)
        {
            params.fixed_runs = runs;
        }
        else
        {
            params.fixed_runs = 1;
        }

        params.use_bucket_lookup = (program["--ter_bucket"] == true);
        if (program["--ter_stats"] == true)
        {
            if (runs <= 0 || program["--autorestart"] == true)
            {
                std::cerr << "Error: --ter_stats butuh --runs N (N > 0) dan tidak bisa dipakai bersama --autorestart.\n";
                std::exit(1);
            }
            params.continue_after_found = true;
        }

        TerResult res = ter_solve(instance_1d.values, instance_1d.target, params);
        found = res.found;
        if (found)
        {
            solution_indices = res.solution_indices;
            printf("Found feasible solution by TER in %.3f s (%d runs)!\n", res.elapsed_seconds, res.runs_attempted);
            std::cout << "Solution indices (" << res.solution_indices.size() << "): ";
            for (size_t idx : res.solution_indices) std::cout << idx << " ";
            std::cout << "\n";

            std::string sol_name = instance_name + ".sol";
            std::string sol_str(instance_1d.values.size(), '0');
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
            // TER is heuristic/probabilistic, not exhaustive (rencana.md
            // §4.4): a run that doesn't find a solution is not proof of
            // infeasibility, so the message must not claim that.
            printf("TER: tidak ditemukan dalam %d run (%.3f s) -- ini BUKAN bukti infeasible. "
                   "Gunakan --autorestart untuk terus mencoba.\n", res.runs_attempted, res.elapsed_seconds);
        }
    }
    else if (can_fit_size_t(instance_1d))
    {
        std::cout << "[SS] Menggunakan Schroeppel-Shamir Solver (64-bit)\n";
        found = shroeppel_shamir_1d<uint64_t>(vals64, target64, instance_name, k_radius, &solution_indices, program["--ss_gpu"] == true);
    }
    else
    {
        std::cout << "[SS] Menggunakan Schroeppel-Shamir Solver (u128)\n";
        found = shroeppel_shamir_1d<u128>(instance_1d.values, instance_1d.target, instance_name, k_radius, &solution_indices);
    }

    if (solver != "ter")
    {
        // SS is exhaustive only when k_radius == -1 (rencana.md §4.4); a
        // restricted k_radius only searched a window around peak_k.
        if (found)
            printf("Found feasible solution!\n");
        else if (k_radius < 0)
            printf("Instance was infeasible (exhaustive Schroeppel-Shamir search, all k).\n");
        else
            printf("Instance was not found by Schroeppel-Shamir on the tried k range (--k_radius %d) "
                   "-- this does NOT prove infeasibility.\n", k_radius);
    }

    if (extsol)
    {
        if (found)
            run_extsol(instance_1d.values, instance_1d.target, solution_indices, instance_name,
                       extsol_swap_size, extsol_max_solutions);
        else
            std::cout << "[extsol] dilewati: tidak ada solusi awal.\n";
    }

    ScopedProfiler::report();

    return 0;
}
