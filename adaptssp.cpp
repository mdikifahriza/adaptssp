
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <random>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <limits>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using u64  = uint64_t;
using u32  = uint32_t;
using u128 = unsigned __int128;

// ============================================================================
// Util u128
// ============================================================================
inline std::string u128_to_string(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) { s.push_back((char)('0' + (int)(v % 10))); v /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}
inline std::ostream& operator<<(std::ostream& os, u128 v) { return os << u128_to_string(v); }
inline u128 parse_u128_decimal(const std::string& str) {
    u128 val = 0;
    for (char c : str) if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
    return val;
}
inline double get_current_peak_ram_mb() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
    return 0.0;
#endif
}

// ============================================================================
// Instance + Preprocessing
// ============================================================================
struct Instance {
    std::vector<u128> elements;
    u128 target = 0;
    u128 total_sum = 0;
    double density = 0.0;
    int k_min = -1, k_max = -1;

    static bool load(const std::string& file_or_str, const std::string& tgt_str, Instance& inst) {
        std::string raw;
        std::ifstream infile(file_or_str);
        if (infile.is_open()) {
            std::string line;
            while (std::getline(infile, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                size_t p = std::min({line.find('#'), line.find("//"), line.find(';')});
                if (p != std::string::npos) line = line.substr(0, p);
                raw += line + " ";
            }
        } else {
            raw = file_or_str;
        }
        inst.elements.clear();
        std::string cur;
        for (char c : raw) {
            if (isdigit((unsigned char)c)) cur += c;
            else if (!cur.empty()) { inst.elements.push_back(parse_u128_decimal(cur)); cur.clear(); }
        }
        if (!cur.empty()) inst.elements.push_back(parse_u128_decimal(cur));
        if (!tgt_str.empty()) inst.target = parse_u128_decimal(tgt_str);

        inst.total_sum = 0;
        u128 max_val = 0;
        for (u128 v : inst.elements) { inst.total_sum += v; if (v > max_val) max_val = v; }
        double max_bits = (max_val == 0) ? 1.0 : std::log2((double)max_val + 1.0);
        inst.density = inst.elements.empty() ? 0.0 : ((double)inst.elements.size() / max_bits);

        compute_k_window(inst);
        return !inst.elements.empty();
    }

    static void compute_k_window(Instance& inst) {
        int n = (int)inst.elements.size();
        std::vector<u128> sorted_desc = inst.elements;
        std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<u128>());
        std::vector<u128> suffix(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) suffix[i] = suffix[i + 1] + sorted_desc[i];

        u128 cum = 0;
        inst.k_min = -1; inst.k_max = -1;
        for (int k = 1; k <= n; ++k) {
            cum += sorted_desc[k - 1];
            if (cum >= inst.target && inst.k_min == -1) inst.k_min = k;
            u128 cum_low = suffix[n - k];
            if (cum_low <= inst.target) inst.k_max = k;
        }
        if (inst.k_min == -1) inst.k_min = 1;
        if (inst.k_max == -1) inst.k_max = n;
    }

    double k_window_ratio() const {
        int n = (int)elements.size();
        if (n <= 0) return 0.0;
        int width = k_max - k_min + 1;
        if (width < 0) width = 0;
        return (double)width / (double)n;
    }
};

// ============================================================================
// Data Structures
// ============================================================================
struct BaseListSoA {
    std::vector<u32>  rem;
    std::vector<u32>  mask;
    std::vector<u128> sum;
    void clear() { rem.clear(); mask.clear(); sum.clear(); }
    void reserve(size_t n) { rem.reserve(n); mask.reserve(n); sum.reserve(n); }
    size_t size() const { return rem.size(); }
    inline void push(u128 s, u32 m, u32 r) { sum.push_back(s); mask.push_back(m); rem.push_back(r); }
};

struct BaseEntry {
    u128 sum = 0;
    u32  mask = 0;
};

struct HalfEntry {
    u128 sum = 0;
    u64  mask_half = 0;
};

struct Witness {
    std::vector<int>  indices;
    std::vector<u128> values;
    u128 sum = 0;
};

inline bool verify_witness(const std::vector<u128>& raw, u128 target, int expected_k,
                            const Witness& w, std::string& msg) {
    if ((int)w.indices.size() != expected_k) {
        msg = "REJECTED: witness size " + std::to_string(w.indices.size()) +
              " != expected K=" + std::to_string(expected_k);
        return false;
    }
    std::vector<bool> seen(raw.size(), false);
    u128 s = 0;
    for (size_t i = 0; i < w.indices.size(); ++i) {
        int idx = w.indices[i];
        if (idx < 0 || idx >= (int)raw.size()) { msg = "REJECTED: index out of range"; return false; }
        if (seen[idx]) { msg = "REJECTED: duplicate index"; return false; }
        seen[idx] = true;
        if (raw[idx] != w.values[i]) { msg = "REJECTED: value mismatch at index " + std::to_string(idx); return false; }
        s += w.values[i];
    }
    if (s != target) {
        msg = "REJECTED: sum " + u128_to_string(s) + " != target " + u128_to_string(target);
        return false;
    }
    msg = "OK: verified independently, sum == target, indices unique.";
    return true;
}

struct ProgressEstimate {
    double elapsed_s = 0.0;
    u64    partitions = 0;
    u64    queries = 0;
    u64    false_positives = 0;
    double partition_rate = 0.0;
    double fp_rate_per_partition = 0.0;
    double min_extra_partitions_95 = 0.0;
    double min_extra_seconds_95 = 0.0;

    static ProgressEstimate compute(double elapsed_s_, u64 partitions_, u64 queries_, u64 fp_) {
        ProgressEstimate e;
        e.elapsed_s = elapsed_s_;
        e.partitions = partitions_;
        e.queries = queries_;
        e.false_positives = fp_;
        if (elapsed_s_ > 0.0) e.partition_rate = (double)partitions_ / elapsed_s_;
        if (partitions_ > 0) {
            e.fp_rate_per_partition = (double)fp_ / (double)partitions_;
            e.min_extra_partitions_95 = (double)partitions_ / 3.0;
            if (e.partition_rate > 0.0)
                e.min_extra_seconds_95 = e.min_extra_partitions_95 / e.partition_rate;
        }
        return e;
    }

    void print(std::ostream& os, double k_window_ratio) const {
        os << "[PROGRESS] t=" << std::fixed << std::setprecision(1) << elapsed_s
           << "s  partitions=" << partitions << "  queries=" << queries
           << "  rate=" << std::setprecision(2) << partition_rate << " part/s"
           << "  k_window_ratio=" << std::setprecision(3) << k_window_ratio
           << "  false_positives=" << false_positives
           << " (" << std::setprecision(3) << (fp_rate_per_partition * 100.0) << "%/partisi)";
        if (partitions > 0) {
            os << "  | min. partisi tambahan (95% CI)~=" << std::setprecision(0)
               << min_extra_partitions_95;
            if (min_extra_seconds_95 > 0.0)
                os << " (~" << std::setprecision(1) << min_extra_seconds_95 << "s)";
        }
        os << "\n";
    }
};

struct Report {
    bool   solved = false;
    double runtime_ms = 0.0;
    double peak_ram_mb = 0.0;
    u64    partitions_evaluated = 0;
    u64    modular_queries_evaluated = 0;
    u64    base_combinations_generated = 0;
    u64    half_candidates_generated = 0;
    u64    false_positives_rejected = 0;
    u32    threads_used = 1;
    Witness witness;
    std::string verification_msg;
};

// ============================================================================
// Adaptive Solver
// ============================================================================
class Solver {
public:
    static Report solve(const Instance& inst_in, unsigned num_threads, double time_limit_s,
                         bool proof_mode, bool verbose) {
        int n = (int)inst_in.elements.size();
        if (n <= 40) {
            return solve_small(inst_in, num_threads, time_limit_s, proof_mode, verbose);
        } else {
            return solve_large(inst_in, num_threads, time_limit_s, proof_mode, verbose);
        }
    }

private:
    // ------------------------------------------------------------------------
    // Helper Sum Extremes
    // ------------------------------------------------------------------------
    static u128 sum_smallest_k(std::vector<u128> v, int k) {
        if (k <= 0) return 0;
        std::nth_element(v.begin(), v.begin() + k, v.end());
        u128 s = 0; for (int i = 0; i < k; ++i) s += v[i];
        return s;
    }
    static u128 sum_largest_k(std::vector<u128> v, int k) {
        if (k <= 0) return 0;
        std::nth_element(v.begin(), v.end() - k, v.end());
        u128 s = 0; for (size_t i = v.size() - k; i < v.size(); ++i) s += v[i];
        return s;
    }

    // ------------------------------------------------------------------------
    // Engine 1: N <= 40 (Two-Pointer Sort-by-Sum, Tanpa Loop Modulus M)
    // ------------------------------------------------------------------------
    static void gen_recursive_small(int pos, int remaining_k, u128 cur_sum, u32 cur_mask,
                                     const std::vector<u128>& A, int q_n,
                                     std::vector<BaseEntry>& out) {
        if (remaining_k == 0) { out.push_back({cur_sum, cur_mask}); return; }
        if (pos >= q_n || (q_n - pos) < remaining_k) return;
        gen_recursive_small(pos + 1, remaining_k, cur_sum, cur_mask, A, q_n, out);
        gen_recursive_small(pos + 1, remaining_k - 1, cur_sum + A[pos],
                            cur_mask | (1U << pos), A, q_n, out);
    }
    static void gen_combinations_small(const std::vector<u128>& A, int q_n, int q_k,
                                        std::vector<BaseEntry>& out) {
        gen_recursive_small(0, q_k, (u128)0, (u32)0, A, q_n, out);
    }

    static void merge_quarters_small(const std::vector<BaseEntry>& list1,
                                     const std::vector<BaseEntry>& list2,
                                     u128 lower_bound, u128 upper_bound,
                                     std::vector<HalfEntry>& out_cand, int q_n,
                                     const std::atomic<bool>& solution_found) {
        size_t N1 = list1.size(), N2 = list2.size();
        if (N1 == 0 || N2 == 0) return;

        int64_t j_high = (int64_t)N2 - 1;
        int64_t j_low = (int64_t)N2;

        for (size_t i = 0; i < N1; ++i) {
            if (solution_found.load(std::memory_order_relaxed)) return;
            if (upper_bound < list1[i].sum) break;

            u128 min_val = (lower_bound > list1[i].sum) ? (lower_bound - list1[i].sum) : 0;
            u128 max_val = upper_bound - list1[i].sum;

            while (j_high >= 0 && list2[j_high].sum > max_val) j_high--;
            while (j_low > 0 && list2[j_low - 1].sum >= min_val) j_low--;

            int64_t start_j = std::max((int64_t)0, j_low);
            for (int64_t j = start_j; j <= j_high; ++j) {
                u128 half_sum = list1[i].sum + list2[j].sum;
                out_cand.push_back({half_sum, ((u64)list2[j].mask << q_n) | (u64)list1[i].mask});
            }
        }
    }

    static Report solve_small(const Instance& inst_in, unsigned num_threads, double time_limit_s,
                              bool proof_mode, bool verbose) {
        Report report;
        auto start_time = std::chrono::steady_clock::now();
        Instance inst = inst_in;
        int n = (int)inst.elements.size();
        int k = n / 2;
        int q_n = n / 4, q_k = k / 4;

        size_t est_combinations = 1;
        for (int i = 0; i < q_k; ++i) est_combinations = est_combinations * (q_n - i) / (i + 1);

        if (num_threads == 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
        report.threads_used = num_threads;

        if (verbose) {
            std::cout << "[ENGINE] Mode: TWO-POINTER SORT-BY-SUM (Optimized for N <= 40)\n";
            std::cout << "[INFO] n=" << n << " k=" << k << " q_n=" << q_n << " q_k=" << q_k << "\n";
            std::cout << "[INFO] C(q_n,q_k)=" << est_combinations << "  threads=" << num_threads << "\n";
        }

        std::atomic<bool> solution_found(false);
        std::atomic<u64>  total_partitions(0);
        std::atomic<u64>  total_queries(0);
        std::atomic<u64>  total_base_entries(0);
        std::atomic<u64>  total_half_candidates(0);
        std::atomic<u64>  total_false_positives(0);
        std::mutex report_mutex;
        std::condition_variable cv_done;
        Witness best_witness;

        auto worker = [&](unsigned thread_id) {
            std::mt19937_64 rng(1337ULL + thread_id * 10007ULL +
                (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;

            std::vector<BaseEntry> L1, L2, R1, R2;
            L1.reserve(est_combinations); L2.reserve(est_combinations);
            R1.reserve(est_combinations); R2.reserve(est_combinations);

            std::vector<HalfEntry> cand_L, cand_R;
            size_t est_cand = est_combinations * est_combinations;
            cand_L.reserve(est_cand); cand_R.reserve(est_cand);

            std::vector<u128> Q1(q_n), Q2(q_n), Q3(q_n), Q4(q_n);
            std::vector<u128> Left_elem(2 * q_n), Right_elem(2 * q_n);

            u64 local_partitions = 0, local_queries = 0, local_base_entries = 0;
            auto flush_counters = [&]() {
                if (local_partitions) total_partitions += local_partitions;
                if (local_queries) total_queries += local_queries;
                if (local_base_entries) total_base_entries += local_base_entries;
                local_partitions = local_queries = local_base_entries = 0;
            };

            while (!solution_found.load(std::memory_order_relaxed)) {
                if (time_limit_s > 0.0) {
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (el >= time_limit_s) break;
                }

                local_partitions++;
                std::shuffle(perm.begin(), perm.end(), rng);

                for (int i = 0; i < q_n; ++i) {
                    Q1[i] = inst.elements[perm[i]];
                    Q2[i] = inst.elements[perm[q_n + i]];
                    Q3[i] = inst.elements[perm[2 * q_n + i]];
                    Q4[i] = inst.elements[perm[3 * q_n + i]];
                    Left_elem[i] = Q1[i]; Left_elem[q_n + i] = Q2[i];
                    Right_elem[i] = Q3[i]; Right_elem[q_n + i] = Q4[i];
                }

                int half_k = 2 * q_k;
                u128 min_L = sum_smallest_k(Left_elem, half_k);
                u128 max_L = sum_largest_k(Left_elem, half_k);
                u128 min_R = sum_smallest_k(Right_elem, half_k);
                u128 max_R = sum_largest_k(Right_elem, half_k);

                u128 lower_bound_L = (inst.target > max_R) ? (inst.target - max_R) : 0;
                u128 upper_bound_L = (inst.target >= min_R) ? (inst.target - min_R) : 0;
                u128 lower_bound_R = (inst.target > max_L) ? (inst.target - max_L) : 0;
                u128 upper_bound_R = (inst.target >= min_L) ? (inst.target - min_L) : 0;

                L1.clear(); L2.clear(); R1.clear(); R2.clear();
                gen_combinations_small(Q1, q_n, q_k, L1);
                gen_combinations_small(Q2, q_n, q_k, L2);
                gen_combinations_small(Q3, q_n, q_k, R1);
                gen_combinations_small(Q4, q_n, q_k, R2);
                local_base_entries += (u64)(L1.size() + L2.size() + R1.size() + R2.size());

                std::sort(L1.begin(), L1.end(), [](const BaseEntry& a, const BaseEntry& b) { return a.sum < b.sum; });
                std::sort(L2.begin(), L2.end(), [](const BaseEntry& a, const BaseEntry& b) { return a.sum < b.sum; });
                std::sort(R1.begin(), R1.end(), [](const BaseEntry& a, const BaseEntry& b) { return a.sum < b.sum; });
                std::sort(R2.begin(), R2.end(), [](const BaseEntry& a, const BaseEntry& b) { return a.sum < b.sum; });

                local_queries++;

                cand_L.clear();
                merge_quarters_small(L1, L2, lower_bound_L, upper_bound_L, cand_L, q_n, solution_found);
                total_half_candidates.fetch_add((u64)cand_L.size(), std::memory_order_relaxed);
                if (solution_found.load(std::memory_order_relaxed) || cand_L.empty()) {
                    flush_counters();
                    continue;
                }

                cand_R.clear();
                merge_quarters_small(R1, R2, lower_bound_R, upper_bound_R, cand_R, q_n, solution_found);
                total_half_candidates.fetch_add((u64)cand_R.size(), std::memory_order_relaxed);
                if (solution_found.load(std::memory_order_relaxed) || cand_R.empty()) {
                    flush_counters();
                    continue;
                }

                std::sort(cand_R.begin(), cand_R.end(),
                    [](const HalfEntry& a, const HalfEntry& b) { return a.sum < b.sum; });

                for (const auto& cL : cand_L) {
                    if (solution_found.load(std::memory_order_relaxed)) break;
                    if (cL.sum > inst.target) continue;
                    u128 req_sum = inst.target - cL.sum;
                    if (req_sum < cand_R.front().sum || req_sum > cand_R.back().sum) continue;

                    HalfEntry dummy{req_sum, 0};
                    auto range = std::equal_range(cand_R.begin(), cand_R.end(), dummy,
                        [](const HalfEntry& a, const HalfEntry& b) { return a.sum < b.sum; });

                    for (auto it = range.first; it != range.second; ++it) {
                        const auto& cR = *it;
                        if (cL.sum + cR.sum != inst.target) continue;

                        Witness cand;
                        for (int b = 0; b < q_n; ++b) {
                            if ((cL.mask_half >> b) & 1ULL) {
                                int idx = perm[b];
                                cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                            }
                            if ((cL.mask_half >> (q_n + b)) & 1ULL) {
                                int idx = perm[q_n + b];
                                cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                            }
                        }
                        for (int b = 0; b < q_n; ++b) {
                            if ((cR.mask_half >> b) & 1ULL) {
                                int idx = perm[2 * q_n + b];
                                cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                            }
                            if ((cR.mask_half >> (q_n + b)) & 1ULL) {
                                int idx = perm[3 * q_n + b];
                                cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                            }
                        }

                        std::vector<std::pair<int, u128>> pairs;
                        for (size_t pi = 0; pi < cand.indices.size(); ++pi)
                            pairs.push_back({cand.indices[pi], cand.values[pi]});
                        std::sort(pairs.begin(), pairs.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
                        cand.indices.clear(); cand.values.clear();
                        cand.sum = 0;
                        for (auto& p : pairs) {
                            cand.indices.push_back(p.first);
                            cand.values.push_back(p.second);
                            cand.sum += p.second;
                        }

                        std::string msg;
                        bool ok = verify_witness(inst.elements, inst.target, k, cand, msg);
                        if (!ok) {
                            total_false_positives.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }

                        std::unique_lock<std::mutex> lock(report_mutex);
                        if (!solution_found.load(std::memory_order_relaxed)) {
                            solution_found = true;
                            best_witness = cand;
                            report.verification_msg = msg;
                            cv_done.notify_all();
                        }
                        flush_counters();
                        return;
                    }
                }
                flush_counters();
            }
            flush_counters();
        };

        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) workers.emplace_back(worker, t);

        std::thread logger;
        std::atomic<bool> stop_logger(false);
        if (verbose) {
            logger = std::thread([&]() {
                double window_ratio = inst.k_window_ratio();
                std::unique_lock<std::mutex> lk(report_mutex);
                while (!solution_found.load() && !stop_logger.load()) {
                    cv_done.wait_for(lk, std::chrono::seconds(10), [&]() {
                        return solution_found.load() || stop_logger.load();
                    });
                    if (solution_found.load() || stop_logger.load()) break;
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    ProgressEstimate est = ProgressEstimate::compute(
                        el, total_partitions.load(), total_queries.load(), total_false_positives.load());
                    est.print(std::cout, window_ratio);
                }
            });
        }

        for (auto& th : workers) if (th.joinable()) th.join();
        {
            std::unique_lock<std::mutex> lock(report_mutex);
            stop_logger = true;
            cv_done.notify_all();
        }
        if (logger.joinable()) logger.join();

        auto end_time = std::chrono::steady_clock::now();
        report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        report.peak_ram_mb = get_current_peak_ram_mb();
        report.partitions_evaluated = total_partitions.load();
        report.modular_queries_evaluated = total_queries.load();
        report.base_combinations_generated = total_base_entries.load();
        report.half_candidates_generated = total_half_candidates.load();
        report.false_positives_rejected = total_false_positives.load();
        report.solved = solution_found.load();
        if (report.solved) report.witness = best_witness;
        else if (report.verification_msg.empty()) report.verification_msg = "NOT FOUND / TIMEOUT.";

        return report;
    }

    // ------------------------------------------------------------------------
    // Engine 2: N >= 48 (HGJ Modulo M Representation - Anti-OOM)
    // ------------------------------------------------------------------------
    static void gen_recursive_large(int pos, int remaining_k, u128 cur_sum, u32 cur_rem, u32 cur_mask,
                                     const std::vector<u128>& A, const std::vector<u32>& A_rem,
                                     int q_n, u32 M, BaseListSoA& out) {
        if (remaining_k == 0) { out.push(cur_sum, cur_mask, cur_rem); return; }
        if (pos >= q_n || (q_n - pos) < remaining_k) return;
        gen_recursive_large(pos + 1, remaining_k, cur_sum, cur_rem, cur_mask, A, A_rem, q_n, M, out);
        u32 new_rem = cur_rem + A_rem[pos];
        if (new_rem >= M) new_rem -= M;
        gen_recursive_large(pos + 1, remaining_k - 1, cur_sum + A[pos], new_rem,
                            cur_mask | (1U << pos), A, A_rem, q_n, M, out);
    }
    static void gen_combinations_large(const std::vector<u128>& A, const std::vector<u32>& A_rem,
                                        int q_n, int q_k, u32 M, BaseListSoA& out) {
        gen_recursive_large(0, q_k, (u128)0, (u32)0, (u32)0, A, A_rem, q_n, M, out);
    }

    static void radix_sort_by_rem(BaseListSoA& list, std::vector<u32>& idx, std::vector<u32>& idx_tmp,
                                   std::vector<u32>& buf_rem, std::vector<u32>& buf_mask, std::vector<u128>& buf_sum,
                                   int total_bits) {
        size_t n = list.size();
        if (n < 2) return;
        idx.resize(n);
        for (size_t i = 0; i < n; ++i) idx[i] = (u32)i;
        idx_tmp.resize(n);

        const int RADIX_BITS = 13;
        const u32 RADIX_SIZE = 1U << RADIX_BITS;
        std::vector<u32> count(RADIX_SIZE + 1);
        int passes = (total_bits + RADIX_BITS - 1) / RADIX_BITS;
        if (passes < 1) passes = 1;

        std::vector<u32>* src = &idx;
        std::vector<u32>* dst = &idx_tmp;
        for (int p = 0; p < passes; ++p) {
            std::fill(count.begin(), count.end(), 0u);
            int shift = p * RADIX_BITS;
            for (u32 id : *src) count[((list.rem[id] >> shift) & (RADIX_SIZE - 1)) + 1]++;
            for (u32 i = 0; i < RADIX_SIZE; ++i) count[i + 1] += count[i];
            for (u32 id : *src) {
                u32 digit = (u32)((list.rem[id] >> shift) & (RADIX_SIZE - 1));
                (*dst)[count[digit]++] = id;
            }
            std::swap(src, dst);
        }

        buf_rem.resize(n); buf_mask.resize(n); buf_sum.resize(n);
        for (size_t i = 0; i < n; ++i) {
            u32 id = (*src)[i];
            buf_rem[i] = list.rem[id]; buf_mask[i] = list.mask[id]; buf_sum[i] = list.sum[id];
        }
        list.rem = buf_rem; list.mask = buf_mask; list.sum = buf_sum;
    }

    static void merge_quarters_sweep_large(const BaseListSoA& list1, const BaseListSoA& list2,
                                           u32 target_mod, u32 M, u128 lower_bound, u128 upper_bound,
                                           std::vector<HalfEntry>& out_cand, int q_n,
                                           const std::atomic<bool>& solution_found) {
        size_t N1 = list1.size(), N2 = list2.size();
        if (N1 == 0 || N2 == 0) return;
        const auto& rem1 = list1.rem; const auto& sum1 = list1.sum; const auto& mask1 = list1.mask;
        const auto& rem2 = list2.rem; const auto& sum2 = list2.sum; const auto& mask2 = list2.mask;

        size_t i = 0; int64_t j = (int64_t)N2 - 1;
        while (j >= 0 && rem2[j] > target_mod) j--;
        while (i < N1 && rem1[i] <= target_mod && j >= 0) {
            if (solution_found.load(std::memory_order_relaxed)) return;
            u32 s = rem1[i] + rem2[j];
            if (s < target_mod) i++;
            else if (s > target_mod) j--;
            else {
                size_t i2 = i; while (i2 < N1 && rem1[i2] == rem1[i]) i2++;
                int64_t j2 = j; while (j2 >= 0 && rem2[j2] == rem2[j]) j2--;
                for (size_t a = i; a < i2; ++a)
                    for (int64_t b = j; b > j2; --b) {
                        u128 half_sum = sum1[a] + sum2[b];
                        if (half_sum < lower_bound || half_sum > upper_bound) continue;
                        out_cand.push_back({half_sum, ((u64)mask2[b] << q_n) | (u64)mask1[a]});
                    }
                i = i2; j = j2;
            }
        }
        u32 target2 = target_mod + M;
        i = 0; while (i < N1 && rem1[i] <= target_mod) i++;
        j = (int64_t)N2 - 1;
        while (i < N1 && j >= 0 && rem2[j] > target_mod) {
            if (solution_found.load(std::memory_order_relaxed)) return;
            u32 s = rem1[i] + rem2[j];
            if (s < target2) i++;
            else if (s > target2) j--;
            else {
                size_t i2 = i; while (i2 < N1 && rem1[i2] == rem1[i]) i2++;
                int64_t j2 = j; while (j2 >= 0 && rem2[j2] == rem2[j]) j2--;
                for (size_t a = i; a < i2; ++a)
                    for (int64_t b = j; b > j2; --b) {
                        u128 half_sum = sum1[a] + sum2[b];
                        if (half_sum < lower_bound || half_sum > upper_bound) continue;
                        out_cand.push_back({half_sum, ((u64)mask2[b] << q_n) | (u64)mask1[a]});
                    }
                i = i2; j = j2;
            }
        }
    }

    static Report solve_large(const Instance& inst_in, unsigned num_threads, double time_limit_s,
                              bool proof_mode, bool verbose) {
        Report report;
        auto start_time = std::chrono::steady_clock::now();
        Instance inst = inst_in;
        int n = (int)inst.elements.size();
        int k = n / 2;
        int q_n = n / 4, q_k = k / 4;

        size_t est_combinations = 1;
        for (int i = 0; i < q_k; ++i) est_combinations = est_combinations * (q_n - i) / (i + 1);

        if (num_threads == 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
        report.threads_used = num_threads;

        u32 M;
        if (est_combinations <= 15) M = 31;
        else if (est_combinations <= 50) M = 61;
        else if (est_combinations <= 150) M = 127;
        else if (est_combinations <= 500) M = 257;
        else if (est_combinations <= 2000) M = 1009;
        else if (est_combinations <= 6000) M = 4093;
        else if (est_combinations <= 20000) M = 16381;
        else if (est_combinations <= 80000) M = 65537;
        else if (est_combinations <= 300000) M = 262139;
        else if (est_combinations <= 1000000) M = 1048573;
        else M = 35989843U;

        int radix_bits_total = 1;
        while ((1ULL << radix_bits_total) < M) radix_bits_total++;

        if (verbose) {
            std::cout << "[ENGINE] Mode: HGJ MODULO M (Anti-OOM, Optimized for N >= 48)\n";
            std::cout << "[INFO] n=" << n << " k=" << k << " q_n=" << q_n << " q_k=" << q_k << "\n";
            std::cout << "[INFO] C(q_n,q_k)=" << est_combinations << "  M=" << M
                      << "  threads=" << num_threads << "\n";
        }

        std::atomic<bool> solution_found(false);
        std::atomic<u64>  total_partitions(0);
        std::atomic<u64>  total_queries(0);
        std::atomic<u64>  total_base_entries(0);
        std::atomic<u64>  total_half_candidates(0);
        std::atomic<u64>  total_false_positives(0);
        std::mutex report_mutex;
        std::condition_variable cv_done;
        Witness best_witness;

        auto worker = [&](unsigned thread_id) {
            std::mt19937_64 rng(1337ULL + thread_id * 10007ULL +
                (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;

            BaseListSoA L1, L2, R1, R2;
            L1.reserve(est_combinations); L2.reserve(est_combinations);
            R1.reserve(est_combinations); R2.reserve(est_combinations);

            std::vector<u32> idx_buf, idx_tmp;
            idx_buf.reserve(est_combinations); idx_tmp.reserve(est_combinations);

            std::vector<u32> buf_rem; std::vector<u32> buf_mask; std::vector<u128> buf_sum;
            buf_rem.reserve(est_combinations); buf_mask.reserve(est_combinations); buf_sum.reserve(est_combinations);

            std::vector<HalfEntry> cand_L, cand_R;
            size_t est_cand = std::min((size_t)262144, std::max((size_t)4096, (est_combinations * est_combinations) / (size_t)M * 2));
            cand_L.reserve(est_cand); cand_R.reserve(est_cand);

            std::vector<u128> Q1(q_n), Q2(q_n), Q3(q_n), Q4(q_n);
            std::vector<u32>  Q1r(q_n), Q2r(q_n), Q3r(q_n), Q4r(q_n);
            std::vector<u128> Left_elem(2 * q_n), Right_elem(2 * q_n);

            u64 local_partitions = 0, local_queries = 0, local_base_entries = 0;
            auto flush_counters = [&]() {
                if (local_partitions) total_partitions += local_partitions;
                if (local_queries) total_queries += local_queries;
                if (local_base_entries) total_base_entries += local_base_entries;
                local_partitions = local_queries = local_base_entries = 0;
            };

            while (!solution_found.load(std::memory_order_relaxed)) {
                if (time_limit_s > 0.0) {
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (el >= time_limit_s) break;
                }

                local_partitions++;
                std::shuffle(perm.begin(), perm.end(), rng);

                for (int i = 0; i < q_n; ++i) {
                    Q1[i] = inst.elements[perm[i]];
                    Q2[i] = inst.elements[perm[q_n + i]];
                    Q3[i] = inst.elements[perm[2 * q_n + i]];
                    Q4[i] = inst.elements[perm[3 * q_n + i]];
                    Left_elem[i] = Q1[i]; Left_elem[q_n + i] = Q2[i];
                    Right_elem[i] = Q3[i]; Right_elem[q_n + i] = Q4[i];
                }

                int half_k = 2 * q_k;
                u128 min_L = sum_smallest_k(Left_elem, half_k);
                u128 max_L = sum_largest_k(Left_elem, half_k);
                u128 min_R = sum_smallest_k(Right_elem, half_k);
                u128 max_R = sum_largest_k(Right_elem, half_k);

                u128 lower_bound_L = (inst.target > max_R) ? (inst.target - max_R) : 0;
                u128 upper_bound_L = (inst.target >= min_R) ? (inst.target - min_R) : 0;
                u128 lower_bound_R = (inst.target > max_L) ? (inst.target - max_L) : 0;
                u128 upper_bound_R = (inst.target >= min_L) ? (inst.target - min_L) : 0;

                L1.clear(); L2.clear(); R1.clear(); R2.clear();
                for (int i = 0; i < q_n; ++i) {
                    Q1r[i] = (u32)(Q1[i] % (u128)M);
                    Q2r[i] = (u32)(Q2[i] % (u128)M);
                    Q3r[i] = (u32)(Q3[i] % (u128)M);
                    Q4r[i] = (u32)(Q4[i] % (u128)M);
                }
                gen_combinations_large(Q1, Q1r, q_n, q_k, M, L1);
                gen_combinations_large(Q2, Q2r, q_n, q_k, M, L2);
                gen_combinations_large(Q3, Q3r, q_n, q_k, M, R1);
                gen_combinations_large(Q4, Q4r, q_n, q_k, M, R2);
                local_base_entries += (u64)(L1.size() + L2.size() + R1.size() + R2.size());

                radix_sort_by_rem(L1, idx_buf, idx_tmp, buf_rem, buf_mask, buf_sum, radix_bits_total);
                radix_sort_by_rem(L2, idx_buf, idx_tmp, buf_rem, buf_mask, buf_sum, radix_bits_total);
                radix_sort_by_rem(R1, idx_buf, idx_tmp, buf_rem, buf_mask, buf_sum, radix_bits_total);
                radix_sort_by_rem(R2, idx_buf, idx_tmp, buf_rem, buf_mask, buf_sum, radix_bits_total);

                u32 target_mod = (u32)(inst.target % (u128)M);

                for (u32 R_mod = 0; R_mod < M && !solution_found; ++R_mod) {
                    local_queries++;
                    u32 R_comp_mod = (target_mod >= R_mod)
                        ? (target_mod - R_mod)
                        : (M - (R_mod - target_mod));

                    cand_L.clear();
                    merge_quarters_sweep_large(L1, L2, R_mod, M, lower_bound_L, upper_bound_L, cand_L, q_n, solution_found);
                    total_half_candidates.fetch_add((u64)cand_L.size(), std::memory_order_relaxed);
                    if (solution_found.load(std::memory_order_relaxed) || cand_L.empty()) continue;

                    cand_R.clear();
                    merge_quarters_sweep_large(R1, R2, R_comp_mod, M, lower_bound_R, upper_bound_R, cand_R, q_n, solution_found);
                    total_half_candidates.fetch_add((u64)cand_R.size(), std::memory_order_relaxed);
                    if (solution_found.load(std::memory_order_relaxed) || cand_R.empty()) continue;

                    std::sort(cand_R.begin(), cand_R.end(),
                        [](const HalfEntry& a, const HalfEntry& b) { return a.sum < b.sum; });

                    for (const auto& cL : cand_L) {
                        if (solution_found.load(std::memory_order_relaxed)) break;
                        if (cL.sum > inst.target) continue;
                        u128 req_sum = inst.target - cL.sum;
                        if (req_sum < cand_R.front().sum || req_sum > cand_R.back().sum) continue;

                        HalfEntry dummy{req_sum, 0};
                        auto range = std::equal_range(cand_R.begin(), cand_R.end(), dummy,
                            [](const HalfEntry& a, const HalfEntry& b) { return a.sum < b.sum; });

                        for (auto it = range.first; it != range.second; ++it) {
                            const auto& cR = *it;
                            if (cL.sum + cR.sum != inst.target) continue;

                            Witness cand;
                            for (int b = 0; b < q_n; ++b) {
                                if ((cL.mask_half >> b) & 1ULL) {
                                    int idx = perm[b];
                                    cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                                }
                                if ((cL.mask_half >> (q_n + b)) & 1ULL) {
                                    int idx = perm[q_n + b];
                                    cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                                }
                            }
                            for (int b = 0; b < q_n; ++b) {
                                if ((cR.mask_half >> b) & 1ULL) {
                                    int idx = perm[2 * q_n + b];
                                    cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                                }
                                if ((cR.mask_half >> (q_n + b)) & 1ULL) {
                                    int idx = perm[3 * q_n + b];
                                    cand.indices.push_back(idx); cand.values.push_back(inst.elements[idx]);
                                }
                            }

                            std::vector<std::pair<int, u128>> pairs;
                            for (size_t pi = 0; pi < cand.indices.size(); ++pi)
                                pairs.push_back({cand.indices[pi], cand.values[pi]});
                            std::sort(pairs.begin(), pairs.end(),
                                [](const auto& a, const auto& b) { return a.first < b.first; });
                            cand.indices.clear(); cand.values.clear();
                            cand.sum = 0;
                            for (auto& p : pairs) {
                                cand.indices.push_back(p.first);
                                cand.values.push_back(p.second);
                                cand.sum += p.second;
                            }

                            std::string msg;
                            bool ok = verify_witness(inst.elements, inst.target, k, cand, msg);
                            if (!ok) {
                                total_false_positives.fetch_add(1, std::memory_order_relaxed);
                                continue;
                            }

                            std::unique_lock<std::mutex> lock(report_mutex);
                            if (!solution_found.load(std::memory_order_relaxed)) {
                                solution_found = true;
                                best_witness = cand;
                                report.verification_msg = msg;
                                cv_done.notify_all();
                            }
                            flush_counters();
                            return;
                        }
                    }
                }
                flush_counters();
            }
            flush_counters();
        };

        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) workers.emplace_back(worker, t);

        std::thread logger;
        std::atomic<bool> stop_logger(false);
        if (verbose) {
            logger = std::thread([&]() {
                double window_ratio = inst.k_window_ratio();
                std::unique_lock<std::mutex> lk(report_mutex);
                while (!solution_found.load() && !stop_logger.load()) {
                    cv_done.wait_for(lk, std::chrono::seconds(10), [&]() {
                        return solution_found.load() || stop_logger.load();
                    });
                    if (solution_found.load() || stop_logger.load()) break;
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    ProgressEstimate est = ProgressEstimate::compute(
                        el, total_partitions.load(), total_queries.load(), total_false_positives.load());
                    est.print(std::cout, window_ratio);
                }
            });
        }

        for (auto& th : workers) if (th.joinable()) th.join();
        {
            std::unique_lock<std::mutex> lock(report_mutex);
            stop_logger = true;
            cv_done.notify_all();
        }
        if (logger.joinable()) logger.join();

        auto end_time = std::chrono::steady_clock::now();
        report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        report.peak_ram_mb = get_current_peak_ram_mb();
        report.partitions_evaluated = total_partitions.load();
        report.modular_queries_evaluated = total_queries.load();
        report.base_combinations_generated = total_base_entries.load();
        report.half_candidates_generated = total_half_candidates.load();
        report.false_positives_rejected = total_false_positives.load();
        report.solved = solution_found.load();
        if (report.solved) report.witness = best_witness;
        else if (report.verification_msg.empty()) report.verification_msg = "NOT FOUND / TIMEOUT.";

        return report;
    }
};

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " instance.txt \"TARGET_DECIMAL\" [threads=0] [time_limit_s=0]"
                  << " [--calibrate_s=10] [--proof-mode]\n";
        return 1;
    }

    std::string file_path = argv[1], tgt_str = argv[2];
    unsigned num_threads = 0;
    double time_limit_s = 0.0, calibrate_s = 10.0;
    bool proof_mode = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--proof-mode") proof_mode = true;
        else if (a.rfind("--calibrate_s=", 0) == 0) calibrate_s = std::atof(a.substr(14).c_str());
        else if (num_threads == 0 && i == 3) num_threads = (unsigned)std::atoi(argv[i]);
        else if (i == 4) time_limit_s = std::atof(argv[i]);
    }

    Instance inst;
    if (!Instance::load(file_path, tgt_str, inst)) {
        std::cerr << "[ERROR] Gagal membaca instance dari: " << file_path << "\n";
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << " HGJ2 ADAPTIVE SOLVER (FAST FOR SMALL N, ANTI-OOM FOR LARGE N)\n";
    std::cout << "================================================================================\n";
    std::cout << "N elemen        : " << inst.elements.size() << "\n";
    std::cout << "Target          : " << inst.target << "\n";
    std::cout << "Densitas        : " << std::fixed << std::setprecision(4) << inst.density << "\n";
    std::cout << "K window (exact): [" << inst.k_min << " .. " << inst.k_max << "]"
              << "  (ratio=" << std::setprecision(3) << inst.k_window_ratio() << ")\n";

    if (inst.k_max < inst.k_min) {
        std::cout << "[UNSAT TERBUKTI DARI PREPROCESSING]\n";
        return 1;
    }

    if (calibrate_s > 0) {
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "[CALIBRATION] " << std::fixed << std::setprecision(1) << calibrate_s
                  << " detik, memakai pipeline HGJ2 yang sama.\n";
        Report cal = Solver::solve(inst, num_threads, calibrate_s, false, false);
        if (cal.solved) {
            std::cout << "[CALIBRATION] SOLVED sebelum long-run.\n";
            std::cout << "Runtime : " << std::fixed << std::setprecision(2) << cal.runtime_ms / 1000.0 << " s\n";
            std::cout << "Partisi : " << cal.partitions_evaluated << "\n";
            std::cout << "Peak RAM: " << std::fixed << std::setprecision(2) << cal.peak_ram_mb << " MB\n";
            return 0;
        }
    }

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "[RUNNING SOLVER]\n";

    Report report = Solver::solve(inst, num_threads, time_limit_s, proof_mode, true);

    std::cout << "\n================================================================================\n";
    std::cout << "                              LAPORAN EKSEKUSI\n";
    std::cout << "================================================================================\n";
    std::cout << "Status               : " << (report.solved ? "SOLVED (verified)" : "NOT SOLVED") << "\n";
    std::cout << "Runtime              : " << std::fixed << std::setprecision(2) << report.runtime_ms / 1000.0 << " detik\n";
    std::cout << "Partisi dievaluasi   : " << report.partitions_evaluated << "\n";
    std::cout << "Query modular        : " << report.modular_queries_evaluated << "\n";
    std::cout << "Kombinasi base total : " << report.base_combinations_generated << "\n";
    std::cout << "Half candidates      : " << report.half_candidates_generated << "\n";
    std::cout << "False-positive       : " << report.false_positives_rejected << "\n";
    std::cout << "Peak RAM             : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB\n";
    std::cout << "Threads              : " << report.threads_used << "\n";
    std::cout << "Pesan verifikasi     : " << report.verification_msg << "\n";

    if (report.solved) {
        std::cout << "SAKSI (" << report.witness.indices.size() << " indeks): [";
        for (size_t i = 0; i < report.witness.indices.size(); ++i)
            std::cout << report.witness.indices[i] << (i + 1 < report.witness.indices.size() ? ", " : "");
        std::cout << "]\nJumlah saksi         : " << report.witness.sum << "\n";
        std::cout << "Target               : " << inst.target << "\n";
        std::cout << "Cocok?               : " << (report.witness.sum == inst.target ? "YA, 100% PRESISI" : "TIDAK (bug!)") << "\n";
    }

    std::cout << "================================================================================\n";
    return report.solved ? 0 : 1;
}
