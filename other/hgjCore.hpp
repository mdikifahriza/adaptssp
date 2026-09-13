#pragma once

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
#include <random>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <numeric>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

// ============================================================================
// 96-bit / 128-bit Native Arithmetic Types & Helpers
// (u128 is KEPT intentionally -- required for exact precision at density~1
//  on 96-bit element instances. Optimizations below reduce HOW OFTEN and
//  HOW EXPENSIVELY u128 ops are invoked, not whether they're used.)
// ============================================================================
using u64  = uint64_t;
using u32  = uint32_t;
using u128 = unsigned __int128;

static const u128 INF128 = ~((u128)0);

inline std::string u128_to_string(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) {
        s.push_back((char)('0' + (int)(v % 10)));
        v /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

inline std::ostream& operator<<(std::ostream& os, u128 v) {
    return os << u128_to_string(v);
}

inline u128 parse_u128_decimal(const std::string& str) {
    u128 val = 0;
    for (char c : str) {
        if (c >= '0' && c <= '9') {
            val = val * 10 + (c - '0');
        }
    }
    return val;
}

inline double get_current_peak_ram_mb() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0;
    }
    return 0.0;
#endif
}

// ============================================================================
// Zero-padding helper: generalizes solver to N not divisible by 8, by
// inserting p (<=16) zero-valued elements. Not needed when N%8==0 already,
// but kept so the solver accepts arbitrary N safely.
// ============================================================================
inline int compute_hgj_padding(int n) {
    for (int p = 0; p <= 24; ++p) {
        int n2 = n + p;
        if (n2 >= 8 && (n2 % 8 == 0)) return p;
    }
    return 0;
}

// ============================================================================
// Data Structures
// ============================================================================
struct HGJInstance {
    std::vector<u128> elements;
    u128 target;
    u128 total_sum;
    double density;

    static bool load_from_txt_or_string(const std::string& file_or_str, const std::string& tgt_str, HGJInstance& inst) {
        std::string raw_content;
        std::ifstream infile(file_or_str);
        if (infile.is_open()) {
            std::string line;
            while (std::getline(infile, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                size_t p = std::string::npos;
                size_t p1 = line.find('#'), p2 = line.find("//"), p3 = line.find(';');
                if (p1 != std::string::npos) p = std::min(p, p1);
                if (p2 != std::string::npos) p = std::min(p, p2);
                if (p3 != std::string::npos) p = std::min(p, p3);
                if (p != std::string::npos) line = line.substr(0, p);
                raw_content += line + " ";
            }
        } else {
            raw_content = file_or_str;
        }

        inst.elements.clear();
        std::string cur;
        for (char c : raw_content) {
            if (isdigit((unsigned char)c)) {
                cur += c;
            } else {
                if (!cur.empty()) {
                    inst.elements.push_back(parse_u128_decimal(cur));
                    cur.clear();
                }
            }
        }
        if (!cur.empty()) inst.elements.push_back(parse_u128_decimal(cur));

        inst.target = parse_u128_decimal(tgt_str);
        inst.total_sum = 0;
        u128 max_val = 0;
        for (u128 val : inst.elements) {
            inst.total_sum += val;
            if (val > max_val) max_val = val;
        }

        double max_bits = (max_val == 0) ? 1.0 : (double)std::log2((double)max_val + 1.0);
        inst.density = (inst.elements.empty()) ? 0.0 : ((double)inst.elements.size() / max_bits);
        return !inst.elements.empty();
    }

    // Applies zero-padding in-place so that N % 8 == 0. Safe: padding elements
    // are value 0 and never change any achievable subset-sum.
    void apply_zero_padding_if_needed() {
        int n = (int)elements.size();
        int p = compute_hgj_padding(n);
        for (int i = 0; i < p; ++i) elements.push_back((u128)0);
    }
};

// Struct-of-Arrays base list. Sorting/merging only ever touches `rem`
// (8 bytes, cache-friendly); `sum` (16 bytes) and `mask` are only read
// once a candidate actually survives modular + interval filtering.
struct BaseListSoA {
    std::vector<u64>  rem;
    std::vector<u32>  mask;
    std::vector<u128> sum;

    void clear() { rem.clear(); mask.clear(); sum.clear(); }
    void reserve(size_t n) { rem.reserve(n); mask.reserve(n); sum.reserve(n); }
    size_t size() const { return rem.size(); }

    inline void push(u128 s, u32 m, u64 r) {
        sum.push_back(s);
        mask.push_back(m);
        rem.push_back(r);
    }
};

struct HalfEntry {
    u128 sum;
    u64  mask_half; // mask within half (combining Q1 and Q2)
};

struct HGJSolutionWitness {
    std::vector<int>  indices;
    std::vector<u128> values;
    u128              sum;
};

struct HGJExecutionReport {
    bool                solved = false;
    double              runtime_ms = 0.0;
    double              peak_ram_mb = 0.0;
    u64                 partitions_evaluated = 0;
    u64                 modular_queries_evaluated = 0;
    u64                 base_combinations_generated = 0;
    u64                 parity_prunes = 0;
    u64                 interval_prunes = 0;
    u64                 highbit_prunes = 0;
    u32                 threads_used = 1;
    bool                verified = false;
    std::string         verification_msg;
    HGJSolutionWitness  witness;
};

inline bool verify_hgj_witness(const std::vector<u128>& raw_elements, u128 target, int expected_k,
                               const HGJSolutionWitness& wit, std::string& out_msg) {
    if (wit.indices.size() != (size_t)expected_k) {
        out_msg = "FAILED: Jumlah elemen saksi (" + std::to_string(wit.indices.size()) +
                  ") != K=" + std::to_string(expected_k);
        return false;
    }
    std::vector<bool> seen(raw_elements.size(), false);
    u128 check_sum = 0;
    for (size_t i = 0; i < wit.indices.size(); ++i) {
        int idx = wit.indices[i];
        if (idx < 0 || idx >= (int)raw_elements.size()) {
            out_msg = "FAILED: Indeks saksi di luar jangkauan: " + std::to_string(idx);
            return false;
        }
        if (seen[idx]) {
            out_msg = "FAILED: Indeks duplikat terdeteksi pada indeks: " + std::to_string(idx);
            return false;
        }
        seen[idx] = true;
        if (raw_elements[idx] != wit.values[i]) {
            out_msg = "FAILED: Nilai elemen saksi tidak cocok dengan array mentah pada indeks " + std::to_string(idx);
            return false;
        }
        check_sum += wit.values[i];
    }
    if (check_sum != target) {
        out_msg = "FAILED: Jumlahan elemen saksi (" + u128_to_string(check_sum) +
                  ") != Target (" + u128_to_string(target) + ")";
        return false;
    }
    out_msg = "OK: 100% Saksi Eksak Terverifikasi Independen (Semua " + std::to_string(expected_k) + " Elemen Unik, Jumlah == Target).";
    return true;
}

// ============================================================================
// Scalable Multi-Core HGJ Representation Engine -- OPTIMIZED
//
// Changes applied vs baseline (all engineering / constant-factor, u128 kept):
//   1. generate_combinations: bitmask popcount-loop  ->  recursive DFS
//      include/exclude that carries (sum, rem) incrementally. Per-combination
//      cost drops from O(q_n) u128-add + 1 u128-mod  to  O(1) u128-add +
//      O(1) u64 add/conditional-subtract. The expensive `% M` on u128 is
//      now computed ONCE per base element (not once per combination).
//   2. Struct-of-Arrays (BaseListSoA) instead of AoS BaseEntry: sort/merge
//      only touch the 8-byte `rem` array, not the 16-byte `sum`.
//   3. Radix sort (2-pass, 13-bit digit) on `rem` instead of std::sort with
//      a struct comparator -- O(N) instead of O(N log N).
//   4. Atomic counters (partitions/queries/base-entries/prunes) accumulate
//      thread-locally and flush to the shared atomics only periodically,
//      eliminating per-iteration cache-line bouncing.
//   5. Interval bounds (min/max of half_k elements) computed via
//      std::nth_element (O(N) avg) instead of a full std::sort.
// ============================================================================
class HGJSolver {
public:
    static HGJExecutionReport solve(const HGJInstance& inst_in, unsigned num_threads = 0,
                                     double time_limit_s = 0.0, bool verbose = true) {
        HGJExecutionReport report;
        auto start_time = std::chrono::steady_clock::now();

        // Auto zero-pad if needed (generalizes away the strict N%8==0 requirement).
        HGJInstance inst = inst_in;
        inst.apply_zero_padding_if_needed();

        int n = (int)inst.elements.size();
        int k = n / 2;
        if (n < 4 || (n % 4 != 0) || (k % 4 != 0)) {
            report.solved = false;
            report.verification_msg = "ERROR: Instance tidak bisa dibuat kompatibel walau dengan zero-padding.";
            return report;
        }

        int q_n = n / 4;
        int q_k = k / 4;

        size_t est_combinations = 1;
        for (int i = 0; i < q_k; ++i) {
            est_combinations = est_combinations * (q_n - i) / (i + 1);
        }

        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }
        if (n <= 24) num_threads = 1;
        report.threads_used = num_threads;

        u64 M;
        if (est_combinations <= 15) {
            M = 31;
        } else if (est_combinations <= 50) {
            M = 61;
        } else if (est_combinations <= 150) {
            M = 127;
        } else if (est_combinations <= 500) {
            M = 257;
        } else if (est_combinations <= 2000) {
            M = 1009;
        } else if (est_combinations <= 6000) {
            M = 4093;
        } else if (est_combinations <= 20000) {
            M = 16381;
        } else if (est_combinations <= 80000) {
            M = 65537;
        } else if (est_combinations <= 300000) {
            M = 262139;
        } else if (est_combinations <= 1000000) {
            M = 1048573;
        } else {
            M = 35989843ULL;
        }

        // Radix width needed to cover M (rem < M).
        int radix_bits_total = 1;
        while ((1ULL << radix_bits_total) < M) radix_bits_total++;

        // Amortize the (expensive) generate+sort cost over more queries for
        // large instances, per suggestion (poin 7).
        int queries_per_partition;
        if (est_combinations >= 1000000)      queries_per_partition = 300;
        else if (est_combinations >= 500000)  queries_per_partition = 150;
        else if (est_combinations >= 5000)    queries_per_partition = 20;
        else                                  queries_per_partition = 5;

        std::atomic<bool> solution_found(false);
        std::atomic<u64>  total_partitions(0);
        std::atomic<u64>  total_queries(0);
        std::atomic<u64>  total_base_entries(0);
        std::atomic<u64>  total_parity_prunes(0);
        std::atomic<u64>  total_interval_prunes(0);
        std::atomic<u64>  total_highbit_prunes(0);

        std::mutex report_mutex;
        HGJSolutionWitness best_witness;

        size_t est_cand = (n >= 96) ? 262144 : 4096;

        auto worker = [&](unsigned thread_id) {
            std::mt19937_64 rng(1337ULL + thread_id * 10007ULL +
                               (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());

            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;

            BaseListSoA L1, L2, R1, R2;
            L1.reserve(est_combinations);
            L2.reserve(est_combinations);
            R1.reserve(est_combinations);
            R2.reserve(est_combinations);

            // Scratch buffers reused across partitions for radix sort (poin 3).
            std::vector<u32> idx_buf, idx_tmp;
            idx_buf.reserve(est_combinations);
            idx_tmp.reserve(est_combinations);

            std::vector<HalfEntry> cand_L, cand_R;
            cand_L.reserve(est_cand);
            cand_R.reserve(est_cand);

            std::vector<u128> Q1(q_n), Q2(q_n), Q3(q_n), Q4(q_n);
            std::vector<u64>  Q1r(q_n), Q2r(q_n), Q3r(q_n), Q4r(q_n);
            std::vector<u128> Left_elem(2 * q_n), Right_elem(2 * q_n);

            // Thread-local counters flushed at end of each partition or when solution is found.
            u64 local_partitions = 0, local_queries = 0, local_base_entries = 0, local_interval_prunes = 0;

            auto flush_counters = [&]() {
                if (local_partitions)      total_partitions += local_partitions;
                if (local_queries)         total_queries += local_queries;
                if (local_base_entries)    total_base_entries += local_base_entries;
                if (local_interval_prunes) total_interval_prunes += local_interval_prunes;
                local_partitions = local_queries = local_base_entries = local_interval_prunes = 0;
            };

            while (!solution_found) {
                if (time_limit_s > 0.0) {
                    auto now = std::chrono::steady_clock::now();
                    double el = std::chrono::duration<double>(now - start_time).count();
                    if (el >= time_limit_s) break;
                }

                local_partitions++;
                std::shuffle(perm.begin(), perm.end(), rng);

                for (int i = 0; i < q_n; ++i) {
                    Q1[i] = inst.elements[perm[i]];
                    Q2[i] = inst.elements[perm[q_n + i]];
                    Q3[i] = inst.elements[perm[2 * q_n + i]];
                    Q4[i] = inst.elements[perm[3 * q_n + i]];

                    Left_elem[i] = Q1[i];
                    Left_elem[q_n + i] = Q2[i];
                    Right_elem[i] = Q3[i];
                    Right_elem[q_n + i] = Q4[i];
                }

                // [OPTIMIZATION: nth_element instead of full sort for bounds]
                int half_k = 2 * q_k;
                u128 min_L = sum_smallest_k(Left_elem, half_k);
                u128 max_L = sum_largest_k(Left_elem, half_k);
                u128 min_R = sum_smallest_k(Right_elem, half_k);
                u128 max_R = sum_largest_k(Right_elem, half_k);

                u128 lower_bound_L = (inst.target > max_R) ? (inst.target - max_R) : 0;
                u128 upper_bound_L = (inst.target >= min_R) ? (inst.target - min_R) : 0;
                u128 lower_bound_R = (inst.target > max_L) ? (inst.target - max_L) : 0;
                u128 upper_bound_R = (inst.target >= min_L) ? (inst.target - min_L) : 0;

                // [OPTIMIZATION: incremental DFS combination generation]
                L1.clear(); L2.clear(); R1.clear(); R2.clear();
                for (int i = 0; i < q_n; ++i) {
                    Q1r[i] = (u64)(Q1[i] % (u128)M);
                    Q2r[i] = (u64)(Q2[i] % (u128)M);
                    Q3r[i] = (u64)(Q3[i] % (u128)M);
                    Q4r[i] = (u64)(Q4[i] % (u128)M);
                }
                generate_combinations_dfs(Q1, Q1r, q_n, q_k, M, L1);
                generate_combinations_dfs(Q2, Q2r, q_n, q_k, M, L2);
                generate_combinations_dfs(Q3, Q3r, q_n, q_k, M, R1);
                generate_combinations_dfs(Q4, Q4r, q_n, q_k, M, R2);

                local_base_entries += (u64)(L1.size() + L2.size() + R1.size() + R2.size());

                // [OPTIMIZATION: radix sort by rem instead of std::sort<struct>]
                radix_sort_by_rem(L1, idx_buf, idx_tmp, radix_bits_total);
                radix_sort_by_rem(L2, idx_buf, idx_tmp, radix_bits_total);
                radix_sort_by_rem(R1, idx_buf, idx_tmp, radix_bits_total);
                radix_sort_by_rem(R2, idx_buf, idx_tmp, radix_bits_total);

                u64 target_mod = (u64)(inst.target % (u128)M);

                for (int q_iter = 0; q_iter < queries_per_partition && !solution_found; ++q_iter) {
                    local_queries++;
                    u64 R_mod = rng() % M;
                    u64 R_comp_mod = (target_mod >= R_mod) ? (target_mod - R_mod) : (M - (R_mod - target_mod));

                    cand_L.clear();
                    merge_quarters_sweep(L1, L2, R_mod, M, lower_bound_L, upper_bound_L,
                                         cand_L, q_n, local_interval_prunes, solution_found);
                    if (solution_found || cand_L.empty()) continue;

                    cand_R.clear();
                    merge_quarters_sweep(R1, R2, R_comp_mod, M, lower_bound_R, upper_bound_R,
                                         cand_R, q_n, local_interval_prunes, solution_found);
                    if (solution_found || cand_R.empty()) continue;

                    std::sort(cand_R.begin(), cand_R.end(), [](const HalfEntry& a, const HalfEntry& b) {
                        return a.sum < b.sum;
                    });

                    for (const auto& cL : cand_L) {
                        if (solution_found) break;
                        if (cL.sum > inst.target) continue;

                        u128 req_sum = inst.target - cL.sum;
                        if (req_sum < cand_R.front().sum || req_sum > cand_R.back().sum) continue;

                        HalfEntry dummy{req_sum, 0};
                        auto range = std::equal_range(cand_R.begin(), cand_R.end(), dummy,
                            [](const HalfEntry& a, const HalfEntry& b) {
                                return a.sum < b.sum;
                            });

                        for (auto it = range.first; it != range.second; ++it) {
                            const auto& cR = *it;
                            if (cL.sum + cR.sum == inst.target) {
                                std::unique_lock<std::mutex> lock(report_mutex);
                                if (!solution_found) {
                                    solution_found = true;
                                    best_witness.indices.clear();
                                    best_witness.values.clear();

                                    for (int b = 0; b < q_n; ++b) {
                                        if ((cL.mask_half >> b) & 1ULL) {
                                            int idx = perm[b];
                                            best_witness.indices.push_back(idx);
                                            best_witness.values.push_back(inst.elements[idx]);
                                        }
                                        if ((cL.mask_half >> (q_n + b)) & 1ULL) {
                                            int idx = perm[q_n + b];
                                            best_witness.indices.push_back(idx);
                                            best_witness.values.push_back(inst.elements[idx]);
                                        }
                                    }
                                    for (int b = 0; b < q_n; ++b) {
                                        if ((cR.mask_half >> b) & 1ULL) {
                                            int idx = perm[2 * q_n + b];
                                            best_witness.indices.push_back(idx);
                                            best_witness.values.push_back(inst.elements[idx]);
                                        }
                                        if ((cR.mask_half >> (q_n + b)) & 1ULL) {
                                            int idx = perm[3 * q_n + b];
                                            best_witness.indices.push_back(idx);
                                            best_witness.values.push_back(inst.elements[idx]);
                                        }
                                    }

                                    std::vector<std::pair<int, u128>> pairs;
                                    for (size_t pidx = 0; pidx < best_witness.indices.size(); ++pidx) {
                                        pairs.push_back({best_witness.indices[pidx], best_witness.values[pidx]});
                                    }
                                    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                                        return a.first < b.first;
                                    });
                                    best_witness.indices.clear();
                                    best_witness.values.clear();
                                    u128 s = 0;
                                    for (const auto& p : pairs) {
                                        // Skip zero-padding indices from the final witness if they
                                        // happen to lie beyond original element count.
                                        best_witness.indices.push_back(p.first);
                                        best_witness.values.push_back(p.second);
                                        s += p.second;
                                    }
                                    best_witness.sum = s;
                                }
                                flush_counters();
                                return;
                            }
                        }
                    }
                }

                flush_counters();
            }
            flush_counters();
        };

        (void)verbose;

        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) {
            workers.emplace_back(worker, t);
        }
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }

        auto end_time = std::chrono::steady_clock::now();
        report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        report.peak_ram_mb = get_current_peak_ram_mb();
        report.partitions_evaluated = total_partitions.load();
        report.modular_queries_evaluated = total_queries.load();
        report.base_combinations_generated = total_base_entries.load();
        report.parity_prunes = total_parity_prunes.load();
        report.interval_prunes = total_interval_prunes.load();
        report.highbit_prunes = total_highbit_prunes.load();
        report.solved = solution_found.load();

        if (report.solved) {
            report.witness = best_witness;
            // Verify against the ORIGINAL (unpadded) element array/K, since padding
            // elements are synthetic and must not appear in a valid witness.
            int orig_n = (int)inst_in.elements.size();
            int orig_k = orig_n / 2;
            // Filter out any witness index that points past the original array
            // (i.e. a zero-padding element) -- with value 0 this cannot happen
            // in a real solution set of size > orig_k, but guard anyway.
            HGJSolutionWitness filtered;
            u128 s = 0;
            for (size_t i = 0; i < report.witness.indices.size(); ++i) {
                if (report.witness.indices[i] < orig_n) {
                    filtered.indices.push_back(report.witness.indices[i]);
                    filtered.values.push_back(report.witness.values[i]);
                    s += report.witness.values[i];
                }
            }
            filtered.sum = s;
            report.witness = filtered;
            report.verified = verify_hgj_witness(inst_in.elements, inst_in.target, orig_k, report.witness, report.verification_msg);
        } else {
            report.verified = false;
            report.verification_msg = "Solver dihentikan (Timeout / Belum Menemukan Solusi).";
        }

        return report;
    }

private:
    // ---- nth_element based min/max-of-k sum (poin 6) --------------------
    static u128 sum_smallest_k(std::vector<u128> v, int k) {
        if (k <= 0) return 0;
        std::nth_element(v.begin(), v.begin() + k, v.end());
        u128 s = 0;
        for (int i = 0; i < k; ++i) s += v[i];
        return s;
    }
    static u128 sum_largest_k(std::vector<u128> v, int k) {
        if (k <= 0) return 0;
        std::nth_element(v.begin(), v.end() - k, v.end());
        u128 s = 0;
        for (size_t i = v.size() - k; i < v.size(); ++i) s += v[i];
        return s;
    }

    // ---- Recursive DFS include/exclude combination generator (poin 2) ---
    // Carries (sum, rem) incrementally: rem is updated with a conditional
    // subtract, never a modulo, inside the hot loop. The one true `% M` per
    // base element is precomputed by the caller into `A_rem`.
    static void gen_recursive(int pos, int remaining_k, u128 cur_sum, u64 cur_rem, u32 cur_mask,
                               const std::vector<u128>& A, const std::vector<u64>& A_rem,
                               int q_n, u64 M, BaseListSoA& out) {
        if (remaining_k == 0) {
            out.push(cur_sum, cur_mask, cur_rem);
            return;
        }
        if (pos >= q_n || (q_n - pos) < remaining_k) return;

        // exclude A[pos]
        gen_recursive(pos + 1, remaining_k, cur_sum, cur_rem, cur_mask, A, A_rem, q_n, M, out);

        // include A[pos] -- O(1) incremental update, no modulo
        u64 new_rem = cur_rem + A_rem[pos];
        if (new_rem >= M) new_rem -= M;
        gen_recursive(pos + 1, remaining_k - 1, cur_sum + A[pos], new_rem,
                      cur_mask | (1U << pos), A, A_rem, q_n, M, out);
    }

    static void generate_combinations_dfs(const std::vector<u128>& A, const std::vector<u64>& A_rem,
                                          int q_n, int q_k, u64 M, BaseListSoA& out) {
        gen_recursive(0, q_k, (u128)0, (u64)0, (u32)0, A, A_rem, q_n, M, out);
    }

    // ---- LSD radix sort on `rem`, permuting mask/sum along with it (poin 3+4) --
    static void radix_sort_by_rem(BaseListSoA& list, std::vector<u32>& idx, std::vector<u32>& idx_tmp,
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
            for (u32 id : *src) {
                u32 digit = (u32)((list.rem[id] >> shift) & (RADIX_SIZE - 1));
                count[digit + 1]++;
            }
            for (u32 i = 0; i < RADIX_SIZE; ++i) count[i + 1] += count[i];
            for (u32 id : *src) {
                u32 digit = (u32)((list.rem[id] >> shift) & (RADIX_SIZE - 1));
                (*dst)[count[digit]++] = id;
            }
            std::swap(src, dst);
        }

        // Gather into sorted order (only needed if the final result landed in idx_tmp).
        std::vector<u64>  new_rem(n);
        std::vector<u32>  new_mask(n);
        std::vector<u128> new_sum(n);
        for (size_t i = 0; i < n; ++i) {
            u32 id = (*src)[i];
            new_rem[i] = list.rem[id];
            new_mask[i] = list.mask[id];
            new_sum[i] = list.sum[id];
        }
        list.rem = std::move(new_rem);
        list.mask = std::move(new_mask);
        list.sum = std::move(new_sum);
    }

    // ---- Two-pointer sweep over SoA lists (adapted from baseline) -------
    static void merge_quarters_sweep(const BaseListSoA& list1, const BaseListSoA& list2,
                                     u64 target_mod, u64 M,
                                     u128 lower_bound, u128 upper_bound,
                                     std::vector<HalfEntry>& out_cand,
                                     int q_n,
                                     u64& local_interval_prunes,
                                     const std::atomic<bool>& solution_found) {
        size_t N1 = list1.size();
        size_t N2 = list2.size();
        if (N1 == 0 || N2 == 0) return;

        const auto& rem1 = list1.rem; const auto& sum1 = list1.sum; const auto& mask1 = list1.mask;
        const auto& rem2 = list2.rem; const auto& sum2 = list2.sum; const auto& mask2 = list2.mask;

        // Case 1: rem1 + rem2 == target_mod
        size_t i = 0;
        int64_t j = (int64_t)N2 - 1;
        while (j >= 0 && rem2[j] > target_mod) j--;

        while (i < N1 && rem1[i] <= target_mod && j >= 0) {
            if (solution_found) return;
            u64 s = rem1[i] + rem2[j];
            if (s < target_mod) {
                i++;
            } else if (s > target_mod) {
                j--;
            } else {
                size_t i2 = i;
                while (i2 < N1 && rem1[i2] == rem1[i]) i2++;
                int64_t j2 = j;
                while (j2 >= 0 && rem2[j2] == rem2[j]) j2--;

                for (size_t a = i; a < i2; ++a) {
                    for (int64_t b = j; b > j2; --b) {
                        u128 half_sum = sum1[a] + sum2[b];
                        if (half_sum < lower_bound || half_sum > upper_bound) {
                            local_interval_prunes++;
                            continue;
                        }
                        u64 combined_mask = ((u64)mask2[b] << q_n) | (u64)mask1[a];
                        out_cand.push_back({half_sum, combined_mask});
                    }
                }
                i = i2;
                j = j2;
            }
        }

        // Case 2: rem1 + rem2 == target_mod + M
        u64 target2 = target_mod + M;
        i = 0;
        while (i < N1 && rem1[i] <= target_mod) i++;
        j = (int64_t)N2 - 1;

        while (i < N1 && j >= 0 && rem2[j] > target_mod) {
            if (solution_found) return;
            u64 s = rem1[i] + rem2[j];
            if (s < target2) {
                i++;
            } else if (s > target2) {
                j--;
            } else {
                size_t i2 = i;
                while (i2 < N1 && rem1[i2] == rem1[i]) i2++;
                int64_t j2 = j;
                while (j2 >= 0 && rem2[j2] == rem2[j]) j2--;

                for (size_t a = i; a < i2; ++a) {
                    for (int64_t b = j; b > j2; --b) {
                        u128 half_sum = sum1[a] + sum2[b];
                        if (half_sum < lower_bound || half_sum > upper_bound) {
                            local_interval_prunes++;
                            continue;
                        }
                        u64 combined_mask = ((u64)mask2[b] << q_n) | (u64)mask1[a];
                        out_cand.push_back({half_sum, combined_mask});
                    }
                }
                i = i2;
                j = j2; 
            }
        }
    }
};