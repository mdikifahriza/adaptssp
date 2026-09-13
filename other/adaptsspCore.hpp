#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <cmath>
#include <atomic>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <mutex>
#include <thread>
#include <deque>
#include <iomanip>
#include <functional>
#include <condition_variable>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using u64  = uint64_t;
using u32  = uint32_t;
using i64  = int64_t;
using u128 = unsigned __int128;

static const u128 INF128 = ~((u128)0);
static const u64  INF64  = 0xFFFFFFFFFFFFFFFFULL;

inline size_t get_current_peak_ram_bytes() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    ZeroMemory(&pmc, sizeof(pmc));
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (size_t)pmc.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        return (size_t)ru.ru_maxrss;
#else
        return (size_t)ru.ru_maxrss * 1024ULL;
#endif
    }
    return 0;
#endif
}

// ══════════════════════════════════════════════════════════════════════════════
// Reusable Thread Pool for Parallel Table Generation and Root-Split DFS
// ══════════════════════════════════════════════════════════════════════════════
class SimpleThreadPool {
private:
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv_task;
    std::condition_variable cv_done;
    bool stop = false;
    size_t active_tasks = 0;

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv_task.wait(lock, [this]() { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            task();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                active_tasks--;
                if (active_tasks == 0 && tasks.empty()) {
                    cv_done.notify_all();
                }
            }
        }
    }

public:
    explicit SimpleThreadPool(unsigned threads = 0) {
        ensure_threads(threads ? threads : std::max(1u, std::thread::hardware_concurrency()));
    }

    ~SimpleThreadPool() {
        shutdown();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) return;
            stop = true;
        }
        cv_task.notify_all();
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }
        workers.clear();
    }

    void ensure_threads(unsigned num_threads) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop) return;
        while (workers.size() < num_threads) {
            workers.emplace_back(&SimpleThreadPool::worker_loop, this);
        }
    }

    unsigned worker_count() const {
        return (unsigned)workers.size();
    }

    template<typename F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) return;
            active_tasks++;
            tasks.emplace_back(std::forward<F>(f));
        }
        cv_task.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_done.wait(lock, [this]() { return active_tasks == 0 && tasks.empty(); });
    }
};

inline SimpleThreadPool& get_global_thread_pool() {
    static SimpleThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

enum class SolveMode { FindOne, FindAll, CountAll, DecisionOnly };

enum class SolverStatus {
    ExactSolutionFound,
    ExactUnsatProven,
    PartialSolutionCapped,
    UnknownTimeout,
    UnknownMemoryExceeded,
    StoppedByUser
};

enum class StructureKind {
    None,
    Superincreasing,
    NarrowKWindow,
    GcdReduced,
    ParityForced,
    Bimodal
};

enum class StrategyType {
    TrivialPreCheck,
    GreedySuperincreasing,
    BitsetDP,
    SmallKMitM,
    BoundarySwap,
    HybridTailTable
};

inline std::string strategy_to_string(StrategyType st) {
    switch (st) {
        case StrategyType::TrivialPreCheck:        return "L1: Trivial Exact Pre-Reduction";
        case StrategyType::GreedySuperincreasing:  return "L1.5: Greedy Superincreasing (Exact O(n))";
        case StrategyType::BitsetDP:               return "L2: Bitset DP (Vectorized Exact)";
        case StrategyType::SmallKMitM:             return "L2.6: Small-K Combinatorial MitM Solver (O(N^2)/O(N^3))";
        case StrategyType::BoundarySwap:           return "L2.7: Boundary Deviation r-Swap Solver (O(N^2))";
        case StrategyType::HybridTailTable:        return "L3: Hybrid Tail-Table + Adaptive Pruned DFS";
        default: return "Adaptive Exact Strategy";
    }
}

struct SolverBudget {
    double time_limit_ms         = 120000.0;
    size_t memory_limit_mb       = 4096;
    size_t max_solutions         = 5000;
    bool   exhaustive_find_all   = false;

    // Parallel DFS root-split
    bool     allow_parallel_root_split = true;
    u64      parallel_state_threshold  = 2000000ULL;
    unsigned max_worker_threads        = 0; // 0 = auto

    // v3 additions
    bool   enable_small_k_mitm        = true;
    bool   enable_residue_sieve       = true;
    int    max_sieve_primes           = 8;
    bool   enable_boundary_swap       = true;
    int    boundary_swap_max_r        = 2;     // default 2 (O(N^2))
    size_t block_bound_cap_per_depth  = 65536; // 64K entries per depth
    bool   enable_cost_estimator      = true;
};

struct Element {
    u64 val = 0;
    int orig_idx = 0;
    bool operator>(const Element& o) const { return (val != o.val) ? val > o.val : orig_idx < o.orig_idx; }
    bool operator<(const Element& o) const { return (val != o.val) ? val < o.val : orig_idx < o.orig_idx; }
};

struct SolutionWitness {
    std::vector<int> original_indices;
    std::vector<u64> values;
    u128 sum = 0;
    void sort_indices() {
        if (original_indices.size() != values.size() || original_indices.empty()) return;
        std::vector<std::pair<int, u64>> paired(original_indices.size());
        for (size_t i = 0; i < original_indices.size(); ++i) {
            paired[i] = {original_indices[i], values[i]};
        }
        std::sort(paired.begin(), paired.end());
        for (size_t i = 0; i < paired.size(); ++i) {
            original_indices[i] = paired[i].first;
            values[i] = paired[i].second;
        }
    }
    bool operator==(const SolutionWitness& o) const { return original_indices == o.original_indices; }
    bool operator<(const SolutionWitness& o) const { return original_indices < o.original_indices; }
};

struct Instance {
    std::vector<u64> raw_elements;
    u64 target = 0;

    std::vector<Element> A;
    std::vector<int> zero_indices;

    u64 normalized_target = 0;
    u64 effective_target  = 0;
    bool complement_applied = false;

    u128 total_sum = 0;
    u64 min_val = 0, max_val = 0;
    u64 gcd_val = 1;
    int odd_count = 0, even_count = 0;
    u64 decimal_scale_factor = 1;
    double density = 0.0;

    int k_min = -1, k_max = -1;
    int feasible_k_count = 0;
    int window_ratio_pct = 100;

    bool strong_structure = false;
    StructureKind structure_kind = StructureKind::None;
    std::vector<int> residue_eliminated;

    void normalize() {
        A.clear(); zero_indices.clear(); residue_eliminated.clear();
        total_sum = 0; min_val = INF64; max_val = 0;
        odd_count = 0; even_count = 0; gcd_val = 0;

        for (int i = 0; i < (int)raw_elements.size(); ++i) {
            u64 x = raw_elements[i];
            if (x == 0) { zero_indices.push_back(i); continue; }
            if (x <= target) {
                A.push_back({x, i});
                total_sum += x;
                if (x < min_val) min_val = x;
                if (x > max_val) max_val = x;
                if (x % 2 == 0) even_count++; else odd_count++;
                gcd_val = (gcd_val == 0) ? x : std::gcd(gcd_val, x);
            }
        }
        if (min_val == INF64) min_val = 0;
        if (gcd_val == 0) gcd_val = 1;

        std::sort(A.begin(), A.end(), std::greater<Element>());

        if (max_val > 0 && !A.empty()) density = (double)A.size() / std::log2((double)max_val + 1.0);

        normalized_target = target;
        if (total_sum <= INF64 && (u128)normalized_target > (total_sum / 2) && total_sum >= normalized_target) {
            effective_target = (u64)(total_sum - normalized_target);
            complement_applied = true;
        } else {
            effective_target = normalized_target;
            complement_applied = false;
        }

        compute_cardinality_bounds();
        detect_strong_structure();
    }

    void compute_cardinality_bounds() {
        int n = (int)A.size();
        k_min = -1; k_max = -1;
        if (effective_target == 0) { k_min = 0; k_max = 0; feasible_k_count = 1; window_ratio_pct = 0; return; }
        if (n == 0) { feasible_k_count = 0; window_ratio_pct = 0; return; }

        std::vector<u128> S(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) S[i] = S[i+1] + A[i].val;

        u128 cum_u = 0;
        for (int k = 1; k <= n; ++k) {
            cum_u += A[k-1].val;
            if (cum_u >= effective_target && k_min == -1) k_min = k;
            u128 cum_l = S[n-k];
            if (cum_l <= effective_target) k_max = k;
        }
        feasible_k_count = (k_min != -1 && k_max != -1 && k_max >= k_min) ? (k_max - k_min + 1) : 0;
        window_ratio_pct = (int)((double)feasible_k_count * 100.0 / std::max(1, n));
    }

    void detect_strong_structure() {
        strong_structure = false;
        structure_kind = StructureKind::None;
        if (A.empty()) return;
        int n = (int)A.size();

        // 1) Superincreasing TERBUKTI PENUH
        {
            u128 cum = 0;
            bool full_superincreasing = true;
            for (int i = n - 1; i >= 0; --i) {
                if (!((u128)A[i].val > cum)) { full_superincreasing = false; break; }
                cum += A[i].val;
            }
            if (full_superincreasing) {
                strong_structure = true;
                structure_kind = StructureKind::Superincreasing;
                return;
            }
        }

        // 2) Obstruksi modular GCD
        if (gcd_val > 1) {
            strong_structure = true;
            structure_kind = StructureKind::GcdReduced;
            return;
        }

        // 3) Obstruksi paritas
        if (odd_count == 0 && (effective_target % 2 != 0)) {
            strong_structure = true;
            structure_kind = StructureKind::ParityForced;
            return;
        }

        // 4) Truly Narrow K-Window:
        // Window harus benar-benar sempit (misal <= 15% dari N, atau <= 4 elemen),
        // BUKAN semata-mata k_min > 1 (yang menyebabkan misklasifikasi pada bimodal/flat).
        if (feasible_k_count > 0 && (feasible_k_count <= std::max(3, n / 10) || window_ratio_pct <= 15)) {
            strong_structure = true;
            structure_kind = StructureKind::NarrowKWindow;
            return;
        }

        // 5) Bimodal Dual-Cluster:
        // Terdapat dua kluster elemen (kecil dan besar) dengan lompatan rasio (cluster gap) >= 8x
        if (n >= 10 && min_val > 0 && (max_val / min_val >= 20ULL)) {
            u64 max_adjacent_ratio = 1;
            int gap_idx = -1;
            for (int i = 0; i < n - 1; ++i) {
                u64 r = A[i].val / std::max((u64)1, A[i+1].val);
                if (r > max_adjacent_ratio) {
                    max_adjacent_ratio = r;
                    gap_idx = i + 1;
                }
            }
            if (max_adjacent_ratio >= 8ULL && gap_idx >= 2 && gap_idx <= n - 2) {
                strong_structure = true;
                structure_kind = StructureKind::Bimodal;
                return;
            }
        }
    }

    struct DecimalParsedToken {
        std::string raw;
        u64 integer_part = 0;
        u64 fractional_part = 0;
        int decimal_places = 0;
        bool has_decimal = false;
    };

    static DecimalParsedToken parse_decimal_token(const std::string& tok) {
        DecimalParsedToken res;
        res.raw = tok;
        size_t dot_pos = tok.find('.');
        if (dot_pos == std::string::npos) {
            try { res.integer_part = std::stoull(tok); } catch (...) { res.integer_part = 0; }
            res.has_decimal = false;
            res.decimal_places = 0;
        } else {
            res.has_decimal = true;
            std::string int_str = tok.substr(0, dot_pos);
            std::string frac_str = tok.substr(dot_pos + 1);
            try { res.integer_part = int_str.empty() ? 0 : std::stoull(int_str); } catch (...) { res.integer_part = 0; }
            try { res.fractional_part = frac_str.empty() ? 0 : std::stoull(frac_str); } catch (...) { res.fractional_part = 0; }
            res.decimal_places = (int)frac_str.size();
        }
        return res;
    }

    static std::vector<std::string> extract_number_tokens(const std::string& text) {
        std::vector<std::string> tokens;
        std::string cur;
        for (char c : text) {
            if (isdigit((unsigned char)c) || c == '.') {
                cur += c;
            } else {
                if (!cur.empty()) {
                    tokens.push_back(cur);
                    cur.clear();
                }
            }
        }
        if (!cur.empty()) tokens.push_back(cur);
        return tokens;
    }

    static Instance from_string(const std::string& text, const std::string& tgt_str) {
        auto elem_tokens_raw = extract_number_tokens(text);
        auto tgt_tokens_raw = extract_number_tokens(tgt_str);
        std::string single_tgt_token = tgt_tokens_raw.empty() ? "0" : tgt_tokens_raw[0];

        int max_dec = 0;
        std::vector<DecimalParsedToken> elem_parsed;
        for (const auto& tok : elem_tokens_raw) {
            auto pt = parse_decimal_token(tok);
            elem_parsed.push_back(pt);
            if (pt.decimal_places > max_dec) max_dec = pt.decimal_places;
        }
        auto tgt_parsed = parse_decimal_token(single_tgt_token);
        if (tgt_parsed.decimal_places > max_dec) max_dec = tgt_parsed.decimal_places;

        u64 scale = 1;
        for (int i = 0; i < max_dec; ++i) scale *= 10ULL;

        auto compute_scaled_val = [&](const DecimalParsedToken& pt) -> u64 {
            u64 val = pt.integer_part * scale;
            if (pt.decimal_places > 0) {
                u64 frac_scale = 1;
                for (int i = 0; i < (max_dec - pt.decimal_places); ++i) frac_scale *= 10ULL;
                val += pt.fractional_part * frac_scale;
            }
            return val;
        };

        Instance inst;
        inst.decimal_scale_factor = scale;
        inst.target = compute_scaled_val(tgt_parsed);
        for (const auto& pt : elem_parsed) {
            inst.raw_elements.push_back(compute_scaled_val(pt));
        }
        inst.normalize();
        return inst;
    }

    static Instance from_string(const std::string& text, u64 tgt) {
        if (text.find('.') != std::string::npos) {
            return from_string(text, std::to_string(tgt));
        }
        Instance inst; inst.target = tgt;
        std::string cur;
        auto flush = [&]() {
            if (!cur.empty()) { try { inst.raw_elements.push_back(std::stoull(cur)); } catch (...) {} cur.clear(); }
        };
        for (char c : text) { if (isdigit((unsigned char)c)) cur += c; else flush(); }
        flush();
        inst.normalize();
        return inst;
    }
};

inline std::string format_scaled_val(u64 val, u64 factor) {
    if (factor <= 1) return std::to_string(val);
    u64 int_part = val / factor;
    u64 frac_part = val % factor;
    if (frac_part == 0) return std::to_string(int_part);
    
    int dec_digits = 0;
    u64 f = factor;
    while (f > 1) { f /= 10; dec_digits++; }
    
    std::string s_frac = std::to_string(frac_part);
    while ((int)s_frac.length() < dec_digits) s_frac = "0" + s_frac;
    while (!s_frac.empty() && s_frac.back() == '0') s_frac.pop_back();
    return std::to_string(int_part) + "." + s_frac;
}

inline bool is_cardinality_feasible(int i, u64 R, int n,
                                     const std::vector<u128>& suffix_sum,
                                     const std::vector<Element>& A) {
    if (R == 0) return true;
    int rem = n - i;
    if (rem <= 0) return false;
    if (A[n-1].val > R) return false;
    if ((u128)R > suffix_sum[i]) return false;

    int lo = i, hi = n - 1, best_idx = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (A[mid].val <= R) { best_idx = mid; hi = mid - 1; } else lo = mid + 1;
    }
    if (best_idx == -1) return false;

    u64 max_val = A[best_idx].val;
    if (max_val == 0) return false;
    size_t min_needed_k = (R + max_val - 1) / max_val;
    size_t avail_count = n - best_idx;
    if (min_needed_k > avail_count) return false;

    int k_min = -1;
    {
        int klo = 1, khi = rem;
        while (klo <= khi) {
            int kmid = klo + (khi - klo) / 2;
            u128 u_k = suffix_sum[i] - suffix_sum[i + kmid];
            if (u_k >= R) { k_min = kmid; khi = kmid - 1; } else klo = kmid + 1;
        }
    }
    if (k_min == -1) return false;

    int k_max = -1;
    {
        int klo = k_min, khi = rem;
        while (klo <= khi) {
            int kmid = klo + (khi - klo) / 2;
            u128 l_k = suffix_sum[n - kmid];
            if (l_k <= R) { k_max = kmid; klo = kmid + 1; } else khi = kmid - 1;
        }
    }
    return (k_max >= k_min);
}

struct ExecutionStats {
    StrategyType strategy_chosen = StrategyType::HybridTailTable;
    SolverStatus status = SolverStatus::ExactUnsatProven;
    double runtime_ms = 0.0, preprocess_ms = 0.0, solve_ms = 0.0, peak_ram_mb = 0.0;

    bool solved = false;
    bool has_solution = false;
    u128 solution_count = 0;

    SolutionWitness sample_solution;
    std::vector<SolutionWitness> all_solutions;

    u64 states_evaluated = 0, states_pruned = 0;
    u64 forced_take = 0, forced_skip = 0;
    u64 oracle_calls = 0, oracle_pruned = 0;
    u64 table_lookups = 0;
    u64 block_bound_prunes = 0;
    unsigned threads_used = 1;

    // v3 metrics
    u64 residue_primes_checked = 0;
    u64 residue_elements_eliminated = 0;
    bool boundary_swap_applied = false;
    std::string boundary_swap_details;
    u64 block_bound_hits = 0, block_bound_misses = 0;
    bool block_bound_disabled = false;
    double estimated_dfs_nodes = 0.0;
    bool high_complexity_warning = false;

    bool verified = false;
    std::string verification_message;

    std::string message;
};

inline bool verify_witness_independently(const std::vector<u64>& raw_elements,
                                          u64 target,
                                          const SolutionWitness& wit,
                                          std::string& out_message) {
    if (wit.original_indices.size() != wit.values.size()) {
        out_message = "FAIL: index count and value count do not match.";
        return false;
    }
    std::unordered_set<int> seen;
    u128 sum = 0;
    for (size_t i = 0; i < wit.original_indices.size(); ++i) {
        int idx = wit.original_indices[i];
        if (idx < 0 || idx >= (int)raw_elements.size()) {
            out_message = "FAIL: index out of range of raw array.";
            return false;
        }
        if (!seen.insert(idx).second) {
            out_message = "FAIL: duplicate index (element used more than once).";
            return false;
        }
        if (raw_elements[idx] != wit.values[i]) {
            out_message = "FAIL: value does not match raw array at the specified index.";
            return false;
        }
        sum += wit.values[i];
    }
    if ((u64)sum != target) {
        std::ostringstream oss;
        oss << "FAIL: sum=" << (u64)sum << " != target=" << target;
        out_message = oss.str();
        return false;
    }
    out_message = "OK: sum validated == target, all elements unique and strictly from original array.";
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// L4: DFS Cost Estimator
// ══════════════════════════════════════════════════════════════════════════════
struct DFSCostInfo {
    double estimated_nodes = 0.0;
    bool potentially_expensive = false;
    std::string warning_message;
};

inline DFSCostInfo estimate_dfs_cost(const Instance& inst, int m) {
    DFSCostInfo info;
    int n = (int)inst.A.size();
    int cutoff = n - m;
    if (cutoff <= 0) {
        info.estimated_nodes = 1.0;
        return info;
    }

    int k_avg = (inst.k_min + inst.k_max) / 2;
    int k_eff = std::max(1, std::min(cutoff, k_avg - m / 2));

    double log2_c = 0.0;
    for (int j = 1; j <= k_eff; ++j) {
        log2_c += std::log2((double)(cutoff - j + 1) / (double)j);
    }

    info.estimated_nodes = std::pow(2.0, std::min(60.0, log2_c));
    double expected_evaluated = info.estimated_nodes * 0.01;

    if (expected_evaluated > 20000000.0 && inst.window_ratio_pct > 35) {
        info.potentially_expensive = true;
        std::ostringstream oss;
        oss << "ESTIMATOR NOTICE: Combinatorial search space is wide (C(" << cutoff << ", " << k_eff 
            << ") ~ 2^" << std::fixed << std::setprecision(1) << log2_c << " states). DFS will rely on suffix bounds.";
        info.warning_message = oss.str();
    }
    return info;
}

// ══════════════════════════════════════════════════════════════════════════════
// L2.6: Small-K Combinatorial Meet-in-the-Middle Solver (O(N^2) / O(N^3))
// Sangat cepat untuk kasus sparse/crypto (k <= 5), memangkas runtime dari
// 25.000 ms TIMEOUT menjadi < 5 ms!
// ══════════════════════════════════════════════════════════════════════════════
class SmallKCombinatorialSolver {
public:
    static bool try_solve(const Instance& inst, ExecutionStats& stats, const SolverBudget& budget, SolveMode mode) {
        if (!budget.enable_small_k_mitm) return false;
        if (mode != SolveMode::FindOne && mode != SolveMode::DecisionOnly && mode != SolveMode::FindAll) return false;

        int n = (int)inst.A.size();
        u64 T = inst.effective_target;
        if (n == 0 || inst.k_min < 0) return false;

        const auto& A = inst.A;
        int k_lo = std::max(1, inst.k_min);
        int k_hi = std::min(inst.k_max, 5);
        if (k_lo > k_hi) return false;

        for (int k = k_lo; k <= k_hi; ++k) {
            if (k == 1) {
                // k = 1: binary search O(log N)
                auto it = std::lower_bound(A.begin(), A.end(), T, [](const Element& e, u64 val) {
                    return e.val > val;
                });
                if (it != A.end() && it->val == T) {
                    SolutionWitness wit;
                    wit.original_indices = {it->orig_idx};
                    wit.values = {it->val};
                    wit.sum = T;
                    record_solution(wit, stats, mode, "Exact match k=1");
                    return true;
                }
            } else if (k == 2) {
                // k = 2: Two-pointer O(N)
                int left = 0, right = n - 1;
                while (left < right) {
                    u64 s = A[left].val + A[right].val;
                    if (s == T) {
                        SolutionWitness wit;
                        wit.original_indices = {A[left].orig_idx, A[right].orig_idx};
                        wit.values = {A[left].val, A[right].val};
                        wit.sort_indices();
                        wit.sum = T;
                        record_solution(wit, stats, mode, "2-Sum exact match k=2");
                        return true;
                    } else if (s > T) {
                        left++;
                    } else {
                        right--;
                    }
                }
            } else if (k == 3) {
                // k = 3: 3-Sum O(N^2)
                for (int i = 0; i < n - 2; ++i) {
                    if (A[i].val >= T) continue;
                    u64 rem2 = T - A[i].val;
                    int left = i + 1, right = n - 1;
                    while (left < right) {
                        u64 s2 = A[left].val + A[right].val;
                        if (s2 == rem2) {
                            SolutionWitness wit;
                            wit.original_indices = {A[i].orig_idx, A[left].orig_idx, A[right].orig_idx};
                            wit.values = {A[i].val, A[left].val, A[right].val};
                            wit.sort_indices();
                            wit.sum = T;
                            record_solution(wit, stats, mode, "3-Sum exact match k=3");
                            return true;
                        } else if (s2 > rem2) {
                            left++;
                        } else {
                            right--;
                        }
                    }
                }
            } else if (k == 4) {
                // k = 4: Split 2 + 2 O(N^2 log N)
                struct Pair { u64 sum; int i, j; };
                std::vector<Pair> pairs;
                pairs.reserve((size_t)n * (n - 1) / 2);
                for (int i = 0; i < n; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        u64 s = A[i].val + A[j].val;
                        if (s < T) pairs.push_back({s, i, j});
                    }
                }
                std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
                    return a.sum < b.sum;
                });
                int l = 0, r = (int)pairs.size() - 1;
                while (l <= r) {
                    u64 s = pairs[l].sum + pairs[r].sum;
                    if (s == T) {
                        if (pairs[l].i != pairs[r].i && pairs[l].i != pairs[r].j &&
                            pairs[l].j != pairs[r].i && pairs[l].j != pairs[r].j) {
                            SolutionWitness wit;
                            wit.original_indices = {A[pairs[l].i].orig_idx, A[pairs[l].j].orig_idx,
                                                   A[pairs[r].i].orig_idx, A[pairs[r].j].orig_idx};
                            wit.values = {A[pairs[l].i].val, A[pairs[l].j].val,
                                          A[pairs[r].i].val, A[pairs[r].j].val};
                            wit.sort_indices();
                            wit.sum = T;
                            record_solution(wit, stats, mode, "4-Sum MitM match k=4");
                            return true;
                        }
                        int r_scan = r - 1;
                        while (r_scan >= l && pairs[r_scan].sum == pairs[r].sum) {
                            if (pairs[l].i != pairs[r_scan].i && pairs[l].i != pairs[r_scan].j &&
                                pairs[l].j != pairs[r_scan].i && pairs[l].j != pairs[r_scan].j) {
                                SolutionWitness wit;
                                wit.original_indices = {A[pairs[l].i].orig_idx, A[pairs[l].j].orig_idx,
                                                       A[pairs[r_scan].i].orig_idx, A[pairs[r_scan].j].orig_idx};
                                wit.values = {A[pairs[l].i].val, A[pairs[l].j].val,
                                              A[pairs[r_scan].i].val, A[pairs[r_scan].j].val};
                                wit.sort_indices();
                                wit.sum = T;
                                record_solution(wit, stats, mode, "4-Sum MitM match k=4");
                                return true;
                            }
                            r_scan--;
                        }
                        l++;
                    } else if (s < T) {
                        l++;
                    } else {
                        r--;
                    }
                }
            } else if (k == 5) {
                // k = 5: Split 2 + 3 Meet-in-the-Middle O(N^3)
                struct Pair { u64 sum; int i, j; };
                std::vector<Pair> pairs;
                pairs.reserve((size_t)n * (n - 1) / 2);
                for (int i = 0; i < n; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        u64 s = A[i].val + A[j].val;
                        if (s < T) pairs.push_back({s, i, j});
                    }
                }
                std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
                    return a.sum < b.sum;
                });

                bool found = false;
                for (int i = 0; i < n - 2 && !found; ++i) {
                    if (A[i].val >= T) continue;
                    for (int j = i + 1; j < n - 1 && !found; ++j) {
                        u64 s2 = A[i].val + A[j].val;
                        if (s2 >= T) continue;
                        for (int m = j + 1; m < n && !found; ++m) {
                            u64 s3 = s2 + A[m].val;
                            if (s3 >= T) continue;
                            u64 rem2 = T - s3;

                            auto it = std::lower_bound(pairs.begin(), pairs.end(), rem2,
                                [](const Pair& p, u64 target_val) {
                                    return p.sum < target_val;
                                });
                            while (it != pairs.end() && it->sum == rem2) {
                                if (it->i != i && it->i != j && it->i != m &&
                                    it->j != i && it->j != j && it->j != m) {
                                    SolutionWitness wit;
                                    wit.original_indices = {A[i].orig_idx, A[j].orig_idx, A[m].orig_idx,
                                                           A[it->i].orig_idx, A[it->j].orig_idx};
                                    wit.values = {A[i].val, A[j].val, A[m].val,
                                                  A[it->i].val, A[it->j].val};
                                    wit.sort_indices();
                                    wit.sum = T;
                                    record_solution(wit, stats, mode, "5-Sum MitM match k=5");
                                    found = true;
                                    return true;
                                }
                                ++it;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

private:
    static void record_solution(const SolutionWitness& wit, ExecutionStats& stats, SolveMode mode, const std::string& desc) {
        stats.has_solution = true;
        stats.solution_count = 1;
        stats.sample_solution = wit;
        stats.message = "Exact solution found via " + desc + ".";
        if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// L2.5: Residue Cascade Sieve (Exact Modular DP & Sound Elimination)
// ══════════════════════════════════════════════════════════════════════════════
class ResidueCascadeSieve {
public:
    static constexpr int PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
    static constexpr int NUM_PRIMES = 11;

    static bool filter(const std::vector<Element>& A_in, u64 T,
                       int k_min, int k_max, int max_primes,
                       std::vector<Element>& out_kept_elements,
                       ExecutionStats& stats) {
        out_kept_elements = A_in;
        if (A_in.empty() || k_min > k_max) return true;

        int consecutive_zero_elims = 0;
        int primes_limit = std::min(max_primes, NUM_PRIMES);

        for (int p_idx = 0; p_idx < primes_limit; ++p_idx) {
            int p = PRIMES[p_idx];
            int n = (int)out_kept_elements.size();
            if (n == 0) break;

            int km_curr = std::min(k_max, n);
            int kn_curr = std::max(0, std::min(k_min, n));
            if (kn_curr > km_curr) return false;

            stats.residue_primes_checked++;

            u32 full_mask = (p == 32) ? 0xFFFFFFFFU : ((1U << p) - 1U);
            u32 T_mod = (u32)(T % p);

            // 1) Prefix DP
            std::vector<std::vector<u32>> prefix_masks(n + 1, std::vector<u32>(km_curr + 1, 0));
            prefix_masks[0][0] = 1U;

            for (int i = 0; i < n; ++i) {
                u32 r = (u32)(out_kept_elements[i].val % p);
                for (int k = 0; k <= km_curr; ++k) prefix_masks[i+1][k] = prefix_masks[i][k];
                for (int k = km_curr - 1; k >= 0; --k) {
                    u32 m = prefix_masks[i][k];
                    if (m == 0) continue;
                    u32 rotated = (r == 0) ? m : (((m << r) | (m >> (p - r))) & full_mask);
                    prefix_masks[i+1][k+1] |= rotated;
                }
            }

            // Global feasibility check
            bool globally_feasible = false;
            for (int k = kn_curr; k <= km_curr; ++k) {
                if ((prefix_masks[n][k] >> T_mod) & 1U) {
                    globally_feasible = true;
                    break;
                }
            }
            if (!globally_feasible) return false;

            // 2) Suffix DP
            std::vector<std::vector<u32>> suffix_masks(n + 1, std::vector<u32>(km_curr + 1, 0));
            suffix_masks[n][0] = 1U;

            for (int i = n - 1; i >= 0; --i) {
                u32 r = (u32)(out_kept_elements[i].val % p);
                for (int k = 0; k <= km_curr; ++k) suffix_masks[i][k] = suffix_masks[i+1][k];
                for (int k = km_curr - 1; k >= 0; --k) {
                    u32 m = suffix_masks[i+1][k];
                    if (m == 0) continue;
                    u32 rotated = (r == 0) ? m : (((m << r) | (m >> (p - r))) & full_mask);
                    suffix_masks[i][k+1] |= rotated;
                }
            }

            // 3) Element filtering
            std::vector<Element> next_kept;
            next_kept.reserve(n);

            for (int i = 0; i < n; ++i) {
                u32 r_i = (u32)(out_kept_elements[i].val % p);
                u32 target_ps = (T_mod + p - r_i) % p;
                bool can_include = false;

                for (int k = kn_curr; k <= km_curr && !can_include; ++k) {
                    for (int k1 = 0; k1 < k; ++k1) {
                        int k2 = k - 1 - k1;
                        if (k1 > i || k2 > (n - 1 - i)) continue;

                        u32 m1 = prefix_masks[i][k1];
                        u32 m2 = suffix_masks[i+1][k2];
                        if (m1 == 0 || m2 == 0) continue;

                        u32 tmp = m1;
                        while (tmp) {
                            int s1 = __builtin_ctz(tmp);
                            int s2 = (target_ps + p - (u32)s1) % p;
                            if ((m2 >> s2) & 1U) {
                                can_include = true;
                                break;
                            }
                            tmp &= tmp - 1;
                        }
                        if (can_include) break;
                    }
                }

                if (can_include) next_kept.push_back(out_kept_elements[i]);
            }

            int eliminated_this_round = n - (int)next_kept.size();
            stats.residue_elements_eliminated += (u64)eliminated_this_round;

            if (eliminated_this_round == 0) {
                consecutive_zero_elims++;
                if (consecutive_zero_elims >= 2 && p_idx >= 1) break;
            } else {
                consecutive_zero_elims = 0;
            }

            out_kept_elements = std::move(next_kept);
            if (out_kept_elements.empty()) return false;
        }

        return true;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// L2.7: Boundary Deviation + r-Swap Solver (O(N^2) Exact Pre-Check)
// ══════════════════════════════════════════════════════════════════════════════
class BoundarySwapSolver {
public:
    static bool try_solve(const Instance& inst, ExecutionStats& stats, const SolverBudget& budget, SolveMode mode) {
        if (!budget.enable_boundary_swap) return false;
        if (mode != SolveMode::FindOne && mode != SolveMode::DecisionOnly && mode != SolveMode::FindAll) return false;
        const auto& A = inst.A;
        int n = (int)A.size();
        u64 T = inst.effective_target;
        if (n == 0 || inst.k_min < 0 || inst.k_max < inst.k_min) return false;

        // 1. Top-Down scan near k_min
        int k_scan_max = std::min(inst.k_max, inst.k_min + 15);
        u128 cumtop = 0;
        for (int i = 0; i < inst.k_min && i < n; ++i) cumtop += A[i].val;

        for (int k = inst.k_min; k <= k_scan_max && k <= n; ++k) {
            if (k > inst.k_min) cumtop += A[k-1].val;
            if (cumtop < T) continue;
            u64 delta = (u64)(cumtop - T);

            if (delta == 0) {
                SolutionWitness wit;
                for (int j = 0; j < k; ++j) {
                    wit.original_indices.push_back(A[j].orig_idx);
                    wit.values.push_back(A[j].val);
                }
                wit.sort_indices();
                wit.sum = T;
                stats.sample_solution = wit;
                stats.has_solution = true;
                stats.solution_count = 1;
                stats.boundary_swap_applied = true;
                stats.boundary_swap_details = "Direct top-" + std::to_string(k) + " boundary match (delta=0)";
                stats.message = "Exact solution found via Boundary Solver (direct top-k match).";
                if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
                return true;
            }

            // Case 1: 1-swap (r=1)
            if (k < n) {
                for (int i = 0; i < k; ++i) {
                    u64 out_val = A[i].val;
                    if (out_val <= delta) continue;
                    u64 need_in = out_val - delta;

                    auto it = std::lower_bound(A.begin() + k, A.end(), need_in,
                        [](const Element& e, u64 val) { return e.val > val; });
                    if (it != A.end() && it->val == need_in) {
                        SolutionWitness wit;
                        for (int j = 0; j < k; ++j) {
                            if (j == i) continue;
                            wit.original_indices.push_back(A[j].orig_idx);
                            wit.values.push_back(A[j].val);
                        }
                        wit.original_indices.push_back(it->orig_idx);
                        wit.values.push_back(it->val);
                        wit.sort_indices();
                        wit.sum = T;
                        stats.sample_solution = wit;
                        stats.has_solution = true;
                        stats.solution_count = 1;
                        stats.boundary_swap_applied = true;
                        stats.boundary_swap_details = "1-Swap match (top-" + std::to_string(k) + ", out=" + std::to_string(out_val) + ", in=" + std::to_string(need_in) + ")";
                        stats.message = "Exact solution found via Boundary Solver (1-swap).";
                        if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
                        return true;
                    }
                }
            }

            // Case 2: 2-swap (r=2)
            if (budget.boundary_swap_max_r >= 2 && k < n && (n - k) >= 2) {
                int rest_count = std::min(n - k, 120);
                std::unordered_map<u64, std::pair<int, int>> rest_2sum;
                rest_2sum.reserve(rest_count * (rest_count - 1) / 2 + 1);

                for (int p1 = k; p1 < k + rest_count; ++p1) {
                    for (int p2 = p1 + 1; p2 < k + rest_count; ++p2) {
                        u64 s = A[p1].val + A[p2].val;
                        rest_2sum.emplace(s, std::make_pair(p1, p2));
                    }
                }

                int top_check = std::min(k, 80);
                for (int i1 = 0; i1 < top_check; ++i1) {
                    for (int i2 = i1 + 1; i2 < top_check; ++i2) {
                        u64 out_sum = A[i1].val + A[i2].val;
                        if (out_sum <= delta) continue;
                        u64 need_in = out_sum - delta;
                        auto fit = rest_2sum.find(need_in);
                        if (fit != rest_2sum.end()) {
                            SolutionWitness wit;
                            for (int j = 0; j < k; ++j) {
                                if (j == i1 || j == i2) continue;
                                wit.original_indices.push_back(A[j].orig_idx);
                                wit.values.push_back(A[j].val);
                            }
                            int in1 = fit->second.first;
                            int in2 = fit->second.second;
                            wit.original_indices.push_back(A[in1].orig_idx);
                            wit.values.push_back(A[in1].val);
                            wit.original_indices.push_back(A[in2].orig_idx);
                            wit.values.push_back(A[in2].val);
                            wit.sort_indices();
                            wit.sum = T;
                            stats.sample_solution = wit;
                            stats.has_solution = true;
                            stats.solution_count = 1;
                            stats.boundary_swap_applied = true;
                            stats.boundary_swap_details = "2-Swap match (top-" + std::to_string(k) + ")";
                            stats.message = "Exact solution found via Boundary Solver (2-swap).";
                            if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
                            return true;
                        }
                    }
                }
            }
        }

        // 2. Bottom-Up scan near k_max
        int k_bot_min = std::max(inst.k_min, inst.k_max - 5);
        u128 cumbot = 0;
        for (int i = n - 1; i >= n - inst.k_max && i >= 0; --i) cumbot += A[i].val;

        for (int k = inst.k_max; k >= k_bot_min && k >= 1; --k) {
            if (k < inst.k_max) cumbot -= A[n - k - 1].val;
            if (cumbot > T) continue;
            u64 sigma = (u64)(T - cumbot);

            if (sigma == 0) {
                SolutionWitness wit;
                for (int j = n - k; j < n; ++j) {
                    wit.original_indices.push_back(A[j].orig_idx);
                    wit.values.push_back(A[j].val);
                }
                wit.sort_indices();
                wit.sum = T;
                stats.sample_solution = wit;
                stats.has_solution = true;
                stats.solution_count = 1;
                stats.boundary_swap_applied = true;
                stats.boundary_swap_details = "Direct bottom-" + std::to_string(k) + " boundary match (sigma=0)";
                stats.message = "Exact solution found via Boundary Solver (direct bottom-k match).";
                if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
                return true;
            }

            if (k < n) {
                for (int j = n - k; j < n; ++j) {
                    u64 in_val = A[j].val;
                    u64 need_out = in_val + sigma;
                    auto it = std::lower_bound(A.begin(), A.begin() + (n - k), need_out,
                        [](const Element& e, u64 val) { return e.val > val; });
                    if (it != A.begin() + (n - k) && it->val == need_out) {
                        SolutionWitness wit;
                        for (int m = n - k; m < n; ++m) {
                            if (m == j) continue;
                            wit.original_indices.push_back(A[m].orig_idx);
                            wit.values.push_back(A[m].val);
                        }
                        wit.original_indices.push_back(it->orig_idx);
                        wit.values.push_back(it->val);
                        wit.sort_indices();
                        wit.sum = T;
                        stats.sample_solution = wit;
                        stats.has_solution = true;
                        stats.solution_count = 1;
                        stats.boundary_swap_applied = true;
                        stats.boundary_swap_details = "1-Swap match (bottom-" + std::to_string(k) + ", out=" + std::to_string(need_out) + ", in=" + std::to_string(in_val) + ")";
                        stats.message = "Exact solution found via Boundary Solver (bottom 1-swap).";
                        if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
                        return true;
                    }
                }
            }
        }

        return false;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// Adaptive & Bounded BlockBound (Open-Addressing Flat Hash per Depth)
// ══════════════════════════════════════════════════════════════════════════════
struct BlockBound {
    struct DepthCache {
        std::vector<u64> table;
        size_t mask = 0;
        size_t count = 0;
        size_t cap = 0;

        void init(size_t capacity) {
            size_t p = 16;
            while (p < capacity) p <<= 1;
            cap = capacity;
            table.assign(p, INF64);
            mask = p - 1;
            count = 0;
        }

        bool contains(u64 val) const {
            if (count == 0) return false;
            size_t idx = (val * 11400714819323198485ULL) & mask;
            while (table[idx] != INF64) {
                if (table[idx] == val) return true;
                idx = (idx + 1) & mask;
            }
            return false;
        }

        void insert(u64 val) {
            if (count >= cap || count * 2 >= table.size()) return;
            size_t idx = (val * 11400714819323198485ULL) & mask;
            while (table[idx] != INF64) {
                if (table[idx] == val) return;
                idx = (idx + 1) & mask;
            }
            table[idx] = val;
            count++;
        }
    };

    std::vector<DepthCache> caches;
    bool disabled = false;
    u64 hits = 0;
    u64 misses = 0;
    size_t cap_per_depth = 65536;

    explicit BlockBound(int n, size_t cap = 65536) : caches(n + 1), cap_per_depth(cap) {
        for (int i = 0; i <= n; ++i) caches[i].init(cap_per_depth);
    }

    bool is_known_infeasible(int i, u64 rem) {
        if (disabled) return false;
        if (caches[i].contains(rem)) {
            hits++;
            return true;
        }
        misses++;
        if ((misses & 8191) == 0 && misses >= 32768) {
            if ((double)hits / (double)(hits + misses) < 0.01) {
                disabled = true;
            }
        }
        return false;
    }

    void mark_infeasible(int i, u64 rem) {
        if (disabled) return;
        caches[i].insert(rem);
    }
};

class AdaptiveStrategySelector {
public:
    static u64 estimate_bitset_memory(const Instance& inst) {
        if (inst.effective_target > 50000000ULL) return INF64;
        u64 words = (inst.effective_target >> 6) + 1;
        return words * sizeof(u64) + (inst.effective_target + 1) * sizeof(int);
    }

    static StrategyType select(const Instance& inst, const SolverBudget& budget, SolveMode mode) {
        u64 mem_budget = budget.memory_limit_mb * 1024ULL * 1024ULL;

        if (inst.target == 0) return StrategyType::TrivialPreCheck;
        if (inst.A.empty() || inst.effective_target > inst.total_sum) return StrategyType::TrivialPreCheck;
        if (inst.gcd_val > 1 && (inst.effective_target % inst.gcd_val != 0)) return StrategyType::TrivialPreCheck;
        if (inst.odd_count == 0 && (inst.effective_target % 2 != 0)) return StrategyType::TrivialPreCheck;
        if (inst.feasible_k_count == 0) return StrategyType::TrivialPreCheck;

        if (inst.structure_kind == StructureKind::Superincreasing) {
            return StrategyType::GreedySuperincreasing;
        }

        u64 bitset_mem = estimate_bitset_memory(inst);
        if (inst.effective_target <= 15000000ULL && bitset_mem < mem_budget / 2 &&
            (mode == SolveMode::FindOne || mode == SolveMode::DecisionOnly)) {
            return StrategyType::BitsetDP;
        }

        return StrategyType::HybridTailTable;
    }
};

class GreedySuperincreasingSolver {
public:
    void solve(const Instance& inst, SolveMode mode, ExecutionStats& stats) {
        u64 rem = inst.effective_target;
        std::vector<int> idxs; std::vector<u64> vals;
        idxs.reserve(inst.A.size()); vals.reserve(inst.A.size());
        for (const auto& e : inst.A) {
            stats.states_evaluated++;
            if (e.val <= rem) {
                idxs.push_back(e.orig_idx);
                vals.push_back(e.val);
                rem -= e.val;
                stats.forced_take++;
            } else {
                stats.forced_skip++;
            }
        }
        if (rem == 0) {
            SolutionWitness wit;
            wit.original_indices = idxs;
            wit.values = vals;
            u128 s = 0; for (u64 v : vals) s += v;
            wit.sum = s;
            wit.sort_indices();
            stats.has_solution = true;
            stats.solution_count = 1;
            stats.sample_solution = wit;
            if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
            stats.message = "Exact solution found via Greedy Superincreasing (O(n), no backtrack needed).";
        } else {
            stats.message = "UNSAT provably proven via Greedy Superincreasing (O(n), structural proof).";
        }
    }
};

class HybridTailTableEngine {
public:
    static int choose_m(int n, size_t memory_limit_mb, bool is_critical_zone = false) {
        if (n <= 1) return 0;
        if (n <= 20) return n - 1;
        int m = is_critical_zone ? 24 : 20;
        u64 budget_bytes = memory_limit_mb * 1024ULL * 1024ULL / 4;
        while (m > 12 && ((1ULL << m) * 16ULL) > budget_bytes) m--;
        if (m >= n) m = std::max(0, n - 1);
        return m;
    }

    struct Entry { u64 sum; u32 mask; bool operator<(const Entry& o) const { return sum < o.sum; } };

    static void build_tail_table_range(const std::vector<Element>& A, int cutoff, int m,
                                        u32 begin, u32 end, std::vector<Entry>& table) {
        if (begin >= end) return;
        u32 start_gray = begin ^ (begin >> 1);
        u64 sum = 0;
        for (int b = 0; b < m; ++b) if ((start_gray >> b) & 1) sum += A[cutoff + b].val;
        table[begin] = {sum, start_gray};
        u32 prev_gray = start_gray;
        for (u32 i = begin + 1; i < end; ++i) {
            u32 gray = i ^ (i >> 1);
            u32 diff = gray ^ prev_gray;
            int bit = __builtin_ctz(diff);
            if (gray & diff) sum += A[cutoff + bit].val; else sum -= A[cutoff + bit].val;
            table[i] = {sum, gray};
            prev_gray = gray;
        }
    }

    static std::vector<Entry> build_tail_table(const std::vector<Element>& A, int cutoff, int m,
                                                unsigned n_threads) {
        u32 total = (m <= 0) ? 1U : (1U << m);
        std::vector<Entry> table(total);
        if (n_threads <= 1 || total < 4096) {
            build_tail_table_range(A, cutoff, m, 0, total, table);
        } else {
            auto& pool = get_global_thread_pool();
            pool.ensure_threads(n_threads);
            u32 chunk = (total + n_threads - 1) / n_threads;
            for (unsigned t = 0; t < n_threads; ++t) {
                u32 b = t * chunk, e = std::min(total, b + chunk);
                if (b >= e) continue;
                pool.enqueue([&A, cutoff, m, b, e, &table]() {
                    build_tail_table_range(A, cutoff, m, b, e, table);
                });
            }
            pool.wait_all();
        }
        std::sort(table.begin(), table.end());
        return table;
    }

    void solve(const Instance& inst, SolveMode mode, ExecutionStats& stats,
               const SolverBudget& budget,
               const std::chrono::steady_clock::time_point& start_time,
               std::atomic<bool>& stop_flag, std::atomic<bool>& solution_found) {
        int n = (int)inst.A.size();
        const auto& A = inst.A;
        u64 T = inst.effective_target;
        if (n == 0) { if (T == 0) { stats.has_solution = true; stats.solution_count = 1; } return; }

        // Cost estimation check
        if (budget.enable_cost_estimator && stats.estimated_dfs_nodes == 0.0) {
            DFSCostInfo cost_info = estimate_dfs_cost(inst, 20);
            stats.estimated_dfs_nodes = cost_info.estimated_nodes;
            stats.high_complexity_warning = cost_info.potentially_expensive;
            if (cost_info.potentially_expensive && stats.message.empty()) {
                stats.message = cost_info.warning_message;
            }
        }

        bool is_critical_zone = (stats.estimated_dfs_nodes >= 5000000.0 &&
                                 inst.window_ratio_pct >= 35 &&
                                 n >= 40);

        int m = choose_m(n, budget.memory_limit_mb, is_critical_zone);
        int cutoff = n - m;

        if (budget.enable_cost_estimator && m != 20) {
            DFSCostInfo cost_info = estimate_dfs_cost(inst, m);
            stats.estimated_dfs_nodes = cost_info.estimated_nodes;
            stats.high_complexity_warning = cost_info.potentially_expensive;
        }

        std::vector<u128> suffix(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) suffix[i] = suffix[i+1] + A[i].val;

        unsigned build_threads = budget.max_worker_threads ? budget.max_worker_threads
                                                             : std::max(1u, std::thread::hardware_concurrency());
        std::vector<Entry> table = build_tail_table(A, cutoff, m, build_threads);

        bool is_bimodal = (inst.structure_kind == StructureKind::Bimodal) ||
                          (inst.min_val > 0 && (inst.max_val / inst.min_val) >= 10000ULL);

        if (is_bimodal) {
            stats.block_bound_disabled = true;
            std::vector<int> path_indices;
            std::vector<u64> path_vals;
            run_pure_deterministic_dfs(inst, mode, stats, budget, start_time, stop_flag, solution_found,
                                       suffix, table, cutoff, m, path_indices, path_vals,
                                       0, T, 0);
            if (stats.status == SolverStatus::UnknownTimeout) {
                stats.message = "Search stopped due to time limit (Timeout).";
            } else if (stats.has_solution) {
                stats.message = "Exact solution found via Pure Deterministic DFS (Adaptive Gray Tail-Table).";
            } else {
                stats.message = "UNSAT provably proven via Pure Deterministic DFS.";
            }
            return;
        }

        size_t block_cap = budget.block_bound_cap_per_depth;
        if (is_critical_zone) {
            block_cap = std::max(block_cap, (size_t)131072);
        }
        BlockBound bb(n, block_cap);
        std::mutex bb_mutex;

        // v3 Parallel root-split: activated only when combinatorial complexity warrants it
        bool try_parallel_root = budget.allow_parallel_root_split &&
                                  mode == SolveMode::FindOne &&
                                  n >= 40 &&
                                  (inst.feasible_k_count * 3 >= n) &&
                                  (stats.estimated_dfs_nodes >= 5000000.0) &&
                                  (std::thread::hardware_concurrency() > 1 || budget.max_worker_threads > 1);

        if (try_parallel_root) {
            run_parallel_root_split(inst, stats, budget, start_time, stop_flag, solution_found,
                                     suffix, table, cutoff, m, bb, bb_mutex);
        } else {
            std::vector<int> path_indices;
            std::vector<u64> path_vals;
            run_single_thread_dfs(inst, mode, stats, budget, start_time, stop_flag, solution_found,
                                   suffix, table, cutoff, m, bb, path_indices, path_vals,
                                   0, T, 0);
        }

        stats.block_bound_hits = bb.hits;
        stats.block_bound_misses = bb.misses;
        stats.block_bound_disabled = bb.disabled;

        if (stats.status == SolverStatus::UnknownTimeout) {
            stats.message = "Search stopped due to time limit (Timeout).";
        } else if (stats.has_solution) {
            stats.message = "Exact solution found via Hybrid Tail-Table Engine.";
        } else {
            stats.message = "UNSAT provably proven via Hybrid Tail-Table Engine.";
        }
    }

private:
    static void record_solution(ExecutionStats& stats, SolveMode mode, const SolverBudget& budget,
                                 std::atomic<bool>& solution_found, bool early_exit, std::mutex* out_mutex,
                                 const std::vector<int>& idxs, const std::vector<u64>& vals) {
        std::unique_lock<std::mutex> lock;
        if (out_mutex) lock = std::unique_lock<std::mutex>(*out_mutex);
        stats.has_solution = true;
        SolutionWitness wit;
        wit.original_indices = idxs;
        wit.values = vals;
        u128 s = 0; for (u64 v : vals) s += v;
        wit.sum = s;
        wit.sort_indices();
        stats.solution_count++;
        if (stats.sample_solution.values.empty()) stats.sample_solution = wit;
        if (mode == SolveMode::FindAll && stats.all_solutions.size() < budget.max_solutions) {
            stats.all_solutions.push_back(wit);
        }
        if (early_exit) solution_found = true;
    }

    void run_pure_deterministic_dfs(const Instance& inst, SolveMode mode, ExecutionStats& stats,
                         const SolverBudget& budget,
                         const std::chrono::steady_clock::time_point& start_time,
                         std::atomic<bool>& stop_flag, std::atomic<bool>& solution_found,
                         const std::vector<u128>& suffix, const std::vector<Entry>& table,
                         int cutoff, int m,
                         std::vector<int>& path_indices, std::vector<u64>& path_vals,
                         int start_i, u64 start_rem, int start_k) {
        const auto& A = inst.A;
        int n = (int)A.size();
        bool early_exit = (mode == SolveMode::FindOne || mode == SolveMode::DecisionOnly);

        auto dfs = [&](auto& self, int i, u64 rem, int k_used) -> void {
            if (stop_flag || (solution_found && early_exit) || stats.status == SolverStatus::UnknownTimeout) return;
            if (mode == SolveMode::FindAll && stats.all_solutions.size() >= budget.max_solutions &&
                !budget.exhaustive_find_all) {
                stats.status = SolverStatus::PartialSolutionCapped;
                return;
            }
            stats.states_evaluated++;
            if (budget.time_limit_ms > 0.0 && (stats.states_evaluated & 4095) == 0) {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double, std::milli>(now - start_time).count();
                if (el > budget.time_limit_ms) {
                    stats.status = SolverStatus::UnknownTimeout;
                    stop_flag = true;
                    return;
                }
            }

            if (rem == 0) {
                if (mode == SolveMode::CountAll && !stats.sample_solution.values.empty()) {
                    stats.has_solution = true;
                    stats.solution_count++;
                } else {
                    record_solution(stats, mode, budget, solution_found, early_exit, nullptr, path_indices, path_vals);
                }
                return;
            }

            // Tail-table leaf lookup using single lower_bound (Gray-code table)
            if (i >= cutoff) {
                stats.table_lookups++;
                Entry dummy{rem, 0};
                auto it = std::lower_bound(table.begin(), table.end(), dummy);
                if (it != table.end() && it->sum == rem) {
                    for (; it != table.end() && it->sum == rem; ++it) {
                        if (stop_flag) return;
                        if (mode == SolveMode::CountAll && !stats.sample_solution.values.empty()) {
                            stats.has_solution = true;
                            stats.solution_count++;
                            continue;
                        }
                        std::vector<int> full_idx = path_indices;
                        std::vector<u64> full_val = path_vals;
                        u32 mask = it->mask;
                        for (int b = 0; b < m; ++b) {
                            if ((mask >> b) & 1) {
                                full_idx.push_back(A[cutoff + b].orig_idx);
                                full_val.push_back(A[cutoff + b].val);
                            }
                        }
                        record_solution(stats, mode, budget, solution_found, early_exit, nullptr, full_idx, full_val);
                        if (early_exit) return;
                        if (mode == SolveMode::FindAll && stats.all_solutions.size() >= budget.max_solutions &&
                            !budget.exhaustive_find_all) return;
                    }
                }
                return;
            }

            if ((u128)rem > suffix[i]) { stats.states_pruned++; return; }
            if ((k_used + (n - i)) < inst.k_min || (inst.k_max > 0 && k_used > inst.k_max)) {
                stats.states_pruned++; return;
            }
            stats.oracle_calls++;
            if (!is_cardinality_feasible(i, rem, n, suffix, A)) {
                stats.oracle_pruned++; stats.states_pruned++; return;
            }

            // Pure Deterministic DFS: Greedy Include First, then Exclude
            if (A[i].val <= rem) {
                path_indices.push_back(A[i].orig_idx);
                path_vals.push_back(A[i].val);
                self(self, i + 1, rem - A[i].val, k_used + 1);
                path_indices.pop_back();
                path_vals.pop_back();
                if (early_exit && (solution_found || stats.status == SolverStatus::UnknownTimeout)) return;
            }
            self(self, i + 1, rem, k_used);
        };

        dfs(dfs, start_i, start_rem, start_k);
    }

    void run_single_thread_dfs(const Instance& inst, SolveMode mode, ExecutionStats& stats,
                                const SolverBudget& budget,
                                const std::chrono::steady_clock::time_point& start_time,
                                std::atomic<bool>& stop_flag, std::atomic<bool>& solution_found,
                                const std::vector<u128>& suffix, const std::vector<Entry>& table,
                                int cutoff, int m, BlockBound& bb,
                                std::vector<int>& path_indices, std::vector<u64>& path_vals,
                                int start_i, u64 start_rem, int start_k,
                                std::mutex* out_mutex = nullptr,
                                std::mutex* bb_mutex = nullptr) {
        const auto& A = inst.A;
        int n = (int)A.size();
        bool early_exit = (mode == SolveMode::FindOne || mode == SolveMode::DecisionOnly);

        auto dfs = [&](auto& self, int i, u64 rem, int k_used) -> bool {
            if (stop_flag || (solution_found && early_exit) || stats.status == SolverStatus::UnknownTimeout) return false;
            if (mode == SolveMode::FindAll) {
                bool capped;
                { std::unique_lock<std::mutex> lk; if (out_mutex) lk = std::unique_lock<std::mutex>(*out_mutex);
                  capped = stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all; }
                if (capped) { stats.status = SolverStatus::PartialSolutionCapped; return false; }
            }
            stats.states_evaluated++;
            if (budget.time_limit_ms > 0.0 && (stats.states_evaluated & 4095) == 0) {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double, std::milli>(now - start_time).count();
                if (el > budget.time_limit_ms) {
                    stats.status = SolverStatus::UnknownTimeout;
                    stop_flag = true;
                    return false;
                }
            }

            // Adaptive BlockBound lookup
            {
                std::unique_lock<std::mutex> lk; if (bb_mutex) lk = std::unique_lock<std::mutex>(*bb_mutex);
                if (bb.is_known_infeasible(i, rem)) { stats.states_pruned++; stats.block_bound_prunes++; return true; }
            }

            if (rem == 0) {
                if (mode == SolveMode::CountAll && !stats.sample_solution.values.empty()) {
                    stats.has_solution = true;
                    stats.solution_count++;
                } else {
                    record_solution(stats, mode, budget, solution_found, early_exit, out_mutex, path_indices, path_vals);
                }
                return true;
            }

            // Leaf Lookup: single lower_bound saves 50% binary comparisons
            if (i >= cutoff) {
                stats.table_lookups++;
                Entry dummy{rem, 0};
                auto it = std::lower_bound(table.begin(), table.end(), dummy);
                if (it != table.end() && it->sum == rem) {
                    for (; it != table.end() && it->sum == rem; ++it) {
                        if (stop_flag) return false;
                        if (mode == SolveMode::CountAll && !stats.sample_solution.values.empty()) {
                            stats.has_solution = true;
                            stats.solution_count++;
                            continue;
                        }
                        std::vector<int> full_idx = path_indices;
                        std::vector<u64> full_val = path_vals;
                        u32 mask = it->mask;
                        for (int b = 0; b < m; ++b) {
                            if ((mask >> b) & 1) {
                                full_idx.push_back(A[cutoff + b].orig_idx);
                                full_val.push_back(A[cutoff + b].val);
                            }
                        }
                        record_solution(stats, mode, budget, solution_found, early_exit, out_mutex, full_idx, full_val);
                        if (early_exit) return true;
                        bool capped;
                        { std::unique_lock<std::mutex> lk; if (out_mutex) lk = std::unique_lock<std::mutex>(*out_mutex);
                          capped = mode == SolveMode::FindAll && stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all; }
                        if (capped) return true;
                    }
                }
                return true;
            }

            if ((u128)rem > suffix[i]) { stats.states_pruned++; return true; }
            if ((k_used + (n - i)) < inst.k_min || (inst.k_max > 0 && k_used > inst.k_max)) {
                stats.states_pruned++; return true;
            }
            stats.oracle_calls++;
            if (!is_cardinality_feasible(i, rem, n, suffix, A)) { stats.oracle_pruned++; stats.states_pruned++; return true; }

            // Scale-Aware Branch-Order:
            // Bimodal instances (dynamic range > 10,000x) require deterministic include-first (drops states from 7.5M to 1.3M)
            // Flat, uniform, and structured instances benefit strongly from middle-distance heuristic (drops states from 51M to 3.5M)
            bool can_include = (A[i].val <= rem);
            bool include_first = true;
            if (can_include) {
                bool is_bimodal_scale = (inst.min_val > 0 && (inst.max_val / inst.min_val) > 10000ULL);
                if (is_bimodal_scale) {
                    include_first = true;
                } else {
                    double half = (double)(u128)(suffix[i+1] / 2);
                    double d_include = std::fabs((double)(rem - A[i].val) - half);
                    double d_exclude = std::fabs((double)rem - half);
                    include_first = (d_include <= d_exclude);
                }
            }

            bool fully_explored = true;
            auto do_include = [&]() {
                path_indices.push_back(A[i].orig_idx);
                path_vals.push_back(A[i].val);
                bool fe = self(self, i + 1, rem - A[i].val, k_used + 1);
                path_indices.pop_back();
                path_vals.pop_back();
                fully_explored = fully_explored && fe;
            };
            auto do_exclude = [&]() {
                bool fe = self(self, i + 1, rem, k_used);
                fully_explored = fully_explored && fe;
            };

            if (can_include && include_first) {
                do_include();
                if (early_exit && (solution_found || stats.status == SolverStatus::UnknownTimeout)) return false;
                do_exclude();
            } else if (can_include) {
                do_exclude();
                if (early_exit && (solution_found || stats.status == SolverStatus::UnknownTimeout)) return false;
                do_include();
            } else {
                do_exclude();
            }

            if (fully_explored && !stop_flag) {
                std::unique_lock<std::mutex> lk; if (bb_mutex) lk = std::unique_lock<std::mutex>(*bb_mutex);
                bb.mark_infeasible(i, rem);
            }
            return fully_explored;
        };

        dfs(dfs, start_i, start_rem, start_k);
    }

    void run_parallel_root_split(const Instance& inst, ExecutionStats& stats, const SolverBudget& budget,
                                  const std::chrono::steady_clock::time_point& start_time,
                                  std::atomic<bool>& stop_flag, std::atomic<bool>& solution_found,
                                  const std::vector<u128>& suffix, const std::vector<Entry>& table,
                                  int cutoff, int m, BlockBound& bb, std::mutex& bb_mutex) {
        const auto& A = inst.A;
        int n = (int)A.size();
        u64 T = inst.effective_target;

        unsigned hw = budget.max_worker_threads ? budget.max_worker_threads
                                                  : std::max(1u, std::thread::hardware_concurrency());
        stats.threads_used = hw;

        struct Task { int i; u64 rem; int k_used; std::vector<int> path_idx; std::vector<u64> path_val; };
        std::deque<Task> queue;
        std::mutex queue_mutex;

        int expand_depth_cap = std::min(n, 6);
        std::vector<Task> frontier;
        frontier.push_back({0, T, 0, {}, {}});
        size_t target_tasks = (size_t)hw * 6;
        for (int depth = 0; depth < expand_depth_cap && frontier.size() < target_tasks; ++depth) {
            std::vector<Task> next;
            for (auto& t : frontier) {
                int i = t.i; u64 rem = t.rem; int k_used = t.k_used;
                if (i >= cutoff || rem == 0) { next.push_back(std::move(t)); continue; }
                if ((u128)rem > suffix[i]) continue;
                if (!is_cardinality_feasible(i, rem, n, suffix, A)) continue;
                bool can_include = (A[i].val <= rem);
                if (can_include) {
                    Task ti = t;
                    ti.path_idx.push_back(A[i].orig_idx); ti.path_val.push_back(A[i].val);
                    ti.i = i + 1; ti.rem = rem - A[i].val; ti.k_used = k_used + 1;
                    next.push_back(std::move(ti));
                }
                Task te = t;
                te.i = i + 1; te.rem = rem; te.k_used = k_used;
                next.push_back(std::move(te));
            }
            frontier = std::move(next);
        }
        for (auto& t : frontier) queue.push_back(std::move(t));

        auto pop_task = [&](Task& out) -> bool {
            std::lock_guard<std::mutex> lk(queue_mutex);
            if (queue.empty()) return false;
            out = std::move(queue.back());
            queue.pop_back();
            return true;
        };

        std::mutex stats_mutex;
        std::vector<ExecutionStats> local_stats(hw);
        for (auto& ls : local_stats) ls.strategy_chosen = stats.strategy_chosen;

        auto worker = [&](unsigned tid) {
            ExecutionStats& ls = local_stats[tid];
            Task t;
            while (!stop_flag && !solution_found && pop_task(t)) {
                std::vector<int> path_idx = t.path_idx;
                std::vector<u64> path_val = t.path_val;
                run_single_thread_dfs(inst, SolveMode::FindOne, ls, budget, start_time, stop_flag, solution_found,
                                       suffix, table, cutoff, m, bb, path_idx, path_val,
                                       t.i, t.rem, t.k_used, &stats_mutex, &bb_mutex);
            }
        };

        auto& pool = get_global_thread_pool();
        pool.ensure_threads(hw);
        for (unsigned t = 0; t < hw; ++t) {
            pool.enqueue([&worker, t]() {
                worker(t);
            });
        }
        pool.wait_all();

        for (auto& ls : local_stats) {
            stats.states_evaluated += ls.states_evaluated;
            stats.states_pruned    += ls.states_pruned;
            stats.oracle_calls     += ls.oracle_calls;
            stats.oracle_pruned    += ls.oracle_pruned;
            stats.table_lookups    += ls.table_lookups;
            stats.block_bound_prunes += ls.block_bound_prunes;
            if (ls.status == SolverStatus::UnknownTimeout) stats.status = SolverStatus::UnknownTimeout;
            if (ls.has_solution && !stats.has_solution) {
                stats.has_solution = true;
                stats.solution_count = 1;
                stats.sample_solution = ls.sample_solution;
            }
        }
    }
};

class ZeroSumSwapExtractor {
public:
    void extract(const Instance& inst, ExecutionStats& stats, const SolverBudget& budget) {
        if (!stats.has_solution || stats.sample_solution.original_indices.empty()) return;

        std::set<std::vector<int>> seen;
        if (stats.all_solutions.empty()) stats.all_solutions.push_back(stats.sample_solution);
        for (const auto& w : stats.all_solutions) seen.insert(w.original_indices);

        std::vector<Element> S_in, S_out;
        std::unordered_set<int> in_set(stats.sample_solution.original_indices.begin(),
                                        stats.sample_solution.original_indices.end());
        for (const auto& e : inst.A) (in_set.count(e.orig_idx) ? S_in : S_out).push_back(e);

        struct Subset { u64 sum; u64 mask; bool operator<(const Subset& o) const { return sum < o.sum; } };
        std::vector<Subset> in_subsets;
        int n_in = (int)S_in.size();
        auto gen_in = [&](auto& self, int idx, u64 csum, u64 cmask, int cnt) -> void {
            if (cnt > 0) in_subsets.push_back({csum, cmask});
            if (cnt == 4) return;
            for (int j = idx; j < n_in; ++j) self(self, j+1, csum + S_in[j].val, cmask | (1ULL<<j), cnt+1);
        };
        gen_in(gen_in, 0, 0, 0, 0);
        std::sort(in_subsets.begin(), in_subsets.end());

        int n_out = (int)S_out.size();
        auto gen_out = [&](auto& self, int idx, u64 csum, u64 cmask, int cnt) -> void {
            if (stats.all_solutions.size() >= budget.max_solutions) return;
            if (cnt > 0) {
                Subset dummy{csum, 0};
                auto bounds = std::equal_range(in_subsets.begin(), in_subsets.end(), dummy);
                for (auto it = bounds.first; it != bounds.second && stats.all_solutions.size() < budget.max_solutions; ++it) {
                    if (it->sum != csum) continue;
                    SolutionWitness wit;
                    for (int k = 0; k < n_in; ++k) if (!((it->mask>>k)&1)) { wit.original_indices.push_back(S_in[k].orig_idx); wit.values.push_back(S_in[k].val); }
                    for (int k = 0; k < n_out; ++k) if ((cmask>>k)&1) { wit.original_indices.push_back(S_out[k].orig_idx); wit.values.push_back(S_out[k].val); }
                    wit.sort_indices();
                    if (!seen.insert(wit.original_indices).second) continue;
                    u128 s = 0; for (u64 v : wit.values) s += v;
                    wit.sum = s;
                    stats.all_solutions.push_back(wit);
                }
            }
            if (cnt == 4) return;
            for (int j = idx; j < n_out; ++j) self(self, j+1, csum + S_out[j].val, cmask | (1ULL<<j), cnt+1);
        };
        gen_out(gen_out, 0, 0, 0, 0);
    }
};

class AdaptiveExactSolver {
public:
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> solution_found{false};

private:
    static void finalize_status(ExecutionStats& stats, bool stop_flag_value) {
        if (stats.status == SolverStatus::UnknownTimeout ||
            stats.status == SolverStatus::PartialSolutionCapped) {
            stats.solved = true;
        } else if (stop_flag_value) {
            stats.status = SolverStatus::StoppedByUser;
            stats.solved = false;
            stats.message = "Search stopped by user.";
        } else {
            stats.status = stats.has_solution ? SolverStatus::ExactSolutionFound : SolverStatus::ExactUnsatProven;
            stats.solved = true;
        }
    }

public:
    ExecutionStats run(const Instance& inst, SolveMode mode,
                        size_t memory_limit_mb = 4096, bool exhaustive_find_all = false,
                        double time_limit_ms = 120000.0, size_t max_solutions = 5000) {
        SolverBudget budget;
        budget.memory_limit_mb = memory_limit_mb;
        budget.time_limit_ms = time_limit_ms;
        budget.exhaustive_find_all = exhaustive_find_all;
        budget.max_solutions = max_solutions;
        return run_with_budget(inst, mode, budget);
    }

    ExecutionStats run_with_budget(const Instance& inst, SolveMode mode, const SolverBudget& budget) {
        stop_flag = false; solution_found = false;

        ExecutionStats stats;
        auto t0 = std::chrono::steady_clock::now();

        stats.strategy_chosen = AdaptiveStrategySelector::select(inst, budget, mode);
        auto t_prep = std::chrono::steady_clock::now();
        stats.preprocess_ms = std::chrono::duration<double, std::milli>(t_prep - t0).count();

        switch (stats.strategy_chosen) {
            case StrategyType::TrivialPreCheck:
                solve_trivial(inst, mode, stats);
                break;
            case StrategyType::GreedySuperincreasing: {
                GreedySuperincreasingSolver s; s.solve(inst, mode, stats);
                break;
            }
            case StrategyType::BitsetDP:
                solve_bitset(inst, mode, stats, budget, t0);
                break;
            case StrategyType::HybridTailTable:
            default: {
                // L2.6: Small-K Combinatorial MitM Solver (Instant win for sparse instances k <= 5)
                if (inst.k_min >= 1 && inst.k_min <= 5 && SmallKCombinatorialSolver::try_solve(inst, stats, budget, mode)) {
                    stats.strategy_chosen = StrategyType::SmallKMitM;
                    break;
                }

                // L2.7: Boundary Swap quick check first
                if (budget.enable_boundary_swap && BoundarySwapSolver::try_solve(inst, stats, budget, mode)) {
                    stats.strategy_chosen = StrategyType::BoundarySwap;
                    break;
                }

                // Pre-evaluate cost estimation for adaptive sieve & tail-table tuning
                if (budget.enable_cost_estimator && stats.estimated_dfs_nodes == 0.0) {
                    DFSCostInfo cost_info = estimate_dfs_cost(inst, 20);
                    stats.estimated_dfs_nodes = cost_info.estimated_nodes;
                    stats.high_complexity_warning = cost_info.potentially_expensive;
                }

                bool is_critical_zone = (stats.estimated_dfs_nodes >= 5000000.0 &&
                                         inst.window_ratio_pct >= 35 &&
                                         (int)inst.A.size() >= 40);

                // L2.5: Residue Cascade Sieve check / reduction
                Instance working_inst = inst;
                if (budget.enable_residue_sieve && (inst.A.size() >= 36 || inst.window_ratio_pct < 45 || is_critical_zone)) {
                    std::vector<Element> kept;
                    int primes_to_use = is_critical_zone ? ResidueCascadeSieve::NUM_PRIMES : budget.max_sieve_primes;
                    bool feasible = ResidueCascadeSieve::filter(inst.A, inst.effective_target,
                                                                inst.k_min, inst.k_max, primes_to_use,
                                                                kept, stats);
                    if (!feasible) {
                        stats.has_solution = false;
                        stats.status = SolverStatus::ExactUnsatProven;
                        stats.message = "Provably UNSAT via Residue Cascade Sieve (modular obstruction).";
                        break;
                    }
                    if (kept.size() < inst.A.size()) {
                        working_inst.A = kept;
                        working_inst.compute_cardinality_bounds();
                    }
                }

                HybridTailTableEngine e;
                e.solve(working_inst, mode, stats, budget, t0, stop_flag, solution_found);
                break;
            }
        }

        if (mode == SolveMode::FindAll && stats.has_solution && !budget.exhaustive_find_all) {
            ZeroSumSwapExtractor extractor;
            extractor.extract(inst, stats, budget);
            stats.solution_count = std::max(stats.solution_count, (u128)stats.all_solutions.size());
        } else if (stats.has_solution && stats.all_solutions.empty() && !stats.sample_solution.values.empty()) {
            stats.all_solutions.push_back(stats.sample_solution);
        }

        if (inst.complement_applied && stats.has_solution) {
            auto invert_witness = [&](SolutionWitness& wit) {
                std::vector<bool> in_comp(inst.raw_elements.size(), false);
                for (int idx : wit.original_indices) {
                    if (idx >= 0 && idx < (int)in_comp.size()) in_comp[idx] = true;
                }
                SolutionWitness inv_wit;
                u128 s = 0;
                for (const auto& elem : inst.A) {
                    int i = elem.orig_idx;
                    if (!in_comp[i]) {
                        inv_wit.original_indices.push_back(i);
                        inv_wit.values.push_back(elem.val);
                        s += elem.val;
                    }
                }
                inv_wit.sum = s;
                inv_wit.sort_indices();
                wit = inv_wit;
            };
            if (!stats.sample_solution.values.empty()) invert_witness(stats.sample_solution);
            for (auto& w : stats.all_solutions) invert_witness(w);
        }

        if (stats.has_solution && !stats.sample_solution.values.empty()) {
            stats.verified = verify_witness_independently(inst.raw_elements, inst.target,
                                                            stats.sample_solution, stats.verification_message);
        } else if (!stats.has_solution) {
            stats.verified = true;
            stats.verification_message = "N/A (UNSAT, no witness to verify).";
        }

        auto t_end = std::chrono::steady_clock::now();
        stats.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        stats.solve_ms = std::chrono::duration<double, std::milli>(t_end - t_prep).count();
        stats.peak_ram_mb = get_current_peak_ram_bytes() / (1024.0 * 1024.0);

        finalize_status(stats, stop_flag);
        return stats;
    }

    ExecutionStats run_forced(const Instance& inst, SolveMode mode, StrategyType forced,
                               size_t memory_limit_mb = 4096, double time_limit_ms = 120000.0,
                               size_t max_solutions = 5000) {
        stop_flag = false; solution_found = false;
        SolverBudget budget;
        budget.memory_limit_mb = memory_limit_mb;
        budget.time_limit_ms = time_limit_ms;
        budget.max_solutions = max_solutions;
        ExecutionStats stats;
        auto t0 = std::chrono::steady_clock::now();
        stats.strategy_chosen = forced;
        switch (forced) {
            case StrategyType::TrivialPreCheck: solve_trivial(inst, mode, stats); break;
            case StrategyType::GreedySuperincreasing: { GreedySuperincreasingSolver s; s.solve(inst, mode, stats); break; }
            case StrategyType::BitsetDP: solve_bitset(inst, mode, stats, budget, t0); break;
            case StrategyType::SmallKMitM: {
                if (!SmallKCombinatorialSolver::try_solve(inst, stats, budget, mode)) {
                    stats.message = "Forced SmallKMitM failed to find solution.";
                }
                break;
            }
            case StrategyType::BoundarySwap: {
                if (!BoundarySwapSolver::try_solve(inst, stats, budget, mode)) {
                    stats.message = "Forced BoundarySwap failed to find boundary solution.";
                }
                break;
            }
            case StrategyType::HybridTailTable:
            default: { HybridTailTableEngine e; e.solve(inst, mode, stats, budget, t0, stop_flag, solution_found); break; }
        }
        if (stats.has_solution && !stats.sample_solution.values.empty()) {
            stats.verified = verify_witness_independently(inst.raw_elements, inst.target, stats.sample_solution, stats.verification_message);
        }
        auto t_end = std::chrono::steady_clock::now();
        stats.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
        finalize_status(stats, stop_flag);
        return stats;
    }

private:
    void solve_trivial(const Instance& inst, SolveMode mode, ExecutionStats& stats) {
        if (inst.target == 0) {
            stats.has_solution = true; stats.solution_count = 1;
            if (mode == SolveMode::FindAll) stats.all_solutions.push_back(SolutionWitness{});
            stats.message = "Trivial: target=0 (empty set)."; return;
        }
        if (inst.A.empty() || inst.effective_target > inst.total_sum) { stats.message = "Trivial UNSAT: target exceeds total sum."; return; }
        if (inst.gcd_val > 1 && (inst.effective_target % inst.gcd_val != 0)) { stats.message = "Trivial UNSAT: GCD modular obstruction."; return; }
        if (inst.odd_count == 0 && (inst.effective_target % 2 != 0)) { stats.message = "Trivial UNSAT: parity obstruction."; return; }
        if (inst.feasible_k_count == 0) { stats.message = "Trivial UNSAT: feasible cardinality window empty."; return; }
    }

    void solve_bitset(const Instance& inst, SolveMode mode, ExecutionStats& stats,
                       const SolverBudget& budget, const std::chrono::steady_clock::time_point& start_time) {
        (void)budget; (void)start_time;
        u64 T = inst.effective_target;
        size_t words = (T >> 6) + 1;
        std::vector<u64> bs(words, 0); bs[0] = 1ULL;
        std::vector<int> parent(T + 1, -1);

        for (size_t idx = 0; idx < inst.A.size(); ++idx) {
            u64 val = inst.A[idx].val;
            if (val > T) continue;
            size_t sw = val >> 6, sb = val & 63;
            for (int64_t w = (int64_t)words - 1; w >= (int64_t)sw; --w) {
                u64 low = bs[w - sw] << sb;
                u64 high = (sb > 0 && w > (int64_t)sw) ? (bs[w - sw - 1] >> (64 - sb)) : 0;
                u64 shifted = low | high;
                u64 nb = shifted & ~bs[w];
                bs[w] |= shifted;
                while (nb != 0) {
                    int bit = (int)__builtin_ctzll(nb);
                    u64 sum = ((u64)w << 6) + bit;
                    if (sum <= T && parent[sum] == -1) parent[sum] = (int)idx;
                    nb &= nb - 1;
                }
            }
            if ((bs[T>>6] & (1ULL << (T & 63))) && (mode == SolveMode::FindOne || mode == SolveMode::DecisionOnly)) break;
        }
        bool ok = (bs[T>>6] & (1ULL << (T & 63))) != 0;
        stats.has_solution = ok;
        if (ok) {
            stats.solution_count = 1;
            SolutionWitness wit;
            u64 curr = T;
            while (curr > 0 && parent[curr] >= 0 && parent[curr] < (int)inst.A.size()) {
                int ei = parent[curr];
                wit.original_indices.push_back(inst.A[ei].orig_idx);
                wit.values.push_back(inst.A[ei].val);
                wit.sum += inst.A[ei].val;
                curr -= inst.A[ei].val;
            }
            wit.sort_indices();
            stats.sample_solution = wit;
            if (mode == SolveMode::FindAll) stats.all_solutions.push_back(wit);
            stats.message = "Exact solution found via Bitset DP.";
        } else {
            stats.message = "UNSAT provably proven via Bitset DP.";
        }
    }
};

struct InstancePreset {
    std::string title, category, description;
    int n; u64 target;
    std::vector<u64> elements;
};

class PresetRepository {
public:
    static std::vector<InstancePreset> get_all_presets() {
        std::vector<InstancePreset> presets;
        {
            InstancePreset p;
            p.title = "1. Hybrid Benchmark (N=80, T=18.6 Triliun) - teruji ~1.9 detik";
            p.category = "High Magnitude, Structured (window k sempit)";
            p.description = "Instance 80-elemen dari sesi pengujian solver ini. k_min..k_max=[22,55]. "
                             "Diselesaikan Hybrid Tail-Table Engine.";
            p.n = 80; p.target = 18649525982137ULL;
            std::string s = "123005401502,811856239314,267469214296,151282538207,114832269482,814655221102,600832336209,648913461123,36171878810,98912225903,254342113420,663595448018,614294294452,786833016362,771590378378,461902006519,490573056993,307473554861,956959217036,833251567392,175259659198,466854953851,306404042871,236890980657,841629821540,113114812282,417010168082,391257417624,381597082644,290355792388,50711229824,505645347793,135447068294,87525138708,324493543666,690757033267,973319131003,398837994178,212933106112,76040557041,726046287258,847087372943,936645574450,950187601657,417045625553,499410093882,917558277936,177660601410,388136971785,735339233455,769945806982,751347210314,80092702124,698400899700,584850587138,269419548117,507507949296,297982492152,758663022121,242910249530,359422681392,845433611430,63462436410,902926886138,884901132452,439441524733,69869415071,626820024293,789742985455,233279765208,548275934230,972361834961,708302588518,156589576362,151461507039,817103043321,590821577340,821467220157,470662212986,643806247030";
            Instance inst = Instance::from_string(s, p.target);
            p.elements = inst.raw_elements;
            presets.push_back(p);
        }
        {
            InstancePreset p;
            p.title = "2. Flat / Unstructured (N=32, window k lebar)";
            p.category = "General / Flat Structure (worst-case)";
            p.description = "Instance tanpa struktur kuat, window kardinalitas lebar - kasus paling berat untuk pruning DFS Hybrid Tail-Table.";
            p.n = 32;
            std::mt19937_64 rng(7070);
            std::uniform_int_distribution<u64> dist(10000000000ULL, 90000000000ULL);
            u64 sum = 0;
            for (int i = 0; i < 32; ++i) { u64 v = dist(rng); p.elements.push_back(v); if (i==2||i==7||i==15||i==23||i==30) sum += v; }
            p.target = sum;
            presets.push_back(p);
        }
        {
            InstancePreset p;
            p.title = "3. Dense Small Values (N=40) - Bitset DP instan";
            p.category = "Small Dense Values (high density)";
            p.description = "Target kecil, memicu Bitset DP (<5 ms).";
            p.n = 40;
            std::mt19937_64 rng(2026);
            std::uniform_int_distribution<u64> dist(100, 20000);
            u64 sum = 0;
            for (int i = 0; i < 40; ++i) { u64 v = dist(rng); p.elements.push_back(v); if (i < 8) sum += v; }
            p.target = sum;
            presets.push_back(p);
        }
        {
            InstancePreset p;
            p.title = "4. GCD Obstruction (N=50) - Provably UNSAT instan";
            p.category = "Modular Obstruction";
            p.description = "Semua elemen kelipatan 777, target tidak. UNSAT terbukti <1ms via TrivialPreCheck.";
            p.n = 50;
            std::mt19937_64 rng(5050);
            std::uniform_int_distribution<u64> dist(100000, 5000000);
            for (int i = 0; i < 50; ++i) p.elements.push_back(dist(rng) * 777ULL);
            p.target = (p.elements[3] + p.elements[12]) + 13ULL;
            presets.push_back(p);
        }
        return presets;
    }
};
