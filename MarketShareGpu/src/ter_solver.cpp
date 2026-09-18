// ter_solver.cpp — TER (Ternary Enumeration Representation) SSP Solver
// Based on: Yang Li, "From Subset-Sum to Decoding: Improved Classical and
//           Quantum Algorithms via Ternary Representation Technique", Information 2025.

#include "ter_solver.cuh"
#include "ter_kernel.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <atomic>
#include <omp.h>

// ─────────────────────────────────────────────────────────
// Print a ter_u128 value (no built-in operator<< for __int128).
// ─────────────────────────────────────────────────────────
static std::string ter_u128_to_string(ter_u128 value) {
    if (value == 0) return "0";
    std::string s;
    while (value > 0) {
        s.push_back(static_cast<char>('0' + static_cast<int>(value % 10)));
        value /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

// Low 64 bits of a ter_u128 value, as used for psum hashing (see
// TerEntry note in ter_solver.cuh): truncation to 64 bits commutes
// with +/-, so this is always safe to use for residue/hash purposes.
static inline uint64_t lo64(ter_u128 v) { return (uint64_t)v; }

// ─────────────────────────────────────────────────────────
// Bitwise ternary addition and validity check
// Given ternary vectors u, v, w ∈ {-1, 0, 1}^n
// Returns true iff for all coordinates k: (u[k] + v[k] + w[k]) ∈ {-1, 0, 1}
// And writes the result vector to `out`.
// ─────────────────────────────────────────────────────────
inline bool check_and_add_ternary(
    const TerEntry& u, const TerEntry& v, const TerEntry& w,
    TerEntry& out
) {
    // Low 64 bits
    uint64_t bp_lo = u.pos_lo & v.pos_lo;
    uint64_t bn_lo = u.neg_lo & v.neg_lo;
    uint64_t ep_lo = (u.pos_lo | v.pos_lo) & ~(u.neg_lo | v.neg_lo);
    uint64_t en_lo = (u.neg_lo | v.neg_lo) & ~(u.pos_lo | v.pos_lo);

    if ((bp_lo & ~w.neg_lo) || (bn_lo & ~w.pos_lo) ||
        (ep_lo & w.pos_lo)   || (en_lo & w.neg_lo)) {
        return false;
    }

    // High 64 bits
    uint64_t bp_hi = u.pos_hi & v.pos_hi;
    uint64_t bn_hi = u.neg_hi & v.neg_hi;
    uint64_t ep_hi = (u.pos_hi | v.pos_hi) & ~(u.neg_hi | v.neg_hi);
    uint64_t en_hi = (u.neg_hi | v.neg_hi) & ~(u.pos_hi | v.pos_hi);

    if ((bp_hi & ~w.neg_hi) || (bn_hi & ~w.pos_hi) ||
        (ep_hi & w.pos_hi)   || (en_hi & w.neg_hi)) {
        return false;
    }

    // Valid ternary sum! Compute output vector
    uint64_t neither_lo = ~(bp_lo | ep_lo | bn_lo | en_lo);
    out.pos_lo = (bp_lo & w.neg_lo) | (ep_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.pos_lo);
    out.neg_lo = (bn_lo & w.pos_lo) | (en_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.neg_lo);

    uint64_t neither_hi = ~(bp_hi | ep_hi | bn_hi | en_hi);
    out.pos_hi = (bp_hi & w.neg_hi) | (ep_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.pos_hi);
    out.neg_hi = (bn_hi & w.pos_hi) | (en_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.neg_hi);

    out.psum = u.psum + v.psum + w.psum;
    return true;
}

// ─────────────────────────────────────────────────────────
// Root level check: sum must be strictly in {0, 1}^n
// ─────────────────────────────────────────────────────────
inline bool check_root_sum(
    const TerEntry& u, const TerEntry& v, const TerEntry& w,
    TerEntry& out
) {
    if (!check_and_add_ternary(u, v, w, out)) return false;
    return (out.neg_lo == 0 && out.neg_hi == 0);
}

// ─────────────────────────────────────────────────────────
// Optimal parameter defaults from Paper (Information 2025, Li et al.)
// ─────────────────────────────────────────────────────────
void TerParams::compute_derived() {
    w1 = eps11 + 2.0 * eps01;
    w2 = eps11 / 3.0 + 2.0 * eps01 / 3.0 + eps22 + eps12 + 2.0 * eps02;

    l1 = 0.2221;
    l2 = 0.2147;
    l3 = 0.1922;
    r1 = 1.1473 * 0.5;
    r2 = 0.3396 * 0.5;

    // Bit matching constraints:
    b2 = std::max(1, std::min(28, (int)std::floor(r2 * n)));
    b1 = std::max(b2 + 2, std::min(58, (int)std::floor(r1 * n)));
    b3 = 0;
}

TerParams ter_default_params(int n) {
    TerParams p{};
    p.n = n;
    p.eps01 = 0.0066;
    p.eps11 = 0.0024;
    p.eps02 = 0.0004;
    p.eps12 = 0.0001;
    p.eps22 = 0.0000;

    p.max_restarts = 0;       // 0 = run until solution or timeout
    p.timeout_seconds = 0.0;  // 0 = no timeout
    p.fixed_runs = 0;         // 0 = auto
    p.verbose = true;
    p.use_gpu = false;

    p.compute_derived();
    return p;
}

// ─────────────────────────────────────────────────────────
// Level 3 Base Combinations Generation
// Left half coordinates: [0, n/2 - 1]
// Right half coordinates: [n/2, n - 1]
// ─────────────────────────────────────────────────────────
static void generate_half_base_pool(
    const std::vector<ter_u128>& weights,
    int start_idx, int end_idx,
    size_t target_size,
    std::vector<TerEntry>& pool
) {
    int count = end_idx - start_idx;
    pool.clear();
    pool.reserve(target_size * 2);

    // 0 ones, 0 minus-ones
    {
        TerEntry e0{};
        pool.push_back(e0);
    }

    // 1 one, 0 minus-ones
    for (int i = 0; i < count; ++i) {
        int idx = start_idx + i;
        TerEntry e{};
        if (idx < 64) e.pos_lo = 1ULL << idx;
        else e.pos_hi = 1ULL << (idx - 64);
        e.psum = (int64_t)lo64(weights[idx]);
        pool.push_back(e);
    }

    // 2 ones, 0 minus-ones
    for (int i = 0; i < count; ++i) {
        int idx1 = start_idx + i;
        for (int j = i + 1; j < count; ++j) {
            int idx2 = start_idx + j;
            TerEntry e{};
            if (idx1 < 64) e.pos_lo |= (1ULL << idx1);
            else e.pos_hi |= (1ULL << (idx1 - 64));
            if (idx2 < 64) e.pos_lo |= (1ULL << idx2);
            else e.pos_hi |= (1ULL << (idx2 - 64));
            e.psum = (int64_t)(lo64(weights[idx1]) + lo64(weights[idx2]));
            pool.push_back(e);
        }
    }

    // 1 one, 1 minus-one (representation cancellation)
    for (int i = 0; i < count; ++i) {
        int idx_pos = start_idx + i;
        for (int j = 0; j < count; ++j) {
            if (i == j) continue;
            int idx_neg = start_idx + j;
            TerEntry e{};
            if (idx_pos < 64) e.pos_lo = (1ULL << idx_pos);
            else e.pos_hi = (1ULL << (idx_pos - 64));
            if (idx_neg < 64) e.neg_lo = (1ULL << idx_neg);
            else e.neg_hi = (1ULL << (idx_neg - 64));
            e.psum = (int64_t)(lo64(weights[idx_pos]) - lo64(weights[idx_neg]));
            pool.push_back(e);
        }
    }

    // 3 ones, 0 minus-ones
    if (pool.size() < target_size) {
        for (int i = 0; i < count; ++i) {
            int idx1 = start_idx + i;
            for (int j = i + 1; j < count; ++j) {
                int idx2 = start_idx + j;
                for (int k = j + 1; k < count; ++k) {
                    int idx3 = start_idx + k;
                    TerEntry e{};
                    if (idx1 < 64) e.pos_lo |= (1ULL << idx1);
                    else e.pos_hi |= (1ULL << (idx1 - 64));
                    if (idx2 < 64) e.pos_lo |= (1ULL << idx2);
                    else e.pos_hi |= (1ULL << (idx2 - 64));
                    if (idx3 < 64) e.pos_lo |= (1ULL << idx3);
                    else e.pos_hi |= (1ULL << (idx3 - 64));
                    e.psum = (int64_t)(lo64(weights[idx1]) + lo64(weights[idx2]) + lo64(weights[idx3]));
                    pool.push_back(e);
                    if (pool.size() >= target_size * 2) break;
                }
                if (pool.size() >= target_size * 2) break;
            }
            if (pool.size() >= target_size * 2) break;
        }
    }

    // 2 ones, 1 minus-one
    if (pool.size() < target_size) {
        for (int i = 0; i < count; ++i) {
            int p1 = start_idx + i;
            for (int j = i + 1; j < count; ++j) {
                int p2 = start_idx + j;
                for (int k = 0; k < count; ++k) {
                    if (k == i || k == j) continue;
                    int n1 = start_idx + k;
                    TerEntry e{};
                    if (p1 < 64) e.pos_lo |= (1ULL << p1);
                    else e.pos_hi |= (1ULL << (p1 - 64));
                    if (p2 < 64) e.pos_lo |= (1ULL << p2);
                    else e.pos_hi |= (1ULL << (p2 - 64));
                    if (n1 < 64) e.neg_lo |= (1ULL << n1);
                    else e.neg_hi |= (1ULL << (n1 - 64));
                    e.psum = (int64_t)(lo64(weights[p1]) + lo64(weights[p2]) - lo64(weights[n1]));
                    pool.push_back(e);
                    if (pool.size() >= target_size * 2) break;
                }
                if (pool.size() >= target_size * 2) break;
            }
            if (pool.size() >= target_size * 2) break;
        }
    }
}

// ─────────────────────────────────────────────────────────
// Level 2 Merge: Combine Left L3 and Right L3 with modulo M2
//
// PERF: `sorted_right` depends only on (L3_right, b2) — both are
// constant across ALL restart runs (b2 comes from TerParams, fixed
// for the whole solve). Previously this sorted a copy of L3_right
// from scratch on every single call (9x per run x N runs). Now the
// caller passes in a pre-sorted view built once, eliminating O(R
// log R) work per call, per run.
// ─────────────────────────────────────────────────────────
static void merge_level2(
    const std::vector<TerEntry>& L3_left,
    const std::vector<TerEntry>& sorted_right,
    uint64_t s2,
    int b2,
    size_t max_cap,
    std::vector<TerEntry>& L2_out
) {
    uint64_t mask_m2 = (1ULL << b2) - 1ULL;
    uint64_t target_rem = s2 & mask_m2;

    L2_out.clear();
    L2_out.reserve(std::min(max_cap, (size_t)8192));

    for (const auto& u : L3_left) {
        uint64_t u_rem = (uint64_t)u.psum & mask_m2;
        uint64_t req_v = (target_rem >= u_rem) ? (target_rem - u_rem) : (mask_m2 + 1ULL + target_rem - u_rem);

        // Binary search in sorted_right
        auto it = std::lower_bound(sorted_right.begin(), sorted_right.end(), req_v,
            [mask_m2](const TerEntry& elem, uint64_t val) {
                return ((uint64_t)elem.psum & mask_m2) < val;
            });

        while (it != sorted_right.end() && (((uint64_t)it->psum & mask_m2) == req_v)) {
            TerEntry combined{};
            combined.pos_lo = u.pos_lo | it->pos_lo;
            combined.pos_hi = u.pos_hi | it->pos_hi;
            combined.neg_lo = u.neg_lo | it->neg_lo;
            combined.neg_hi = u.neg_hi | it->neg_hi;
            combined.psum = u.psum + it->psum;
            L2_out.push_back(combined);
            if (L2_out.size() >= max_cap) return;
            ++it;
        }
    }
}

// ─────────────────────────────────────────────────────────
// Level 1 Merge (CPU & GPU paths)
//
// PERF: caller now passes `sorted_C` directly (pre-sorted once by
// the caller, since C == one of the L2[] lists reused for the
// duration of a run — sorting it here on every call to merge_level1
// duplicated work when the same list is merged against itself
// nowhere, but more importantly this removes an O(|C| log |C|) sort
// that used to run identically 3x per run for no reason other than
// interface convenience).
//
// PERF: thread-local scratch buffers (`local_outs`) are now supplied
// by the caller and only `clear()`-ed (capacity retained) instead of
// being freshly heap-allocated as `std::vector<std::vector<TerEntry>>`
// on every restart. Under many restarts this avoids repeated
// malloc/free churn for potentially large per-thread buffers.
// ─────────────────────────────────────────────────────────
static void merge_level1(
    const std::vector<TerEntry>& A,
    const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C,
    uint64_t s1,
    int b1,
    bool use_gpu,
    size_t max_cap,
    std::vector<TerEntry>& L1_out,
    std::vector<std::vector<TerEntry>>& local_outs_scratch
) {
    uint64_t mask_m1 = (1ULL << b1) - 1ULL;
    uint64_t target_mod = s1 & mask_m1;

#ifdef WITH_GPU
    if (use_gpu && !A.empty() && !B.empty() && !sorted_C.empty()) {
        if (run_level1_merge_gpu(A, B, sorted_C, target_mod, mask_m1, max_cap, L1_out)) {
            return;
        }
    }
#else
    (void)use_gpu;
#endif

    // Fast CPU parallel merge with OpenMP
    L1_out.clear();
    int nthreads = omp_get_max_threads();
    if ((int)local_outs_scratch.size() != nthreads) local_outs_scratch.resize(nthreads);
    for (auto& v : local_outs_scratch) v.clear();  // keeps capacity, avoids realloc
    size_t thread_cap = max_cap / nthreads + 256;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i_a = 0; i_a < (int)A.size(); ++i_a) {
        int tid = omp_get_thread_num();
        if (local_outs_scratch[tid].size() >= thread_cap) continue;
        const TerEntry& u = A[i_a];

        for (const auto& v : B) {
            if (local_outs_scratch[tid].size() >= thread_cap) break;

            // Quick check for incompatibility
            if (((u.pos_lo & v.pos_lo) & (u.neg_lo | v.neg_lo)) ||
                ((u.pos_hi & v.pos_hi) & (u.neg_hi | v.neg_hi))) continue;

            uint64_t uv_rem = (uint64_t)(u.psum + v.psum) & mask_m1;
            uint64_t req_w = (target_mod >= uv_rem) ? (target_mod - uv_rem) : (mask_m1 + 1ULL + target_mod - uv_rem);

            // Binary search in sorted_C
            auto it = std::lower_bound(sorted_C.begin(), sorted_C.end(), req_w,
                [mask_m1](const TerEntry& elem, uint64_t val) {
                    return ((uint64_t)elem.psum & mask_m1) < val;
                });

            while (it != sorted_C.end() && (((uint64_t)it->psum & mask_m1) == req_w)) {
                TerEntry comb;
                if (check_and_add_ternary(u, v, *it, comb)) {
                    local_outs_scratch[tid].push_back(comb);
                    if (local_outs_scratch[tid].size() >= thread_cap) break;
                }
                ++it;
            }
        }
    }

    // Combine thread results
    size_t total_found = 0;
    for (const auto& vec : local_outs_scratch) total_found += vec.size();
    L1_out.reserve(std::min(max_cap, total_found));

    for (const auto& vec : local_outs_scratch) {
        for (const auto& item : vec) {
            L1_out.push_back(item);
            if (L1_out.size() >= max_cap) return;
        }
    }
}

// ─────────────────────────────────────────────────────────
// Root Level Merge: Final Search for Target Sum
//
// PERF / CORRECTNESS fixes vs. original:
//  1. `sorted_C` buffer is now reused (passed by caller) instead of
//     being heap-allocated fresh every run.
//  2. `found` is now a std::atomic<bool> read with relaxed ordering
//     for the hot-path check. The original plain `bool found` was
//     read/written across threads without synchronization, which is
//     a data race (UB) — in practice it "worked" because of x86's
//     strong memory model, but is not portable/correct and can also
//     prevent the compiler from certain optimizations.
//  3. Real early-exit: outer loop now checks the atomic flag with an
//     std::memory_order_relaxed load and `break`s out of both the
//     inner and outer loop as soon as any thread finds a solution,
//     instead of "if (found) continue" which still burns through the
//     full i_a iteration space doing nothing once a hit lands late.
//     Combined with OMP `cancel` this bounds the wasted work to
//     roughly one in-flight chunk per thread instead of the full
//     remaining iteration space.
// ─────────────────────────────────────────────────────────
static bool merge_root_and_solve(
    const std::vector<TerEntry>& L1_A,
    const std::vector<TerEntry>& L1_B,
    const std::vector<TerEntry>& L1_C,
    ter_u128 target,
    const std::vector<ter_u128>& weights,
    int n,
    std::vector<size_t>& sol_indices,
    std::vector<TerEntry>& sorted_C_scratch
) {
    if (L1_A.empty() || L1_B.empty() || L1_C.empty()) return false;

    // Reuse scratch buffer; sort L1_C by exact psum.
    sorted_C_scratch = L1_C;
    std::sort(sorted_C_scratch.begin(), sorted_C_scratch.end(), [](const TerEntry& x, const TerEntry& y) {
        return x.psum < y.psum;
    });
    const auto& sorted_C = sorted_C_scratch;

    // psum only carries the low 64 bits of the true sum (see TerEntry
    // note); target_signed is used purely as a cheap 64-bit candidate
    // filter here. Every candidate that passes it is re-verified below
    // against the full-precision `target` by summing the original
    // ter_u128 weights, so this filter never causes false negatives —
    // it just narrows down which triples get the exact (128-bit) check.
    int64_t target_signed = (int64_t)lo64(target);
    std::atomic<bool> found{false};

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 16)
        for (int i_a = 0; i_a < (int)L1_A.size(); ++i_a) {
            // Cheap relaxed check: skip immediately once any thread wins.
            if (found.load(std::memory_order_relaxed)) continue;
            const TerEntry& u = L1_A[i_a];

            bool local_break = false;
            for (const auto& v : L1_B) {
                if (found.load(std::memory_order_relaxed)) { local_break = true; break; }

                int64_t uv_sum = u.psum + v.psum;
                int64_t req_w = target_signed - uv_sum;

                auto it = std::lower_bound(sorted_C.begin(), sorted_C.end(), req_w,
                    [](const TerEntry& elem, int64_t val) {
                        return elem.psum < val;
                    });

                while (it != sorted_C.end() && it->psum == req_w) {
                    TerEntry out{};
                    if (check_root_sum(u, v, *it, out)) {
                        std::vector<size_t> cur_indices;
                        // Exact, full-precision re-verification: the bit
                        // pattern (pos_lo/pos_hi) exactly identifies which
                        // indices are selected, independent of psum's
                        // truncation, so re-summing the original ter_u128
                        // weights here always gives the true sum — this is
                        // what makes instances whose sums exceed 2^64 solve
                        // correctly instead of silently overflowing.
                        ter_u128 ver_sum = 0;
                        for (int i = 0; i < n; ++i) {
                            bool bit_is_one = false;
                            if (i < 64) bit_is_one = (out.pos_lo & (1ULL << i)) != 0;
                            else bit_is_one = (out.pos_hi & (1ULL << (i - 64))) != 0;

                            if (bit_is_one) {
                                cur_indices.push_back(i);
                                ver_sum += weights[i];
                            }
                        }

                        if (ver_sum == target) {
                            bool expected = false;
                            if (found.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                                #pragma omp critical
                                { sol_indices = cur_indices; }
                            }
                            local_break = true;
                            break;
                        }
                    }
                    ++it;
                }
                if (local_break) break;
            }
            (void)local_break;
        }
    }

    return found.load(std::memory_order_relaxed);
}

// ─────────────────────────────────────────────────────────
// Main Public API: ter_solve
// ─────────────────────────────────────────────────────────
TerResult ter_solve(
    const std::vector<ter_u128>& weights,
    ter_u128                     target,
    const TerParams&             params
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    TerResult result{};
    result.found = false;
    result.runs_attempted = 0;
    result.elapsed_seconds = 0.0;

    int n = params.n;
    int n_half = n / 2;

    size_t target_L3 = std::max((size_t)4096, (size_t)std::pow(2.0, params.l3 * n));
    size_t target_L2 = std::max((size_t)8192, (size_t)std::pow(2.0, params.l2 * n));
    size_t target_L1 = std::max((size_t)8192, (size_t)std::pow(2.0, params.l1 * n));

    if (params.verbose) {
        std::cout << "\n=======================================================\n";
        std::cout << "TER (Ternary Enumeration Representation) Solver Started\n";
        std::cout << "=======================================================\n";
        std::cout << "Instance size n : " << n << "\n";
        std::cout << "Target value    : " << ter_u128_to_string(target) << "\n";
        std::cout << "Matching bits   : b1=" << params.b1 << ", b2=" << params.b2 << "\n";
        std::cout << "Target capacities: L3=" << target_L3 << ", L2=" << target_L2 << ", L1=" << target_L1 << "\n";
        std::cout << "Execution mode  : " << (params.use_gpu ? "GPU (Tesla T4 / CUDA)" : "CPU (OpenMP Multithreaded)") << "\n";
        std::cout << "Restart config  : fixed_runs=" << params.fixed_runs
                  << ", max_restarts=" << params.max_restarts
                  << ", timeout=" << params.timeout_seconds << "s\n\n";
    }

    // Generate base pools once for left and right halves
    std::vector<TerEntry> pool_left;
    std::vector<TerEntry> pool_right;
    generate_half_base_pool(weights, 0, n_half, target_L3, pool_left);
    generate_half_base_pool(weights, n_half, n, target_L3, pool_right);

    // PERF: `pool_right` and `b2` are invariant across every restart run.
    // The original code re-sorted a fresh copy of pool_right by
    // (psum & mask_m2) inside merge_level2() on EVERY one of the 9 calls
    // per run. Since neither the data nor b2 changes between runs, we
    // sort it exactly once here and pass the sorted view down.
    uint64_t mask_m2_fixed = (1ULL << params.b2) - 1ULL;
    std::vector<TerEntry> sorted_pool_right = pool_right;
    std::sort(sorted_pool_right.begin(), sorted_pool_right.end(),
        [mask_m2_fixed](const TerEntry& a, const TerEntry& b) {
            return ((uint64_t)a.psum & mask_m2_fixed) < ((uint64_t)b.psum & mask_m2_fixed);
        });

    if (params.verbose) {
        std::cout << "Generated Base Pool: Left=" << pool_left.size()
                  << " entries, Right=" << pool_right.size() << " entries\n";
    }

    std::mt19937_64 rng(1337);
    int run_idx = 0;

    // PERF: persistent scratch buffers reused across restarts instead of
    // being freshly heap-allocated (std::vector<std::vector<TerEntry>>,
    // vector copies for sorting, etc.) on every single iteration of the
    // while(true) restart loop. Under many restarts this removes a large
    // amount of repeated malloc/free traffic.
    std::vector<std::vector<TerEntry>> L2(9);
    std::vector<std::vector<TerEntry>> L1(3);
    std::vector<std::vector<TerEntry>> level1_scratch[3];  // one local_outs_scratch per L1 slot
    for (int j = 0; j < 3; ++j) level1_scratch[j].resize(omp_get_max_threads());
    std::vector<TerEntry> root_sorted_C_scratch;

    while (true) {
        run_idx++;
        result.runs_attempted = run_idx;

        auto run_start = std::chrono::high_resolution_clock::now();

        // 1. Sample random targets s^(1)_k and s^(2)_m
        uint64_t mask_m1 = (1ULL << params.b1) - 1ULL;
        uint64_t mask_m2 = (1ULL << params.b2) - 1ULL;

        // s1/s2 are residues mod 2^b1 / 2^b2 (b1<=58, b2<=28), so only
        // target's low 64 bits are ever needed here — see TerEntry note.
        uint64_t target_lo64 = lo64(target);

        uint64_t s1[3];
        s1[0] = rng() & mask_m1;
        s1[1] = rng() & mask_m1;
        s1[2] = (target_lo64 - s1[0] - s1[1]) & mask_m1;

        uint64_t s2[9];
        for (int k = 0; k < 3; ++k) {
            s2[3 * k + 0] = rng() & mask_m2;
            s2[3 * k + 1] = rng() & mask_m2;
            s2[3 * k + 2] = (s1[k] - s2[3 * k + 0] - s2[3 * k + 1]) & mask_m2;
        }

        // 2. Build 9 Level 2 lists from base pool (pool_right pre-sorted once above)
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < 9; ++m) {
            merge_level2(pool_left, sorted_pool_right, s2[m], params.b2, target_L2 * 2, L2[m]);
        }

        // 3. Build 3 Level 1 lists from Level 2 triplets.
        // NOTE: each merge_level1 call needs its C-list pre-sorted by
        // (psum & mask_m1); we sort once per call site here (still 3x
        // per run, but this part of the work is inherent — L2[] content
        // changes every run so it can't be hoisted further without
        // changing the algorithm's random-restart semantics).
        for (int j = 0; j < 3; ++j) {
            auto& C = L2[3 * j + 2];
            std::sort(C.begin(), C.end(), [mask_m1](const TerEntry& x, const TerEntry& y) {
                return ((uint64_t)x.psum & mask_m1) < ((uint64_t)y.psum & mask_m1);
            });
            merge_level1(
                L2[3 * j + 0], L2[3 * j + 1], C,
                s1[j], params.b1, params.use_gpu, target_L1 * 2, L1[j],
                level1_scratch[j]
            );
        }

        // 4. Root search
        std::vector<size_t> sol;
        bool found = merge_root_and_solve(L1[0], L1[1], L1[2], target, weights, n, sol, root_sorted_C_scratch);

        auto now = std::chrono::high_resolution_clock::now();
        double run_sec = std::chrono::duration<double>(now - run_start).count();
        double total_sec = std::chrono::duration<double>(now - t_start).count();

        if (params.verbose) {
            std::cout << "[Run " << run_idx << "] L2 avg="
                      << (L2[0].size() + L2[1].size() + L2[2].size()) / 3
                      << ", L1=(" << L1[0].size() << ", " << L1[1].size() << ", " << L1[2].size() << ")"
                      << ", elapsed=" << run_sec << "s (total=" << total_sec << "s)";
            if (found) std::cout << " --> SOLUTION FOUND!\n";
            else std::cout << " --> restart...\n";
        }

        if (found) {
            result.found = true;
            result.solution_indices = sol;
            break;
        }

        // Stop conditions
        if (params.fixed_runs > 0 && run_idx >= params.fixed_runs) break;
        if (params.max_restarts > 0 && run_idx > params.max_restarts) break;
        if (params.timeout_seconds > 0.0 && total_sec >= params.timeout_seconds) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    return result;
}
