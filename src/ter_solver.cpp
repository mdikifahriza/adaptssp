#include "ter_solver.cuh"
#ifdef WITH_GPU
#include "ter_kernel.cuh"
#endif
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <omp.h>
#include <thread>

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

static inline uint64_t lo64(ter_u128 v) { return (uint64_t)v; }

inline bool check_and_add_ternary(
    const TerEntry& u, const TerEntry& v, const TerEntry& w,
    TerEntry& out
) {
    uint64_t bp_lo = u.pos_lo & v.pos_lo;
    uint64_t bn_lo = u.neg_lo & v.neg_lo;
    uint64_t ep_lo = (u.pos_lo | v.pos_lo) & ~(u.neg_lo | v.neg_lo);
    uint64_t en_lo = (u.neg_lo | v.neg_lo) & ~(u.pos_lo | v.pos_lo);

    if ((bp_lo & ~w.neg_lo) || (bn_lo & ~w.pos_lo) ||
        (ep_lo & w.pos_lo)   || (en_lo & w.neg_lo)) {
        return false;
    }

    uint64_t bp_hi = u.pos_hi & v.pos_hi;
    uint64_t bn_hi = u.neg_hi & v.neg_hi;
    uint64_t ep_hi = (u.pos_hi | v.pos_hi) & ~(u.neg_hi | v.neg_hi);
    uint64_t en_hi = (u.neg_hi | v.neg_hi) & ~(u.pos_hi | v.pos_hi);

    if ((bp_hi & ~w.neg_hi) || (bn_hi & ~w.pos_hi) ||
        (ep_hi & w.pos_hi)   || (en_hi & w.neg_hi)) {
        return false;
    }

    uint64_t neither_lo = ~(bp_lo | ep_lo | bn_lo | en_lo);
    out.pos_lo = (bp_lo & w.neg_lo) | (ep_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.pos_lo);
    out.neg_lo = (bn_lo & w.pos_lo) | (en_lo & ~(w.pos_lo | w.neg_lo)) | (neither_lo & w.neg_lo);

    uint64_t neither_hi = ~(bp_hi | ep_hi | bn_hi | en_hi);
    out.pos_hi = (bp_hi & w.neg_hi) | (ep_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.pos_hi);
    out.neg_hi = (bn_hi & w.pos_hi) | (en_hi & ~(w.pos_hi | w.neg_hi)) | (neither_hi & w.neg_hi);

    out.psum = u.psum + v.psum + w.psum;
    return true;
}

inline bool check_root_sum(
    const TerEntry& u, const TerEntry& v, const TerEntry& w,
    TerEntry& out
) {
    if (!check_and_add_ternary(u, v, w, out)) return false;
    return (out.neg_lo == 0 && out.neg_hi == 0);
}

static int bit_width_u128(ter_u128 v) {
    int bits = 0;
    while (v) { v >>= 1; ++bits; }
    return bits;
}

void TerParams::compute_derived() {
    w2 = eps11 / 3.0 + 2.0 * eps01 / 3.0 + eps22 + eps12 + 2.0 * eps02;

    l1 = 0.2221;
    l2 = 0.2147;
    l3 = 0.1922;
    r1 = 1.1473 * 0.5;
    r2 = 0.3396 * 0.5;

    target_L3 = std::max((size_t)4096, (size_t)std::pow(2.0, l3 * n));
    target_L2 = std::max((size_t)8192, (size_t)std::pow(2.0, l2 * n));
    target_L1 = std::max((size_t)8192, (size_t)std::pow(2.0, l1 * n));

    const double delta_L3 = std::log2((double)target_L3) - l3 * n;
    const double delta_L2 = std::log2((double)target_L2) - l2 * n;
    const double delta_L1 = std::log2((double)target_L1) - l1 * n;

    int b2_raw = std::max(1, std::min(28, (int)std::lround(r2 * n + 2.0 * delta_L3 - delta_L2)));
    int b1_raw = std::max(b2_raw + 2, std::min(58, (int)std::lround(r1 * n + 3.0 * delta_L2 - delta_L1)));

    b2_raw += b2_delta;

    int usable_bits = (total_weight_sum == 0)
        ? 128
        : bit_width_u128(2 * total_weight_sum + 1);

    const int sum_range_bits = (total_weight_sum == 0)
        ? 128
        : bit_width_u128(total_weight_sum) - 1;

    b2 = std::max(1, std::min(b2_raw, usable_bits));

    int b1_upper = std::min(b1_raw, std::max(b2 + 2, sum_range_bits));
    b1 = std::max(b2 + 2, std::min(b1_upper, usable_bits));
}

TerParams ter_default_params(const std::vector<ter_u128>& weights) {
    TerParams p{};
    p.n = (int)weights.size();

    ter_u128 sum = 0;
    for (ter_u128 w : weights) {
        ter_u128 next = sum + w;
        if (next < sum) { sum = ~(ter_u128)0; break; }
        sum = next;
    }
    p.total_weight_sum = sum;

    p.eps01 = 0.0066;
    p.eps11 = 0.0024;
    p.eps02 = 0.0004;
    p.eps12 = 0.0001;
    p.eps22 = 0.0000;

    p.max_restarts = 0;
    p.timeout_seconds = 0.0;
    p.fixed_runs = 0;
    p.verbose = true;

    p.compute_derived();
    return p;
}

static bool pick_negatives(
    const std::vector<ter_u128>& weights, int start_idx,
    const std::vector<int>& pos_sel, const std::vector<int>& remaining,
    int m, int rstart, int depth, std::vector<int>& neg_sel,
    size_t cap, std::vector<TerEntry>& pool
) {
    if (depth == m) {
        TerEntry e{};
        ter_u128 sum = 0;
        for (int x : pos_sel) {
            int gi = start_idx + x;
            if (gi < 64) e.pos_lo |= (1ULL << gi); else e.pos_hi |= (1ULL << (gi - 64));
            sum += weights[gi];
        }
        for (int x : neg_sel) {
            int gi = start_idx + x;
            if (gi < 64) e.neg_lo |= (1ULL << gi); else e.neg_hi |= (1ULL << (gi - 64));
            sum -= weights[gi];
        }
        e.psum = lo64(sum);
        pool.push_back(e);
        return pool.size() >= cap;
    }
    for (int x = rstart; x < (int)remaining.size(); ++x) {
        neg_sel[depth] = remaining[x];
        if (pick_negatives(weights, start_idx, pos_sel, remaining, m, x + 1, depth + 1, neg_sel, cap, pool))
            return true;
    }
    return false;
}

static bool pick_positives(
    const std::vector<ter_u128>& weights, int start_idx, int count,
    int p, int m, int start, int depth, std::vector<int>& pos_sel,
    size_t cap, std::vector<TerEntry>& pool
) {
    if (depth == p) {
        std::vector<char> used(count, 0);
        for (int x : pos_sel) used[x] = 1;
        std::vector<int> remaining;
        remaining.reserve(count - p);
        for (int x = 0; x < count; ++x) if (!used[x]) remaining.push_back(x);
        std::vector<int> neg_sel(m);
        return pick_negatives(weights, start_idx, pos_sel, remaining, m, 0, 0, neg_sel, cap, pool);
    }
    for (int x = start; x < count; ++x) {
        pos_sel[depth] = x;
        if (pick_positives(weights, start_idx, count, p, m, x + 1, depth + 1, pos_sel, cap, pool))
            return true;
    }
    return false;
}

static bool generate_pm_category(
    const std::vector<ter_u128>& weights, int start_idx, int count,
    int p, int m, size_t cap, std::vector<TerEntry>& pool
) {
    if (p < 0 || m < 0 || p + m > count) return false;
    if (p == 0 && m == 0) return false;
    std::vector<int> pos_sel(p);
    return pick_positives(weights, start_idx, count, p, m, 0, 0, pos_sel, cap, pool);
}

static void generate_pm_shells(
    const std::vector<ter_u128>& weights, int start_idx, int count,
    int p_center, int m_center, size_t cap, size_t target_size,
    std::vector<TerEntry>& pool
) {
    for (int radius = 0; pool.size() < target_size && radius <= 2 * count; ++radius) {
        for (int dp = -radius; dp <= radius; ++dp) {
            int dm_abs = radius - std::abs(dp);
            for (int sign : {1, -1}) {
                if (dm_abs == 0 && sign == -1) continue;
                int p = p_center + dp;
                int m = m_center + sign * dm_abs;
                if (p < 0 || m < 0 || p + m > count) continue;
                if (p == 0 && m == 0) continue;
                if (generate_pm_category(weights, start_idx, count, p, m, cap, pool))
                    return;
                if (pool.size() >= target_size) return;
            }
        }
    }
}

static void generate_half_base_pool(
    const std::vector<ter_u128>& weights,
    int start_idx, int end_idx,
    size_t target_size,
    std::vector<TerEntry>& pool,
    double w2_center, double plus_center
) {
    int count = end_idx - start_idx;
    pool.clear();
    pool.reserve(target_size * 2);
    const size_t cap = target_size * 2;

    {
        TerEntry e0{};
        pool.push_back(e0);
    }

    int p_center = (int)std::lround(plus_center * count);
    int m_center = (int)std::lround(w2_center * count);
    p_center = std::max(0, std::min(count, p_center));
    m_center = std::max(0, std::min(count, m_center));

    generate_pm_shells(weights, start_idx, count, p_center, m_center, cap, target_size, pool);
}

static void merge_level2(
    const std::vector<TerEntry>& L3_left,
    const std::vector<TerEntry>& sorted_right,
    const std::vector<uint64_t>& right_keys,
    uint64_t s2,
    int b2,
    size_t max_cap,
    std::vector<TerEntry>& L2_out
) {
    uint64_t mask_m2 = (1ULL << b2) - 1ULL;
    uint64_t target_rem = s2 & mask_m2;

    L2_out.clear();
    const uint64_t* keys = right_keys.data();
    const size_t nk = right_keys.size();
    const size_t nl = L3_left.size();

    if (nl >= 2048 && omp_get_max_threads() > 1) {
        int nthreads = omp_get_max_threads();
        std::vector<std::vector<TerEntry>> local(nthreads);
        size_t thread_cap = max_cap / (size_t)nthreads + 256;
        for (auto& v : local) v.reserve(std::min(thread_cap, (size_t)4096));

        #pragma omp parallel for schedule(dynamic, 256)
        for (int ii = 0; ii < (int)nl; ++ii) {
            int tid = omp_get_thread_num();
            if (local[tid].size() >= thread_cap) continue;
            const auto& u = L3_left[(size_t)ii];
            uint64_t u_rem = u.psum & mask_m2;
            uint64_t req_v = (target_rem >= u_rem) ? (target_rem - u_rem) : (mask_m2 + 1ULL + target_rem - u_rem);
            const uint64_t* p = std::lower_bound(keys, keys + nk, req_v);
            size_t k = (size_t)(p - keys);
            for (; k < nk && keys[k] == req_v; ++k) {
                const TerEntry& v = sorted_right[k];
                TerEntry combined{};
                combined.pos_lo = u.pos_lo | v.pos_lo;
                combined.pos_hi = u.pos_hi | v.pos_hi;
                combined.neg_lo = u.neg_lo | v.neg_lo;
                combined.neg_hi = u.neg_hi | v.neg_hi;
                combined.psum = u.psum + v.psum;
                local[tid].push_back(combined);
                if (local[tid].size() >= thread_cap) break;
            }
        }
        L2_out.reserve(std::min(max_cap, (size_t)8192));
        for (auto& v : local) for (auto& e : v) { L2_out.push_back(e); if (L2_out.size() >= max_cap) return; }
        return;
    }

    L2_out.reserve(std::min(max_cap, (size_t)8192));
    for (const auto& u : L3_left) {
        uint64_t u_rem = u.psum & mask_m2;
        uint64_t req_v = (target_rem >= u_rem) ? (target_rem - u_rem) : (mask_m2 + 1ULL + target_rem - u_rem);
        const uint64_t* p = std::lower_bound(keys, keys + nk, req_v);
        size_t k = (size_t)(p - keys);
        for (; k < nk && keys[k] == req_v; ++k) {
            const TerEntry& v = sorted_right[k];
            TerEntry combined{};
            combined.pos_lo = u.pos_lo | v.pos_lo;
            combined.pos_hi = u.pos_hi | v.pos_hi;
            combined.neg_lo = u.neg_lo | v.neg_lo;
            combined.neg_hi = u.neg_hi | v.neg_hi;
            combined.psum = u.psum + v.psum;
            L2_out.push_back(combined);
            if (L2_out.size() >= max_cap) return;
        }
    }
}

struct Level1Stats {
    double sidecar_sec = 0.0;
    size_t pairs = 0;
    size_t gpu_calls = 0, gpu_fallbacks = 0, cpu_calls = 0;
    double sum_cpu_frac = 0.0;
    size_t split_calls = 0;
};

static double g_cpu_split_frac = 0.001;
static double g_cpu_split_frac_override = -1.0;
void ter_set_cpu_split_frac_override(double frac) { g_cpu_split_frac_override = (frac >= 0.0) ? frac : -1.0; }
static double g_cpu_split_frac_active() { return (g_cpu_split_frac_override >= 0.0) ? g_cpu_split_frac_override : g_cpu_split_frac; }
static double g_cpu_split_frac_report() { return g_cpu_split_frac_active(); }
static constexpr size_t kGpuMinRows = 4096;
static constexpr double kFracMin = 0.0005, kFracMax = 0.50, kEmaAlpha = 0.5;

static bool build_level1_sidecar(
    const std::vector<TerEntry>& A, const std::vector<TerEntry>& B,
    const std::vector<TerEntry>& sorted_C, uint64_t mask_m1, int b1, Level1Sidecar& sc)
{
    sc.valid = false;
    const size_t nc = sorted_C.size();
    if (nc == 0 || nc >= (size_t)UINT32_MAX) return false;

    sc.c_keys.resize(nc);
    for (size_t k = 0; k < nc; ++k) sc.c_keys[k] = sorted_C[k].psum & mask_m1;
    if (!std::is_sorted(sc.c_keys.begin(), sc.c_keys.end())) return false;

    int bits = 0;
    while (((size_t)1 << (bits + 1)) <= nc) ++bits;
    if (bits < 1) bits = 1;
    if (bits > b1) bits = b1;
    sc.shift = b1 - bits;

    const size_t nb = (size_t)1 << bits;
    sc.start.assign(nb + 1, 0);
    for (size_t k = 0; k < nc; ++k) ++sc.start[(size_t)(sc.c_keys[k] >> sc.shift) + 1];
    for (size_t b = 0; b < nb; ++b) sc.start[b + 1] += sc.start[b];

    sc.a_ps.resize(A.size());
    sc.b_ps.resize(B.size());
    for (size_t i = 0; i < A.size(); ++i) sc.a_ps[i] = A[i].psum;
    for (size_t j = 0; j < B.size(); ++j) sc.b_ps[j] = B[j].psum;

    sc.valid = true;
    return true;
}

static void merge_level1_cpu_range(
    const std::vector<TerEntry>& A, size_t a_off, size_t a_cnt,
    const std::vector<TerEntry>& B, const std::vector<TerEntry>& sorted_C,
    uint64_t target_mod, uint64_t mask_m1, bool sc_ok, const Level1Sidecar& sc,
    size_t max_cap, std::vector<std::vector<TerEntry>>& local_outs)
{
    int nthreads = omp_get_max_threads();
    if ((int)local_outs.size() != nthreads) local_outs.resize(nthreads);
    for (auto& v : local_outs) v.clear();
    size_t thread_cap = max_cap / (size_t)nthreads + 256;
    const size_t nB = B.size();

    if (sc_ok) {
        const uint64_t* keys = sc.c_keys.data();
        const uint32_t* start = sc.start.data();
        const uint64_t* a_ps = sc.a_ps.data();
        const uint64_t* b_ps = sc.b_ps.data();
        const int shift = sc.shift;

        #pragma omp parallel for schedule(dynamic, 16)
        for (int ii = 0; ii < (int)a_cnt; ++ii) {
            size_t i_a = a_off + (size_t)ii;
            int tid = omp_get_thread_num();
            if (local_outs[tid].size() >= thread_cap) continue;
            const uint64_t a_base = target_mod - a_ps[i_a];
            const TerEntry& u = A[i_a];
            for (size_t j = 0; j < nB; ++j) {
                if (local_outs[tid].size() >= thread_cap) break;
                const TerEntry& v = B[j];
                if (((u.pos_lo & v.pos_lo) & (u.neg_lo | v.neg_lo)) ||
                    ((u.pos_hi & v.pos_hi) & (u.neg_hi | v.neg_hi))) continue;
                const uint64_t req_w = (a_base - b_ps[j]) & mask_m1;
                const size_t bucket = (size_t)(req_w >> shift);
                const uint32_t lo = start[bucket], end = start[bucket + 1];
                if (lo == end) continue;
                const uint64_t* p = std::lower_bound(keys + lo, keys + end, req_w);
                size_t k = (size_t)(p - keys);
                if (k >= end || keys[k] != req_w) continue;
                for (; k < end && keys[k] == req_w; ++k) {
                    TerEntry comb;
                    if (check_and_add_ternary(u, v, sorted_C[k], comb)) {
                        local_outs[tid].push_back(comb);
                        if (local_outs[tid].size() >= thread_cap) break;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int ii = 0; ii < (int)a_cnt; ++ii) {
            size_t i_a = a_off + (size_t)ii;
            int tid = omp_get_thread_num();
            if (local_outs[tid].size() >= thread_cap) continue;
            const TerEntry& u = A[i_a];
            for (const auto& v : B) {
                if (local_outs[tid].size() >= thread_cap) break;
                if (((u.pos_lo & v.pos_lo) & (u.neg_lo | v.neg_lo)) ||
                    ((u.pos_hi & v.pos_hi) & (u.neg_hi | v.neg_hi))) continue;
                uint64_t uv_rem = (u.psum + v.psum) & mask_m1;
                uint64_t req_w = (target_mod >= uv_rem) ? (target_mod - uv_rem) : (mask_m1 + 1ULL + target_mod - uv_rem);
                auto it = std::lower_bound(sorted_C.begin(), sorted_C.end(), req_w,
                    [mask_m1](const TerEntry& e, uint64_t v2) { return (e.psum & mask_m1) < v2; });
                while (it != sorted_C.end() && ((it->psum & mask_m1) == req_w)) {
                    TerEntry comb;
                    if (check_and_add_ternary(u, v, *it, comb)) {
                        local_outs[tid].push_back(comb);
                        if (local_outs[tid].size() >= thread_cap) break;
                    }
                    ++it;
                }
            }
        }
    }
}

static void append_capped(std::vector<TerEntry>& out, const std::vector<std::vector<TerEntry>>& local_outs, size_t max_cap) {
    for (const auto& vec : local_outs) {
        for (const auto& item : vec) {
            out.push_back(item);
            if (out.size() >= max_cap) return;
        }
    }
}

static void merge_level1(
    const std::vector<TerEntry>& A, const std::vector<TerEntry>& B, const std::vector<TerEntry>& sorted_C,
    uint64_t s1, int b1, size_t max_cap, std::vector<TerEntry>& L1_out,
    std::vector<std::vector<TerEntry>>& local_outs_scratch, Level1Sidecar& sc, Level1Stats& stats)
{
    uint64_t mask_m1 = (1ULL << b1) - 1ULL;
    uint64_t target_mod = s1 & mask_m1;
    stats.pairs += A.size() * B.size();

    bool sc_ok = false;
    if (!A.empty() && !B.empty() && !sorted_C.empty()) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        sc_ok = build_level1_sidecar(A, B, sorted_C, mask_m1, b1, sc);
        stats.sidecar_sec += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    L1_out.clear();

#ifdef WITH_GPU
    if (!A.empty() && !B.empty() && !sorted_C.empty()) {
        size_t total = A.size();
        double frac = (total < kGpuMinRows) ? 0.0 : g_cpu_split_frac_active();
        size_t cpu_rows = (size_t)(frac * (double)total);
        size_t gpu_rows = total - cpu_rows;

        bool launched = gpu_rows > 0 && gpu_l1_launch(A, B, sorted_C, target_mod, mask_m1, max_cap, sc_ok ? &sc : nullptr, 256, gpu_rows);
        if (gpu_rows > 0 && !launched) {
            if (++stats.gpu_fallbacks == 1)
                std::cerr << "[TER] PERINGATAN: gpu_l1_launch gagal, jalur CPU penuh dipakai.\n";
            cpu_rows = total; gpu_rows = 0;
        }

        const auto t_cpu0 = std::chrono::high_resolution_clock::now();
        std::vector<TerEntry> cpu_out;
        if (cpu_rows > 0) {
            merge_level1_cpu_range(A, gpu_rows, cpu_rows, B, sorted_C, target_mod, mask_m1, sc_ok, sc,
                                   max_cap, local_outs_scratch);
            append_capped(cpu_out, local_outs_scratch, max_cap);
            ++stats.cpu_calls;
        }
        const double t_cpu = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_cpu0).count();

        if (launched) {
            const auto t_gpu0 = std::chrono::high_resolution_clock::now();
            std::vector<TerEntry> gpu_out;
            if (gpu_l1_finish(gpu_out)) {
                ++stats.gpu_calls;
                const double t_gpu = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_gpu0).count();
                L1_out = std::move(gpu_out);
                for (auto& e : cpu_out) { if (L1_out.size() >= max_cap) break; L1_out.push_back(e); }

                if (g_cpu_split_frac_override < 0.0) {
                    if (cpu_rows > 0 && gpu_rows > 0 && t_cpu > 1e-6 && t_gpu > 1e-6) {
                        double rate_cpu = (double)cpu_rows / t_cpu, rate_gpu = (double)gpu_rows / t_gpu;
                        double target_frac = rate_cpu / (rate_cpu + rate_gpu);
                        g_cpu_split_frac = kEmaAlpha * target_frac + (1.0 - kEmaAlpha) * g_cpu_split_frac;
                    } else if (cpu_rows == 0 && gpu_rows > 0) {
                        g_cpu_split_frac = std::max(kFracMin, g_cpu_split_frac * 0.7);
                    }
                    g_cpu_split_frac = std::min(kFracMax, std::max(kFracMin, g_cpu_split_frac));
                }
                stats.sum_cpu_frac += (double)cpu_rows / (double)total;
                ++stats.split_calls;
                return;
            }
            gpu_l1_abort();
            if (++stats.gpu_fallbacks == 1)
                std::cerr << "[TER] PERINGATAN: gpu_l1_finish gagal; sisipan GPU dijatuhkan untuk run ini.\n";
        }
        L1_out = std::move(cpu_out);
        stats.sum_cpu_frac += (cpu_rows > 0) ? (double)cpu_rows / (double)total : 1.0;
        ++stats.split_calls;
        return;
    }
#endif
    ++stats.cpu_calls;
    merge_level1_cpu_range(A, 0, A.size(), B, sorted_C, target_mod, mask_m1, sc_ok, sc, max_cap, local_outs_scratch);
    append_capped(L1_out, local_outs_scratch, max_cap);
}

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

    std::vector<TerEntry> sorted_B = L1_B;
    std::sort(sorted_B.begin(), sorted_B.end(), [](const TerEntry& x, const TerEntry& y) {
        return x.psum < y.psum;
    });

    sorted_C_scratch = L1_C;
    std::sort(sorted_C_scratch.begin(), sorted_C_scratch.end(), [](const TerEntry& x, const TerEntry& y) {
        return x.psum < y.psum;
    });
    const auto& sorted_C = sorted_C_scratch;
    uint64_t target_u64 = lo64(target);

#ifdef WITH_GPU
    {
        Level1Sidecar sc_root;
        bool sidecar_ok = build_level1_sidecar(L1_A, L1_B, sorted_C, ~0ULL, 64, sc_root);
        bool gpu_ok = false;
        std::vector<TerEntry> gpu_candidates;
        size_t verified_count = 0;
        if (sidecar_ok) {
            gpu_ok = run_root_merge_gpu(L1_A, L1_B, sorted_C, target_u64, 64, sc_root, gpu_candidates);
        }
        if (gpu_ok) {
            for (const auto& out : gpu_candidates) {
                std::vector<size_t> cur_indices;
                ter_u128 ver_sum = 0;
                for (int i = 0; i < n; ++i) {
                    bool bit_is_one = false;
                    if (i < 64) bit_is_one = (out.pos_lo & (1ULL << i)) != 0;
                    else bit_is_one = (out.pos_hi & (1ULL << (i - 64))) != 0;
                    if (bit_is_one) { cur_indices.push_back(i); ver_sum += weights[i]; }
                }
                if (ver_sum == target) {
                    ++verified_count;
                    sol_indices = cur_indices;
                }
            }
        }
        std::cerr << "[ROOT-GPU-DEBUG] sidecar_ok=" << sidecar_ok
                  << " gpu_ok=" << gpu_ok
                  << " candidates=" << gpu_candidates.size()
                  << " verified=" << verified_count
                  << " |A|=" << L1_A.size() << " |B|=" << L1_B.size() << " |C|=" << L1_C.size() << "\n";
        if (verified_count > 0) return true;
    }
#endif

    std::atomic<bool> found{false};

    auto try_emit = [&](const TerEntry& u, const TerEntry& v, const TerEntry& w) -> bool {
        TerEntry out{};
        if (!check_root_sum(u, v, w, out)) return false;

        std::vector<size_t> cur_indices;
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

        if (ver_sum != target) return false;

        bool expected = false;
        if (found.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            #pragma omp critical
            { sol_indices = cur_indices; }
        }
        return true;
    };

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 16)
        for (int i_a = 0; i_a < (int)L1_A.size(); ++i_a) {
            if (found.load(std::memory_order_relaxed)) continue;
            const TerEntry& u = L1_A[i_a];
            const uint64_t req = target_u64 - u.psum;

            size_t i = 0;
            size_t j = sorted_C.size();
            bool local_break = false;

            while (i < sorted_B.size() && j > 0 && !local_break) {
                if (found.load(std::memory_order_relaxed)) { local_break = true; break; }

                const uint64_t vsum = sorted_B[i].psum;
                const uint64_t wsum = sorted_C[j - 1].psum;
                const uint64_t pair_sum = vsum + wsum;

                if (pair_sum < req) {
                    ++i;
                } else if (pair_sum > req) {
                    --j;
                } else {
                    
                    size_t ie = i;
                    while (ie < sorted_B.size() && sorted_B[ie].psum == vsum) ++ie;
                    size_t jb = j;
                    while (jb > 0 && sorted_C[jb - 1].psum == wsum) --jb;

                    for (size_t a = i; a < ie && !local_break; ++a)
                        for (size_t b = jb; b < j; ++b)
                            if (try_emit(u, sorted_B[a], sorted_C[b])) { local_break = true; break; }

                    i = ie;
                    j = jb;
                }
            }
        }
    }

    return found.load(std::memory_order_relaxed);
}

TerResult ter_solve(
    const std::vector<ter_u128>& weights,
    ter_u128                     target,
    const TerParams&             params
) {
    using clk = std::chrono::high_resolution_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };

    auto t_start = clk::now();

    TerResult result{};
    result.found = false;
    result.runs_attempted = 0;
    result.elapsed_seconds = 0.0;
    result.successful_runs = 0;

    int n = params.n;
    int n_half = n / 2;

    size_t target_L3 = params.target_L3;
    size_t target_L2 = params.target_L2;
    size_t target_L1 = params.target_L1;
    const size_t l2_cap = target_L2 * 2;
    
    size_t l1_cap = target_L1 * 2;
    const size_t l1_cap_max = target_L1 * 64;
    int consec_capped_fail = 0;

    const bool keep_going = params.continue_after_found &&
                            (params.fixed_runs > 0 || params.max_restarts > 0 || params.timeout_seconds > 0.0);

    if (params.verbose) {
        std::cout << "\n=======================================================\n";
        std::cout << "TER (Ternary Enumeration Representation) Solver Started\n";
        std::cout << "=======================================================\n";
        std::cout << "Instance size n : " << n << "\n";
        std::cout << "Target value    : " << ter_u128_to_string(target) << "\n";
        std::cout << "Matching bits   : b1=" << params.b1 << ", b2=" << params.b2 << "\n";
        {
            const int total_bits = bit_width_u128(params.total_weight_sum);
            const long double ratio = total_bits > params.b1
                ? std::ldexp(1.0L, total_bits - params.b1)
                : 1.0L / std::ldexp(1.0L, params.b1 - total_bits);
            if (ratio < 1.0L) {
                std::cout << "[WARN] total_sum/2^b1 = " << ratio
                          << " < 1: modulus L1 terlalu besar untuk jangkauan subset-sum,\n"
                          << "       berisiko L1 kosong dan banyak run restart "
                          << "(b1 diturunkan ke floor(log2(total)) kalau belum).\n";
            } else {
                std::cout << "Sum-wrap check  : total_sum/2^b1 >= 1 (b1=" << params.b1
                          << ", sum_bits=" << total_bits << "): subset-sum membungkus modulus L1.\n";
            }
        }
        std::cout << "Target capacities: L3=" << target_L3 << ", L2=" << target_L2 << ", L1=" << target_L1 << "\n";
#ifdef WITH_GPU
        std::cout << "Execution mode  : Hybrid CPU (OpenMP) + GPU (Tesla T4 / CUDA) for Level 1 merge\n";
#else
        std::cout << "Execution mode  : CPU (OpenMP Multithreaded)\n";
#endif
        std::cout << "Level 1 lookup  : sidecar key + bucket (auto; fallback binary search jika syarat bucket tidak terpenuhi)\n";
        std::cout << "Restart config  : fixed_runs=" << params.fixed_runs
                  << ", max_restarts=" << params.max_restarts
                  << ", timeout=" << params.timeout_seconds << "s"
                  << (keep_going ? ", mode statistik (lanjut setelah solusi ditemukan)" : "") << "\n\n";
    }

#ifdef WITH_GPU
    gpu_merge_stats_reset();
#endif

    std::vector<TerEntry> pool_left;
    std::vector<TerEntry> pool_right;
    const double plus_center = params.w2 + 1.0 / 18.0;
    const auto tg0 = clk::now();
    generate_half_base_pool(weights, 0, n_half, target_L3, pool_left, params.w2, plus_center);
    generate_half_base_pool(weights, n_half, n, target_L3, pool_right, params.w2, plus_center);
    const double t_gen_pool = secs(tg0, clk::now());

    uint64_t mask_m2_fixed = (1ULL << params.b2) - 1ULL;
    const auto ts0 = clk::now();
    std::vector<TerEntry> sorted_pool_right = pool_right;
    std::sort(sorted_pool_right.begin(), sorted_pool_right.end(),
        [mask_m2_fixed](const TerEntry& a, const TerEntry& b) {
            return ((uint64_t)a.psum & mask_m2_fixed) < ((uint64_t)b.psum & mask_m2_fixed);
        });
    const double t_sort_right = secs(ts0, clk::now());
    std::vector<uint64_t> sorted_right_keys(sorted_pool_right.size());
    for (size_t i = 0; i < sorted_pool_right.size(); ++i) sorted_right_keys[i] = sorted_pool_right[i].psum & mask_m2_fixed;

    if (params.verbose) {
        std::cout << "Generated Base Pool: Left=" << pool_left.size()
                  << " entries, Right=" << pool_right.size() << " entries\n";
        if (pool_left.size() < target_L3 || pool_right.size() < target_L3)
            std::cout << "[TER-PROFIL] PERINGATAN: pool L3 (" << pool_left.size() << " / " << pool_right.size()
                      << ") lebih kecil dari target_L3=" << target_L3
                      << " (generator kehabisan komposisi; L2/L1 kemungkinan lebih kecil dari yang dirancang)\n";
        std::cout << "[TER-PROFIL] waktu sekali-jalan: gen_pool=" << std::fixed << std::setprecision(4) << t_gen_pool
                  << "s sort_pool_right=" << t_sort_right << "s\n";
        std::cout.unsetf(std::ios::floatfield);
        std::cout << std::setprecision(6);
    }

    std::mt19937_64 rng(1337);
    int run_idx = 0;

    const uint64_t mask_m1 = (1ULL << params.b1) - 1ULL;
    const uint64_t mask_m2 = (1ULL << params.b2) - 1ULL;
    const uint64_t target_lo64 = lo64(target);

    std::vector<std::vector<TerEntry>> L2buf[2];
    for (auto& b : L2buf) b.resize(9);
    int cur_buf = 0;

    std::vector<std::vector<TerEntry>> L1(3);
    std::vector<std::vector<TerEntry>> level1_scratch[3];
    for (int j = 0; j < 3; ++j) level1_scratch[j].resize(omp_get_max_threads());
    std::vector<TerEntry> root_sorted_C_scratch;

    Level1Sidecar sidecar;
    Level1Stats l1_stats;

    double tot_l2 = 0.0, tot_sortc = 0.0, tot_l1 = 0.0, tot_root = 0.0;
    double sum_l2_avg = 0.0, sum_l1_size = 0.0;
    size_t l2_cap_hits = 0, l1_cap_hits = 0, runs_l1_empty = 0;

    double hb_next = 0.0;
    auto hb_print = [&]() {
        const double hb_elapsed = secs(t_start, clk::now());
        std::cout << "[TER-HB] t+" << std::fixed << std::setprecision(1) << hb_elapsed
                  << "s run=" << run_idx << " dalam progress (L2+L1+root).\n"
                  << std::defaultfloat;
        hb_next = hb_elapsed + params.heartbeat_seconds;
    };

    uint64_t next_s1[3] = {0, 0, 0}, next_s2[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool have_pipelined = false;
    std::thread bg_l2;
    bool bg_l2_running = false;
    double bg_l2_time = 0.0;

    auto join_l2_bg = [&]() {
        if (bg_l2_running) { bg_l2.join(); bg_l2_running = false; }
    };

    auto sample_s12 = [&](uint64_t* s1, uint64_t* s2) {
        s1[0] = rng() & mask_m1;
        s1[1] = rng() & mask_m1;
        s1[2] = (target_lo64 - s1[0] - s1[1]) & mask_m1;
        for (int k = 0; k < 3; ++k) {
            s2[3 * k + 0] = rng() & mask_m2;
            s2[3 * k + 1] = rng() & mask_m2;
            s2[3 * k + 2] = (s1[k] - s2[3 * k + 0] - s2[3 * k + 1]) & mask_m2;
        }
    };

    while (true) {
        run_idx++;
        result.runs_attempted = run_idx;

        auto run_start = clk::now();

        uint64_t s1[3], s2[9];
        double run_l2 = 0.0;
        if (have_pipelined) {
            cur_buf = 1 - cur_buf;
            if (bg_l2_running) { bg_l2.join(); bg_l2_running = false; }
            std::memcpy(s1, next_s1, sizeof(s1));
            std::memcpy(s2, next_s2, sizeof(s2));
            run_l2 = bg_l2_time;
            have_pipelined = false;
        } else {
            sample_s12(s1, s2);
            const auto p_l2_0 = clk::now();
            for (int m = 0; m < 9; ++m) {
                merge_level2(pool_left, sorted_pool_right, sorted_right_keys, s2[m], params.b2, l2_cap, L2buf[cur_buf][m]);
            }
            run_l2 = secs(p_l2_0, clk::now());
        }

        const bool may_continue =
            !(params.fixed_runs > 0 && run_idx >= params.fixed_runs) &&
            !(params.max_restarts > 0 && run_idx > params.max_restarts) &&
            !(params.timeout_seconds > 0.0 && secs(t_start, clk::now()) >= params.timeout_seconds);
        if (may_continue) {
            sample_s12(next_s1, next_s2);
            int nb = 1 - cur_buf;
            bg_l2 = std::thread([&] {
                const auto t0 = clk::now();
                for (int m = 0; m < 9; ++m)
                    merge_level2(pool_left, sorted_pool_right, sorted_right_keys, next_s2[m], params.b2, l2_cap, L2buf[nb][m]);
                bg_l2_time = secs(t0, clk::now());
            });
            bg_l2_running = true;
            have_pipelined = true;
        }

        if (params.heartbeat_seconds > 0.0 && secs(t_start, clk::now()) >= hb_next) hb_print();
        double run_sortc = 0.0, run_l1 = 0.0;
        double bg_sortc = 0.0;
        auto sort_c_fn = [&](std::vector<TerEntry>& v) {
            const auto s0 = clk::now();
            std::sort(v.begin(), v.end(), [mask_m1](const TerEntry& x, const TerEntry& y) {
                return ((uint64_t)x.psum & mask_m1) < ((uint64_t)y.psum & mask_m1);
            });
            bg_sortc += secs(s0, clk::now());
        };
        sort_c_fn(L2buf[cur_buf][2]);
        run_sortc += bg_sortc; bg_sortc = 0.0;

        std::thread bg_sort;
        bool bg_active = false;
        for (int j = 0; j < 3; ++j) {
            if (bg_active) { bg_sort.join(); bg_active = false; run_sortc += bg_sortc; bg_sortc = 0.0; }
            if (j + 1 < 3) { bg_sort = std::thread(sort_c_fn, std::ref(L2buf[cur_buf][3 * (j + 1) + 2])); bg_active = true; }
            const auto a1 = clk::now();
            merge_level1(
                L2buf[cur_buf][3 * j + 0], L2buf[cur_buf][3 * j + 1], L2buf[cur_buf][3 * j + 2],
                s1[j], params.b1, l1_cap, L1[j],
                level1_scratch[j],
                sidecar, l1_stats
            );
            const auto a2 = clk::now();
            run_l1 += secs(a1, a2);
        }
        if (bg_active) { bg_sort.join(); bg_active = false; run_sortc += bg_sortc; bg_sortc = 0.0; }

        std::vector<size_t> sol;
        const auto r0 = clk::now();
        bool found = merge_root_and_solve(L1[0], L1[1], L1[2], target, weights, n, sol, root_sorted_C_scratch);
        const double run_root = secs(r0, clk::now());

        auto now = clk::now();
        double run_sec = secs(run_start, now);
        double total_sec = secs(t_start, now);

        tot_l2 += run_l2; tot_sortc += run_sortc; tot_l1 += run_l1; tot_root += run_root;
        sum_l2_avg += (double)(L2buf[cur_buf][0].size() + L2buf[cur_buf][1].size() + L2buf[cur_buf][2].size()) / 3.0;
        sum_l1_size += (double)(L1[0].size() + L1[1].size() + L1[2].size());
        for (int m = 0; m < 9; ++m) if (L2buf[cur_buf][m].size() >= l2_cap) ++l2_cap_hits;
        for (int j = 0; j < 3; ++j) if (L1[j].size() >= l1_cap) ++l1_cap_hits;
        if (L1[0].empty() || L1[1].empty() || L1[2].empty()) ++runs_l1_empty;

        if (params.verbose) {
            std::cout << "[Run " << run_idx << "] L2 avg="
                      << (L2buf[cur_buf][0].size() + L2buf[cur_buf][1].size() + L2buf[cur_buf][2].size()) / 3
                      << ", L1=(" << L1[0].size() << ", " << L1[1].size() << ", " << L1[2].size() << ")"
                      << ", elapsed=" << run_sec << "s (total=" << total_sec << "s)"
                      << std::fixed << std::setprecision(1)
                      << " | ms: L2=" << run_l2 * 1e3 << " sortC=" << run_sortc * 1e3
                      << " L1=" << run_l1 * 1e3 << " root=" << run_root * 1e3
                      << std::defaultfloat << std::setprecision(6);
    if (found) std::cout << " --> SOLUTION FOUND!\n";
            else std::cout << " --> restart...\n";
        }

        const bool all_l1_capped_this_run =
            L1[0].size() >= l1_cap && L1[1].size() >= l1_cap && L1[2].size() >= l1_cap;
        if (!found && all_l1_capped_this_run) {
            if (++consec_capped_fail >= 2 && l1_cap < l1_cap_max) {
                const size_t old_cap = l1_cap;
                l1_cap = std::min(l1_cap_max, l1_cap * 2);
                consec_capped_fail = 0;
                if (params.verbose)
                    std::cout << "[TER-ADAPT] " << old_cap << " -> " << l1_cap
                              << ": beberapa restart beruntun penuh di L1 tanpa solusi "
                              << "(kemungkinan solusi asli terpotong akibat wrap modulus); "
                              << "l1_cap dinaikkan.\n";
            }
        } else if (found) {
            consec_capped_fail = 0;
        }

        if (found) {
            ++result.successful_runs;
            if (!result.found) {
                result.found = true;
                result.solution_indices = sol;
            }
            if (!keep_going) { join_l2_bg(); break; }
        }

        if (params.fixed_runs > 0 && run_idx >= params.fixed_runs) { join_l2_bg(); break; }
        if (params.max_restarts > 0 && run_idx > params.max_restarts) { join_l2_bg(); break; }
        if (params.timeout_seconds > 0.0 && total_sec >= params.timeout_seconds) { join_l2_bg(); break; }
    }

    auto t_end = clk::now();
    result.elapsed_seconds = secs(t_start, t_end);

    if (params.verbose) {
        const double runs = (double)result.runs_attempted;
        const double per_run_total = tot_l2 + tot_sortc + tot_l1 + tot_root;
        auto pct = [&](double x) { return per_run_total > 0.0 ? 100.0 * x / per_run_total : 0.0; };

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n[TER-PROFIL] runs=" << result.runs_attempted
                  << " sukses=" << result.successful_runs
                  << " success_rate=" << (runs > 0 ? (double)result.successful_runs / runs : 0.0) << "\n";
        std::cout << std::setprecision(1);
        std::cout << "[TER-PROFIL] rata-rata per run (ms): L2=" << 1e3 * tot_l2 / runs << " (" << pct(tot_l2) << "%)"
                  << "  sortC=" << 1e3 * tot_sortc / runs << " (" << pct(tot_sortc) << "%)"
                  << "  L1=" << 1e3 * tot_l1 / runs << " (" << pct(tot_l1) << "%)"
                  << "  root=" << 1e3 * tot_root / runs << " (" << pct(tot_root) << "%)\n";
        std::cout << "[TER-PROFIL] rata-rata ukuran: L2 avg=" << sum_l2_avg / runs
                  << "  total L1 (3 list)=" << sum_l1_size / runs
                  << "  | L2 kena cap=" << l2_cap_hits << "/" << (size_t)(9 * result.runs_attempted)
                  << "  L1 kena cap=" << l1_cap_hits << "/" << (size_t)(3 * result.runs_attempted)
                  << "  run dg L1 kosong=" << runs_l1_empty << "/" << result.runs_attempted << "\n";
        if (tot_l1 > 0.0)
            std::cout << "[TER-PROFIL] L1: " << (double)l1_stats.pairs / tot_l1 / 1e6 << " Mpasang/s (total "
                      << l1_stats.pairs << " pasangan); sidecar build=" << 1e3 * l1_stats.sidecar_sec << " ms"
                      << "; panggilan GPU=" << l1_stats.gpu_calls << " CPU=" << l1_stats.cpu_calls
                      << " fallback GPU->CPU=" << l1_stats.gpu_fallbacks << "\n";
        if (l1_stats.split_calls > 0)
            std::cout << "[TER-PROFIL] split GPU/CPU: rata-rata porsi baris A ke CPU=" << std::setprecision(3)
                      << 100.0 * l1_stats.sum_cpu_frac / (double)l1_stats.split_calls
                      << "% (adaptif EMA, sekarang=" << 100.0 * g_cpu_split_frac_report() << "%)\n";
#ifdef WITH_GPU
        {
            const GpuMergeStats g = gpu_merge_stats_get();
            std::cout << "[TER-PROFIL] GPU (" << g.calls << " panggilan sukses, " << g.pref_calls
                      << " dari prefetch P3): panggilan pertama=" << g.first_call_ms
                      << " ms (termasuk init konteks CUDA)  alloc=" << g.alloc_ms << "  h2d=" << g.h2d_ms
                      << "  kernel=" << g.kernel_ms << "  d2h=" << g.d2h_ms << "  wait=" << g.wait_ms << " ms total\n";
        }
#endif
        std::cout << std::defaultfloat << std::setprecision(6);
    }
    return result;
}