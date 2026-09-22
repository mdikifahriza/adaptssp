// ter_solver.cpp — TER (Ternary Enumeration Representation) SSP Solver
// Based on: Yang Li, "From Subset-Sum to Decoding: Improved Classical and
//           Quantum Algorithms via Ternary Representation Technique", Information 2025.

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
#include <iomanip>
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
// Number of bits needed to represent v (i.e. floor(log2(v))+1, 0 for v==0).
// Exact integer bit-width, done by hand instead of std::log2 because
// total_weight_sum is a 128-bit value and log2() on a double silently
// loses precision above 2^53 — fine for picking a rough magnitude, not
// fine for a bit count we're about to use as a hard clamp.
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

    // Target list sizes per level, with a hard safety floor for small n so
    // the meet-in-middle / ternary merges always have enough candidates to
    // work with. Computed ONCE here (rather than re-derived independently
    // in ter_solve()) so b1/b2 below can be kept consistent with whatever
    // sizes actually end up being used.
    target_L3 = std::max((size_t)4096, (size_t)std::pow(2.0, l3 * n));
    target_L2 = std::max((size_t)8192, (size_t)std::pow(2.0, l2 * n));
    target_L1 = std::max((size_t)8192, (size_t)std::pow(2.0, l1 * n));

    // How far the floor pushed each level's size above its theoretical
    // 2^(l*n) size, in bits (log2). Zero whenever the floor isn't active
    // (e.g. n=96, where target_L3=358397 is already far above the 4096
    // floor) — so everything below reduces exactly to the old formula for
    // instances the floor doesn't touch.
    const double delta_L3 = std::log2((double)target_L3) - l3 * n;
    const double delta_L2 = std::log2((double)target_L2) - l2 * n;
    const double delta_L1 = std::log2((double)target_L1) - l1 * n;

    // Bit matching constraints. b2 matches 2 pools of size target_L3 down to
    // target_L2 (level-3 -> level-2 MITM merge: expected survivors ~
    // target_L3^2 / 2^b2); b1 matches 3 lists of size target_L2 down to
    // target_L1 (level-2 -> level-1 ternary merge: expected survivors ~
    // target_L2^3 / 2^b1). r1*n and r2*n are calibrated against the
    // *theoretical* (unfloored) sizes 2^(l*n); when the small-n floor bumps
    // a level's actual size up by delta bits, the matching-bit count has to
    // grow by that same multiple (2x for the two pools feeding b2, 3x for
    // the three lists feeding b1) to keep the expected survivor count at
    // the next level unchanged. Previously b1/b2 were computed straight
    // from r1*n/r2*n with no such correction, so for small n (where the
    // floor is active) they came out far too small relative to the actual
    // (floor-inflated) pool sizes, and merge_level2/merge_level1 would
    // constantly hit their output caps (Bug #2).
    int b2_raw = std::max(1, std::min(28, (int)std::lround(r2 * n + 2.0 * delta_L3 - delta_L2)));
    int b1_raw = std::max(b2_raw + 2, std::min(58, (int)std::lround(r1 * n + 3.0 * delta_L2 - delta_L1)));

    // Adaptive clamp (Bug #3 fix): b1/b2 must never ask for more bits of
    // psum than this SPECIFIC instance's weights can actually produce.
    // psum values are sums of signed selections of the original weights,
    // so no partial sum can exceed +-total_weight_sum in magnitude; that
    // range needs at most usable_bits = bit_width(2*total_weight_sum+1)
    // bits to represent uniquely. Sampling/matching residues in a wider
    // space than that (what the pre-clamp b1/b2 above do for small n,
    // where the floor on target_L1/L2/L3 dominates and inflates delta_*)
    // means the uniformly-sampled residue almost never lands in the thin
    // slice of the space the real data actually occupies -> matches
    // become astronomically rare (observed as merges stuck at (0,0,0)).
    //
    // This is intentionally driven by the instance's actual magnitude,
    // not by n or any hardcoded n-range: two instances with the same n
    // but different weight scales get different clamps, and the same
    // instance always gets a clamp consistent with itself. When
    // total_weight_sum is 0 (unset / legacy caller), usable_bits is left
    // at its max (128) so the clamp is a no-op and behavior is unchanged.
    int usable_bits = (total_weight_sum == 0)
        ? 128
        : bit_width_u128(2 * total_weight_sum + 1);

    b2 = std::max(1, std::min(b2_raw, usable_bits));
    b1 = std::max(b2 + 2, std::min(b1_raw, usable_bits));
}

TerParams ter_default_params(const std::vector<ter_u128>& weights) {
    TerParams p{};
    p.n = (int)weights.size();

    // Sum the actual instance weights (full 128-bit precision) so
    // compute_derived() below can clamp b1/b2 to what this specific
    // instance can produce, instead of guessing from n alone. Weights
    // here are the original (non-negative) instance values, so this is
    // directly Σ|w_i|. Saturate instead of wrapping in the (extreme,
    // essentially never hit in practice) case the sum would overflow
    // 128 bits, since compute_derived() only cares about its bit-width.
    ter_u128 sum = 0;
    for (ter_u128 w : weights) {
        ter_u128 next = sum + w;
        if (next < sum) { sum = ~(ter_u128)0; break; } // overflow -> saturate
        sum = next;
    }
    p.total_weight_sum = sum;

    p.eps01 = 0.0066;
    p.eps11 = 0.0024;
    p.eps02 = 0.0004;
    p.eps12 = 0.0001;
    p.eps22 = 0.0000;

    p.max_restarts = 0;       // 0 = run until solution or timeout
    p.timeout_seconds = 0.0;  // 0 = no timeout
    p.fixed_runs = 0;         // 0 = auto
    p.verbose = true;

    p.compute_derived();
    return p;
}

// ─────────────────────────────────────────────────────────
// Generator kombinasi generik: SEMUA entri dengan TEPAT `p` posisi +1 dan
// `m` posisi -1 di antara `count` slot lokal (offset oleh start_idx).
// Ditambahkan ke `pool`; berhenti lebih awal begitu pool.size() >= cap.
// Menggantikan pendekatan lama (blok kode terpisah per kategori (p,m)
// yang di-hardcode) dengan satu fungsi yang berlaku untuk (p,m) berapa pun.
// ─────────────────────────────────────────────────────────
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
    if (p == 0 && m == 0) return false;  // kasus dasar ditangani terpisah
    std::vector<int> pos_sel(p);
    return pick_positives(weights, start_idx, count, p, m, 0, 0, pos_sel, cap, pool);
}

// ─────────────────────────────────────────────────────────
// Melebarkan pencarian kategori (p,m) sebagai "cincin" (shell) berjarak
// Manhattan meningkat dari (p_center, m_center) -- BUKAN dari (0,0) --
// sampai pool cukup atau seluruh ruang komposisi habis.
//
// Ini yang membuat generator ADAPTIF terhadap n: titik pusat komposisi
// "alami" vektor L3 (p_center, m_center) bergeser menjauh dari nol seiring
// n membesar (lihat catatan di generate_half_base_pool), jadi menambah
// lebih banyak kategori berbobot-rendah-dari-nol tidak akan pernah cukup
// untuk n besar. Dengan memulai dari pusat yang benar dan melebar, jumlah
// "cincin" yang dibutuhkan tetap kecil (empiris: 2-3 cincin) untuk n
// berapa pun -- tidak perlu tahu n sebelumnya, tidak perlu edit kode.
// ─────────────────────────────────────────────────────────
static void generate_pm_shells(
    const std::vector<ter_u128>& weights, int start_idx, int count,
    int p_center, int m_center, size_t cap, size_t target_size,
    std::vector<TerEntry>& pool
) {
    for (int radius = 0; pool.size() < target_size && radius <= 2 * count; ++radius) {
        for (int dp = -radius; dp <= radius; ++dp) {
            int dm_abs = radius - std::abs(dp);
            for (int sign : {1, -1}) {
                if (dm_abs == 0 && sign == -1) continue;  // hindari duplikat saat dm=0
                int p = p_center + dp;
                int m = m_center + sign * dm_abs;
                if (p < 0 || m < 0 || p + m > count) continue;
                if (p == 0 && m == 0) continue;  // sudah masuk sebagai kasus dasar
                if (generate_pm_category(weights, start_idx, count, p, m, cap, pool))
                    return;
                if (pool.size() >= target_size) return;
            }
        }
    }
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
    std::vector<TerEntry>& pool,
    double w2_center, double plus_center   // fraksi -1 dan +1 di L3, relatif thd `count` (lihat compute_derived)
) {
    int count = end_idx - start_idx;
    pool.clear();
    pool.reserve(target_size * 2);
    const size_t cap = target_size * 2;

    // ── Kategori dasar: (0,0) selalu disertakan (vektor nol). ──
    {
        TerEntry e0{};
        pool.push_back(e0);
    }

    // ── Titik pusat komposisi "alami" untuk vektor L3, DIHITUNG dari rumus
    //    paper (bukan angka hardcode) -> otomatis menyesuaikan berapa pun n.
    //    p_center = jumlah entri +1 yang diharapkan, m_center = jumlah -1.
    int p_center = (int)std::lround(plus_center * count);
    int m_center = (int)std::lround(w2_center * count);
    p_center = std::max(0, std::min(count, p_center));
    m_center = std::max(0, std::min(count, m_center));

    generate_pm_shells(weights, start_idx, count, p_center, m_center, cap, target_size, pool);
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
// Langkah 7: binary search jalan di `right_keys` (uint64 rapat) bukan langsung di
// TerEntry (40B, stride besar) -> lebih sedikit cache line disentuh per pencarian.
// right_keys = sorted_right[i].psum & mask_m2, dihitung sekali oleh caller (b2 tetap
// sepanjang run) dan dipakai ulang untuk 9 panggilan merge_level2.
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

    // Langkah 7: paralel bila L3_left cukup besar (thread luar hanya 9-way di 9
    // panggilan; ini menambah paralelisme dalam satu panggilan bila core > 9).
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
// Statistik merge_level1 per solve (langkah 4b).
struct Level1Stats {
    double sidecar_sec = 0.0;
    size_t pairs = 0;
    size_t gpu_calls = 0, gpu_fallbacks = 0, cpu_calls = 0;
    double sum_cpu_frac = 0.0;
    size_t split_calls = 0;
};

static double g_cpu_split_frac = 0.05;
static double g_cpu_split_frac_report() { return g_cpu_split_frac; }
static constexpr size_t kGpuMinRows = 4096;
static constexpr double kFracMin = 0.01, kFracMax = 0.50, kEmaAlpha = 0.5;

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

// CPU merge untuk baris A[a_off .. a_off+a_cnt). Pakai bucket bila sc_ok, else binary search.
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
            for (size_t j = 0; j < nB; ++j) {
                if (local_outs[tid].size() >= thread_cap) break;
                const uint64_t req_w = (a_base - b_ps[j]) & mask_m1;
                const size_t bucket = (size_t)(req_w >> shift);
                const uint32_t lo = start[bucket], end = start[bucket + 1];
                if (lo == end) continue;
                const uint64_t* p = std::lower_bound(keys + lo, keys + end, req_w);
                size_t k = (size_t)(p - keys);
                if (k >= end || keys[k] != req_w) continue;
                const TerEntry& u = A[i_a];
                const TerEntry& v = B[j];
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
    std::vector<std::vector<TerEntry>>& local_outs_scratch, bool use_bucket, Level1Sidecar& sc, Level1Stats& stats)
{
    uint64_t mask_m1 = (1ULL << b1) - 1ULL;
    uint64_t target_mod = s1 & mask_m1;
    stats.pairs += A.size() * B.size();

    bool sc_ok = false;
    if (use_bucket && !A.empty() && !B.empty() && !sorted_C.empty()) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        sc_ok = build_level1_sidecar(A, B, sorted_C, mask_m1, b1, sc);
        stats.sidecar_sec += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    L1_out.clear();

#ifdef WITH_GPU
    if (!A.empty() && !B.empty() && !sorted_C.empty()) {
        size_t total = A.size();
        double frac = (total < kGpuMinRows) ? 0.0 : g_cpu_split_frac;
        size_t cpu_rows = (size_t)(frac * (double)total);
        size_t gpu_rows = total - cpu_rows;

        // PENTING: a_count=gpu_rows membatasi GPU ke A[0..gpu_rows) saja, sesuai jatah
        // yang dipakai merge_level1_cpu_range di bawah untuk sisanya (A[gpu_rows..total)).
        // Sebelumnya param ini tidak ada, gpu_l1_launch selalu memproses SELURUH A --
        // untuk n besar ini membuat grid CUDA meledak (>2^31-1 blok) dan launch selalu
        // gagal (fallback penuh ke CPU), dan kalaupun grid-nya masih di bawah batas,
        // baris [gpu_rows, total) akan diproses dobel oleh CPU & GPU.
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

                if (cpu_rows > 0 && gpu_rows > 0 && t_cpu > 1e-6 && t_gpu > 1e-6) {
                    double rate_cpu = (double)cpu_rows / t_cpu, rate_gpu = (double)gpu_rows / t_gpu;
                    double target_frac = rate_cpu / (rate_cpu + rate_gpu);
                    g_cpu_split_frac = kEmaAlpha * target_frac + (1.0 - kEmaAlpha) * g_cpu_split_frac;
                } else if (cpu_rows == 0 && gpu_rows > 0) {
                    g_cpu_split_frac = std::max(kFracMin, g_cpu_split_frac * 0.7);
                }
                g_cpu_split_frac = std::min(kFracMax, std::max(kFracMin, g_cpu_split_frac));
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
    // note); target_u64 is used purely as a cheap 64-bit candidate
    // filter here. Every candidate that passes it is re-verified below
    // against the full-precision `target` by summing the original
    // ter_u128 weights, so this filter never causes false negatives —
    // it just narrows down which triples get the exact (128-bit) check.
    // Unsigned (not int64_t) so uv_sum/req_w wrap as well-defined
    // unsigned arithmetic, matching TerEntry::psum's type (Bug 1, §4.1).
    uint64_t target_u64 = lo64(target);
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

                uint64_t uv_sum = u.psum + v.psum;
                uint64_t req_w = target_u64 - uv_sum;

                auto it = std::lower_bound(sorted_C.begin(), sorted_C.end(), req_w,
                    [](const TerEntry& elem, uint64_t val) {
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

    // NOTE: target_L3/L2/L1 (incl. the small-n safety floor) are computed
    // once in TerParams::compute_derived(), together with b1/b2, so the two
    // stay mutually consistent (Bug #2 fix) — do not recompute them here.
    size_t target_L3 = params.target_L3;
    size_t target_L2 = params.target_L2;
    size_t target_L1 = params.target_L1;
    const size_t l2_cap = target_L2 * 2;
    const size_t l1_cap = target_L1 * 2;

    // Mode statistik (4b): lanjut setelah solusi pertama, tapi hanya bila ada batas run
    // yang jelas; kalau tidak, perilaku lama (berhenti di solusi pertama).
    const bool keep_going = params.continue_after_found &&
                            (params.fixed_runs > 0 || params.max_restarts > 0 || params.timeout_seconds > 0.0);

    if (params.verbose) {
        std::cout << "\n=======================================================\n";
        std::cout << "TER (Ternary Enumeration Representation) Solver Started\n";
        std::cout << "=======================================================\n";
        std::cout << "Instance size n : " << n << "\n";
        std::cout << "Target value    : " << ter_u128_to_string(target) << "\n";
        std::cout << "Matching bits   : b1=" << params.b1 << ", b2=" << params.b2 << "\n";
        std::cout << "Target capacities: L3=" << target_L3 << ", L2=" << target_L2 << ", L1=" << target_L1 << "\n";
#ifdef WITH_GPU
        std::cout << "Execution mode  : Hybrid CPU (OpenMP) + GPU (Tesla T4 / CUDA) for Level 1 merge\n";
#else
        std::cout << "Execution mode  : CPU (OpenMP Multithreaded)\n";
#endif
        std::cout << "Level 1 lookup  : " << (params.use_bucket_lookup ? "sidecar key + bucket (--ter_bucket)" : "binary search (default)") << "\n";
        std::cout << "Restart config  : fixed_runs=" << params.fixed_runs
                  << ", max_restarts=" << params.max_restarts
                  << ", timeout=" << params.timeout_seconds << "s"
                  << (keep_going ? ", mode statistik (lanjut setelah solusi ditemukan)" : "") << "\n\n";
    }

#ifdef WITH_GPU
    gpu_merge_stats_reset();
#endif

    // Generate base pools once for left and right halves.
    // params.w2 = fraksi -1 di level-2 (w^(2) dari paper); plus_center = fraksi
    // +1 sesuai definisi D^n[w^(2), 1/18]. Titik pusat komposisi L3 diturunkan
    // dari sini, bukan hardcode -- lihat generate_pm_shells di atas.
    std::vector<TerEntry> pool_left;
    std::vector<TerEntry> pool_right;
    const double plus_center = params.w2 + 1.0 / 18.0;
    const auto tg0 = clk::now();
    generate_half_base_pool(weights, 0, n_half, target_L3, pool_left, params.w2, plus_center);
    generate_half_base_pool(weights, n_half, n, target_L3, pool_right, params.w2, plus_center);
    const double t_gen_pool = secs(tg0, clk::now());

    // PERF: `pool_right` and `b2` are invariant across every restart run, so it is
    // sorted exactly once here and passed down (see merge_level2).
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

    // PERF: persistent scratch buffers reused across restarts.
    std::vector<std::vector<TerEntry>> L2(9);
    std::vector<std::vector<TerEntry>> L1(3);
    std::vector<std::vector<TerEntry>> level1_scratch[3];  // one local_outs_scratch per L1 slot
    for (int j = 0; j < 3; ++j) level1_scratch[j].resize(omp_get_max_threads());
    std::vector<TerEntry> root_sorted_C_scratch;

    Level1Sidecar sidecar;
    Level1Stats l1_stats;

    // Akumulator profil (4b)
    double tot_l2 = 0.0, tot_sortc = 0.0, tot_l1 = 0.0, tot_root = 0.0;
    double sum_l2_avg = 0.0, sum_l1_size = 0.0;
    size_t l2_cap_hits = 0, l1_cap_hits = 0, runs_l1_empty = 0;

    while (true) {
        run_idx++;
        result.runs_attempted = run_idx;

        auto run_start = clk::now();

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
        const auto p_l2_0 = clk::now();
        for (int m = 0; m < 9; ++m) {
            merge_level2(pool_left, sorted_pool_right, sorted_right_keys, s2[m], params.b2, l2_cap, L2[m]);
        }
        const double run_l2 = secs(p_l2_0, clk::now());

        // 3. Build 3 Level 1 lists from Level 2 triplets (C-list sorted by psum & mask_m1).
        double run_sortc = 0.0, run_l1 = 0.0;
        for (int j = 0; j < 3; ++j) {
            auto& C = L2[3 * j + 2];
            const auto a0 = clk::now();
            std::sort(C.begin(), C.end(), [mask_m1](const TerEntry& x, const TerEntry& y) {
                return ((uint64_t)x.psum & mask_m1) < ((uint64_t)y.psum & mask_m1);
            });
            const auto a1 = clk::now();
            merge_level1(
                L2[3 * j + 0], L2[3 * j + 1], C,
                s1[j], params.b1, l1_cap, L1[j],
                level1_scratch[j],
                params.use_bucket_lookup, sidecar, l1_stats
            );
            const auto a2 = clk::now();
            run_sortc += secs(a0, a1);
            run_l1 += secs(a1, a2);
        }

        // 4. Root search
        std::vector<size_t> sol;
        const auto r0 = clk::now();
        bool found = merge_root_and_solve(L1[0], L1[1], L1[2], target, weights, n, sol, root_sorted_C_scratch);
        const double run_root = secs(r0, clk::now());

        auto now = clk::now();
        double run_sec = secs(run_start, now);
        double total_sec = secs(t_start, now);

        // Akumulasi profil
        tot_l2 += run_l2; tot_sortc += run_sortc; tot_l1 += run_l1; tot_root += run_root;
        sum_l2_avg += (double)(L2[0].size() + L2[1].size() + L2[2].size()) / 3.0;
        sum_l1_size += (double)(L1[0].size() + L1[1].size() + L1[2].size());
        for (int m = 0; m < 9; ++m) if (L2[m].size() >= l2_cap) ++l2_cap_hits;
        for (int j = 0; j < 3; ++j) if (L1[j].size() >= l1_cap) ++l1_cap_hits;
        if (L1[0].empty() || L1[1].empty() || L1[2].empty()) ++runs_l1_empty;

        if (params.verbose) {
            std::cout << "[Run " << run_idx << "] L2 avg="
                      << (L2[0].size() + L2[1].size() + L2[2].size()) / 3
                      << ", L1=(" << L1[0].size() << ", " << L1[1].size() << ", " << L1[2].size() << ")"
                      << ", elapsed=" << run_sec << "s (total=" << total_sec << "s)"
                      << std::fixed << std::setprecision(1)
                      << " | ms: L2=" << run_l2 * 1e3 << " sortC=" << run_sortc * 1e3
                      << " L1=" << run_l1 * 1e3 << " root=" << run_root * 1e3
                      << std::defaultfloat << std::setprecision(6);
            if (found) std::cout << " --> SOLUTION FOUND!\n";
            else std::cout << " --> restart...\n";
        }

        if (found) {
            ++result.successful_runs;
            if (!result.found) {
                result.found = true;
                result.solution_indices = sol;
            }
            if (!keep_going) break;
        }

        // Stop conditions
        if (params.fixed_runs > 0 && run_idx >= params.fixed_runs) break;
        if (params.max_restarts > 0 && run_idx > params.max_restarts) break;
        if (params.timeout_seconds > 0.0 && total_sec >= params.timeout_seconds) break;
    }

    auto t_end = clk::now();
    result.elapsed_seconds = secs(t_start, t_end);

    // ── Ringkasan profil (4b). Baris berawalan [TER-PROFIL] dibuat mudah di-grep. ──
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
            std::cout << "[TER-PROFIL] GPU (" << g.calls << " panggilan sukses): panggilan pertama=" << g.first_call_ms
                      << " ms (termasuk init konteks CUDA)  alloc=" << g.alloc_ms << "  h2d=" << g.h2d_ms
                      << "  kernel=" << g.kernel_ms << "  d2h=" << g.d2h_ms << "  wait=" << g.wait_ms << " ms total\n";
        }
#endif
        std::cout << std::defaultfloat << std::setprecision(6);
    }
    return result;
}