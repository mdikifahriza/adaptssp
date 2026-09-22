#pragma once

// Explorer solusi subset-sum lewat "zero-sum swap" (BFS bertingkat), native untuk project ini.
// Rencana: rencana.md §4.2 dan §9. Tidak bergantung pada solver_header.h milik program lain.
//
// Ide: dari satu solusi awal S (himpunan indeks), solusi baru didapat dengan
//   - membuang subset I dari S (I subset S, |I| <= max_swap_size), dan
//   - menambah subset O dari luar S (O disjoint dari S, |O| <= max_swap_size),
// dengan sum(I) == sum(O). Jumlah total tetap sama dengan target.
//   Tier 0 = solusi awal; Tier k = solusi baru hasil swap dari Tier k-1; berhenti saat
//   satu tier tidak menghasilkan solusi baru (closure) atau salah satu batas tercapai.
//
// BATAS (harus disampaikan ke pengguna):
//   - Yang dijelajahi hanya solusi yang TERHUBUNG dari solusi awal lewat rantai swap
//     berukuran <= max_swap_size per sisi. Closure BUKAN bukti bahwa tidak ada solusi lain.
//   - Biaya per node ~ C(|S|, <=m) + C(n-|S|, <=m). max_solutions dan max_subsets_per_side
//     adalah pengaman keras; kalau kena, hasil ditandai capped (bukan closure).
//
// Ketepatan: semua jumlah subset dihitung eksak dalam u128 (dijamin tidak overflow karena
// total seluruh nilai dicek <= 2^128-1 di awal), pencocokan memakai kesamaan jumlah eksak,
// dan setiap kandidat tetap DIVERIFIKASI ulang dari nilai asli (sum == target) sebelum
// diterima. Solusi awal juga divalidasi (indeks unik, dalam rentang, sum == target).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using u128 = unsigned __int128;

constexpr int kMaxSwapSize = 6;

struct SwapWitness
{
    std::vector<size_t> indices; // terurut naik, unik
    u128 sum = 0;                // presisi penuh, selalu == target
};

struct SwapTierInfo
{
    int tier;
    size_t new_solutions;
    size_t cumulative_total;
};

struct SwapExploreParams
{
    int max_swap_size = 4;                     // 1..kMaxSwapSize, ukuran maksimum subset per sisi
    int max_tiers = 100000;                    // pengaman jumlah generasi
    size_t max_solutions = 1000;               // termasuk solusi awal; 0 = tanpa batas
    size_t max_subsets_per_side = 5000000;     // pengaman memori/waktu per node
    bool verbose = true;
};

struct SwapExploreResult
{
    std::vector<SwapWitness> all_solutions; // [0] = solusi awal
    std::vector<SwapTierInfo> tiers;
    int tiers_run = 0;
    bool capped = true;        // false HANYA bila closure tercapai tanpa node yang dilewati
    size_t skipped_nodes = 0;  // node dilewati karena jumlah subset > max_subsets_per_side
    size_t verify_failures = 0; // kandidat lolos pencocokan tapi gagal verifikasi (seharusnya 0)
    std::string stop_reason;
    std::string error;         // tidak kosong => explorer tidak dijalankan
};

namespace swap_detail
{

struct Sub
{
    u128 sum;
    uint32_t pos[kMaxSwapSize]; // posisi di dalam pool sisi tsb (bukan indeks asli)
    uint8_t cnt;
};

// true bila sum_{k=0..m} C(n,k) <= limit
inline bool subset_count_within(size_t n, int m, size_t limit)
{
    u128 total = 1, term = 1;
    for (int k = 1; k <= m && static_cast<size_t>(k) <= n; ++k)
    {
        term = term * (n - static_cast<size_t>(k) + 1) / static_cast<size_t>(k);
        total += term;
        if (total > limit)
            return false;
    }
    return total <= limit;
}

inline void gen_subsets(const std::vector<u128> &pool_vals, size_t start, int depth, int max_depth,
                        u128 sum, Sub &cur, std::vector<Sub> &out)
{
    cur.sum = sum;
    cur.cnt = static_cast<uint8_t>(depth);
    out.push_back(cur);
    if (depth == max_depth)
        return;
    for (size_t j = start; j < pool_vals.size(); ++j)
    {
        cur.pos[depth] = static_cast<uint32_t>(j);
        gen_subsets(pool_vals, j + 1, depth + 1, max_depth, sum + pool_vals[j], cur, out);
    }
}

// Mengembalikan true bila max_solutions tercapai (proses harus berhenti).
inline bool swap_from_one(const std::vector<u128> &values, u128 target, const SwapWitness &base,
                          const SwapExploreParams &P, std::set<std::vector<size_t>> &seen,
                          SwapExploreResult &R, std::vector<SwapWitness> &newly)
{
    const size_t n = values.size();
    const int m = P.max_swap_size;

    const std::vector<size_t> &in_idx = base.indices;
    std::vector<size_t> out_idx;
    out_idx.reserve(n - in_idx.size());
    {
        size_t p = 0;
        for (size_t i = 0; i < n; ++i)
        {
            if (p < in_idx.size() && in_idx[p] == i)
                ++p;
            else
                out_idx.push_back(i);
        }
    }

    if (!subset_count_within(in_idx.size(), m, P.max_subsets_per_side) ||
        !subset_count_within(out_idx.size(), m, P.max_subsets_per_side))
    {
        ++R.skipped_nodes;
        return false;
    }

    std::vector<u128> in_vals, out_vals;
    in_vals.reserve(in_idx.size());
    out_vals.reserve(out_idx.size());
    for (size_t id : in_idx)
        in_vals.push_back(values[id]);
    for (size_t id : out_idx)
        out_vals.push_back(values[id]);

    std::vector<Sub> A, B;
    Sub cur{};
    gen_subsets(in_vals, 0, 0, m, 0, cur, A);
    gen_subsets(out_vals, 0, 0, m, 0, cur, B);
    auto by_sum = [](const Sub &a, const Sub &b) { return a.sum < b.sum; };
    std::sort(A.begin(), A.end(), by_sum);
    std::sort(B.begin(), B.end(), by_sum);

    auto emit = [&](const Sub &I, const Sub &O) -> bool
    {
        std::vector<char> removed(in_idx.size(), 0);
        for (int t = 0; t < I.cnt; ++t)
            removed[I.pos[t]] = 1;

        std::vector<size_t> ni;
        ni.reserve(in_idx.size() - I.cnt + O.cnt);
        for (size_t t = 0; t < in_idx.size(); ++t)
            if (!removed[t])
                ni.push_back(in_idx[t]);
        for (int t = 0; t < O.cnt; ++t)
            ni.push_back(out_idx[O.pos[t]]);
        std::sort(ni.begin(), ni.end());

        if (seen.count(ni))
            return false;

        u128 s = 0;
        for (size_t id : ni)
            s += values[id];
        if (s != target)
        {
            ++R.verify_failures;
            return false;
        }

        seen.insert(ni);
        SwapWitness w;
        w.indices = std::move(ni);
        w.sum = s;
        R.all_solutions.push_back(w);
        newly.push_back(std::move(w));

        return P.max_solutions != 0 && R.all_solutions.size() >= P.max_solutions;
    };

    size_t i = 0, j = 0;
    while (i < A.size() && j < B.size())
    {
        if (A[i].sum < B[j].sum)
        {
            ++i;
        }
        else if (B[j].sum < A[i].sum)
        {
            ++j;
        }
        else
        {
            size_t ie = i;
            while (ie < A.size() && A[ie].sum == A[i].sum)
                ++ie;
            size_t je = j;
            while (je < B.size() && B[je].sum == B[j].sum)
                ++je;

            for (size_t a = i; a < ie; ++a)
                for (size_t b = j; b < je; ++b)
                {
                    if (A[a].cnt == 0 && B[b].cnt == 0)
                        continue;
                    if (emit(A[a], B[b]))
                        return true;
                }
            i = ie;
            j = je;
        }
    }
    return false;
}

} // namespace swap_detail

inline SwapExploreResult explore_zero_sum_swaps(const std::vector<u128> &values, u128 target,
                                                const std::vector<size_t> &initial_indices,
                                                const SwapExploreParams &P = SwapExploreParams())
{
    SwapExploreResult R;
    const size_t n = values.size();

    if (P.max_swap_size < 1 || P.max_swap_size > kMaxSwapSize)
    {
        R.error = "max_swap_size harus di antara 1 dan " + std::to_string(kMaxSwapSize);
        return R;
    }
    if (n > 0xFFFFFFFFull)
    {
        R.error = "jumlah elemen terlalu besar";
        return R;
    }

    u128 total = 0;
    for (u128 v : values)
    {
        const u128 next = total + v;
        if (next < total)
        {
            R.error = "jumlah total semua nilai > 2^128-1, aritmetika u128 tidak lagi eksak";
            return R;
        }
        total = next;
    }

    std::vector<size_t> init = initial_indices;
    std::sort(init.begin(), init.end());
    for (size_t t = 0; t < init.size(); ++t)
    {
        if (init[t] >= n)
        {
            R.error = "indeks solusi awal di luar rentang";
            return R;
        }
        if (t > 0 && init[t] == init[t - 1])
        {
            R.error = "solusi awal mengandung indeks ganda";
            return R;
        }
    }
    {
        u128 s = 0;
        for (size_t id : init)
            s += values[id];
        if (s != target)
        {
            R.error = "solusi awal tidak berjumlah target (verifikasi gagal)";
            return R;
        }
    }

    std::set<std::vector<size_t>> seen;
    seen.insert(init);
    SwapWitness w0;
    w0.indices = init;
    w0.sum = target;
    R.all_solutions.push_back(w0);

    if (P.verbose)
        std::cout << "[extsol] mulai dari 1 solusi awal (Tier 0), max_swap_size=" << P.max_swap_size
                  << ", max_solutions=" << (P.max_solutions == 0 ? std::string("tanpa batas") : std::to_string(P.max_solutions))
                  << "\n";

    std::deque<SwapWitness> frontier;
    frontier.push_back(w0);
    bool cap_hit = (P.max_solutions != 0 && R.all_solutions.size() >= P.max_solutions);
    int tier = 0;

    while (!frontier.empty() && !cap_hit && tier < P.max_tiers)
    {
        const size_t before = R.all_solutions.size();
        std::deque<SwapWitness> next;

        for (const SwapWitness &w : frontier)
        {
            std::vector<SwapWitness> newly;
            const bool stop = swap_detail::swap_from_one(values, target, w, P, seen, R, newly);
            for (SwapWitness &nw : newly)
                next.push_back(std::move(nw));
            if (stop)
            {
                cap_hit = true;
                break;
            }
        }

        ++tier;
        R.tiers.push_back({tier, R.all_solutions.size() - before, R.all_solutions.size()});
        if (P.verbose)
            std::cout << "[extsol]   Tier " << tier << ": solusi baru = " << (R.all_solutions.size() - before)
                      << "  (total kumulatif = " << R.all_solutions.size() << ")\n";
        frontier = std::move(next);
    }
    R.tiers_run = tier;

    if (cap_hit)
    {
        R.capped = true;
        R.stop_reason = "max_solutions tercapai";
    }
    else if (!frontier.empty())
    {
        R.capped = true;
        R.stop_reason = "max_tiers tercapai";
    }
    else if (R.skipped_nodes > 0)
    {
        R.capped = true;
        R.stop_reason = std::to_string(R.skipped_nodes) +
                        " node dilewati karena jumlah subset melebihi max_subsets_per_side";
    }
    else
    {
        R.capped = false;
        R.stop_reason = "closure";
    }
    return R;
}
