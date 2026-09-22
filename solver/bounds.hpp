#pragma once
// FASE 0.1 - Cardinality bounds.
//
// Untuk target T dan multiset elemen a[0..n-1], jika kita urutkan menurun
// (a_sorted_desc[0] >= a_sorted_desc[1] >= ... ), maka:
//   - k_min = ukuran subset TERKECIL yang secara teoretis bisa mencapai >= T,
//     yaitu jumlah k elemen TERBESAR pertama yang >= T (karena itu adalah
//     cara tercepat mencapai T dengan sesedikit mungkin elemen).
//   - k_max = ukuran subset TERBESAR yang masih bisa <= T, yaitu k terbesar
//     sehingga jumlah (n-k) elemen TERKECIL (residual) masih <= T
//     (karena itu adalah cara "termurah" mengisi n-k slot sisanya).
//
// Ini adalah bukti matematis (bukan estimasi): TIDAK ADA subset berukuran
// < k_min yang bisa mencapai T (jumlah maksimalnya sudah kurang dari T),
// dan TIDAK ADA subset berukuran > k_max yang bisa <= T (jumlah minimalnya
// sudah lebih dari T). Jadi solusi (jika ada) HARUS punya |subset| di
// dalam [k_min, k_max].
#include <vector>
#include "common.hpp"

struct CardinalityBounds {
    int k_min = -1, k_max = -1;
    int feasible_k_count = 0;
    bool exact_hit_possible = true; // false kalau target di luar [sum_min, sum_max] global
};

// a_sorted_desc HARUS sudah terurut menurun (descending).
inline CardinalityBounds compute_cardinality_bounds(const std::vector<int128>& a_sorted_desc, int128 T) {
    int n = (int)a_sorted_desc.size();
    CardinalityBounds b;

    // prefix_sum[k] = jumlah k elemen TERBESAR
    // suffix_sum[k] = jumlah k elemen TERKECIL
    std::vector<int128> prefix(n + 1, 0), suffix(n + 1, 0);
    for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + a_sorted_desc[i];
    for (int i = n - 1; i >= 0; --i) suffix[n - i] = suffix[n - i - 1] + a_sorted_desc[i];

    int128 total = prefix[n];
    if (T < 0 || T > total) {
        b.exact_hit_possible = false;
        return b;
    }

    // k_min: smallest k with prefix[k] >= T (largest-k elements can reach T)
    for (int k = 0; k <= n; ++k) {
        if (prefix[k] >= T) { b.k_min = k; break; }
    }
    // k_max: largest k with suffix[n-k] <= T (n-k smallest elements still <= T,
    // i.e. remaining budget for the k chosen elements is achievable)
    // Equivalent formulation matching the plan: suffix[n-k] <= T means the
    // (n-k) elements NOT chosen (taken as the smallest n-k) sum to <= T,
    // which is required for the complement bound. We instead directly use:
    // a subset of size k can be as small as suffix[k] (k smallest elements)
    // and as large as prefix[k] (k largest elements). Feasibility of size k
    // requires suffix[k] <= T <= prefix[k].
    for (int k = n; k >= 0; --k) {
        if (suffix[k] <= T && T <= prefix[k]) { b.k_max = k; break; }
    }
    // Recompute k_min consistently with the same feasibility condition,
    // to guarantee soundness (suffix[k] <= T <= prefix[k]).
    b.k_min = -1;
    for (int k = 0; k <= n; ++k) {
        if (suffix[k] <= T && T <= prefix[k]) { b.k_min = k; break; }
    }

    if (b.k_min == -1 || b.k_max == -1 || b.k_max < b.k_min) {
        b.feasible_k_count = 0;
        b.exact_hit_possible = (b.k_min != -1 && b.k_max != -1);
    } else {
        b.feasible_k_count = b.k_max - b.k_min + 1;
    }
    return b;
}
