#pragma once
// FASE 0.2 - ResidueCascadeSieve (versi int128).
//
// Ide: untuk sebuah bilangan prima kecil p dan target T, sebuah elemen a_i
// HANYA bisa menjadi bagian solusi kalau ADA subset berukuran (k-1), k di
// [k_min,k_max], dari elemen-elemen LAIN yang jumlahnya == (T - a_i) mod p.
// Kalau untuk SEMUA k di window itu tidak ada subset seperti itu (mod p),
// maka a_i TIDAK MUNGKIN dipakai solusi -- ini bukti matematis (necessary
// condition mod p), bukan heuristik, jadi elemen tsb aman dieliminasi TANPA
// mengorbankan completeness.
//
// Representasi reachability dipakai bitmask 64-bit per hitungan elemen
// (cocok untuk p <= 63), dengan operasi rotasi sirkular sepanjang p bit
// untuk mensimulasikan "+a_j mod p" pada seluruh bitmask residu sekaligus.
#include <vector>
#include <cstdint>
#include "common.hpp"

inline uint64_t rotl_p(uint64_t bits, int shift, int p) {
    if (shift == 0) return bits;
    uint64_t mask = (p >= 64) ? ~0ULL : ((1ULL << p) - 1);
    bits &= mask;
    shift %= p;
    uint64_t rotated = ((bits << shift) | (bits >> (p - shift))) & mask;
    return rotated;
}

struct SieveResult {
    std::vector<char> keep; // keep[i] = 1 kalau elemen i masih mungkin dipakai
    int eliminated_count = 0;
    bool proved_unsat = false; // true kalau SEMUA elemen tereliminasi untuk suatu prima -> UNSAT terbukti
};

// a: seluruh elemen (urutan asli, index penting untuk pemetaan solusi).
// T: target. k_min,k_max: window kardinalitas (dari compute_cardinality_bounds).
// primes: daftar prima kecil (<=61 supaya muat di uint64_t bitmask).
inline SieveResult residue_cascade_sieve(const std::vector<int128>& a, int128 T,
                                          int k_min, int k_max,
                                          const std::vector<int>& primes) {
    int n = (int)a.size();
    SieveResult res;
    res.keep.assign(n, 1);
    if (k_min < 0 || k_max < 0 || k_max < k_min) return res;

    for (int p : primes) {
        if (p > 61) continue; // batasi supaya bitmask 64-bit cukup
        std::vector<uint32_t> a_mod(n);
        for (int j = 0; j < n; ++j) a_mod[j] = (uint32_t)((uint64_t)(((a[j] % p) + p) % p));
        uint32_t target_res = (uint32_t)((uint64_t)(((T % p) + p) % p));

        for (int i = 0; i < n; ++i) {
            if (!res.keep[i]) continue; // sudah tereliminasi prima sebelumnya
            // DP reachability atas elemen selain i, dp[c] = bitmask residu
            // yang bisa dicapai dengan TEPAT c elemen (dari elemen != i).
            std::vector<uint64_t> dp(k_max + 1, 0ULL);
            dp[0] = 1ULL; // residu 0 dicapai dengan 0 elemen
            int cur_max_c = 0;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                int shift = (int)a_mod[j];
                int upper = std::min(cur_max_c, k_max - 1);
                for (int c = upper; c >= 0; --c) {
                    if (dp[c]) {
                        dp[c + 1] |= rotl_p(dp[c], shift, p);
                    }
                }
                if (cur_max_c < k_max) cur_max_c++;
            }
            // needed residue among the OTHER elements: (target_res - a_mod[i]) mod p
            uint32_t need = (target_res + (uint32_t)p - a_mod[i] % p) % p;
            bool possible = false;
            for (int k = k_min; k <= k_max; ++k) {
                int cprev = k - 1; // count among the "other" n-1 elements
                if (cprev < 0 || cprev > k_max) continue;
                if (dp[cprev] & (1ULL << need)) { possible = true; break; }
            }
            if (!possible) {
                res.keep[i] = 0;
                res.eliminated_count++;
            }
        }
    }

    int remaining = 0;
    for (int i = 0; i < n; ++i) if (res.keep[i]) remaining++;
    if (remaining < k_min) res.proved_unsat = true; // not enough survivors to reach k_min

    return res;
}
