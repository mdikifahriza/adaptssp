#pragma once
// FASE 3 - Join core.
//
// Struktur ini TIDAK berubah secara konsep dari versi awal: kita punya 4
// daftar (L1..L4), masing-masing berisi {sum, mask} dari kandidat per blok.
// Untuk mencari 4-tuple (e1,e2,e3,e4) dengan e1.sum+e2.sum+e3.sum+e4.sum==T,
// kita:
//   1) Pilih residu acak R1 (mod 2^24), set R2 = (T-R1) mod 2^24.
//   2) Bangun L12 = { (e1+e2) : (e1.sum+e2.sum) mod 2^24 == R1 } lewat
//      pencarian biner atas L1 yang sudah di-sort menurut mod.
//   3) Bangun L34 = { (e3+e4) : (e3.sum+e4.sum) mod 2^24 == R2 } serupa.
//   4) Sort L12 menurut sum eksak, lalu cari exact match T-e34.sum untuk
//      tiap e34 di L34 lewat binary search.
// Ini valid karena e1+e2+e3+e4==T  =>  (e1+e2) mod 2^24 + (e3+e4) mod 2^24
// == T mod 2^24 (mod 2^24 aritmetika), jadi filter tidak pernah membuang
// pasangan yang benar-benar valid untuk R1 yang "tepat".
//
// Catatan kelengkapan: filter modular ini SOUND (tidak pernah salah buang
// solusi yang cocok dengan R1 yang dicoba), tapi trial R1 acak artinya kita
// tidak menjelajahi semua 2^24 kemungkinan R1 -- jadi kalau max_trials
// habis tanpa hasil, itu TIDAK membuktikan UNSAT untuk m ini, hanya berarti
// belum ditemukan pada residu yang dicoba (bandingkan dengan kelengkapan
// window bobot per blok, yang sifatnya probabilistik-terkontrol, dilaporkan
// terpisah).
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include "common.hpp"

struct Element {
    int128 sum;
    uint128 mask;
};

struct ModElement {
    uint32_t mod_val;
    int128 sum;
    uint128 mask;
    bool operator<(const ModElement& other) const { return mod_val < other.mod_val; }
};

struct JoinResult {
    bool found = false;
    uint128 mask = 0;
    uint64_t trial_found_at = 0;
};

// Versi hemat memori: L1_mod dan L3_mod diterima SUDAH dalam bentuk ModElement
// (dibangun langsung dari kandidat blok tanpa melalui representasi Element
// mentah terpisah) dan SUDAH terurut. Ini menghindari menyimpan L1/L3 dua
// kali (sekali sebagai Element, sekali sebagai ModElement) yang boros
// memori saat list-nya besar (jutaan elemen, 32-48 byte/elemen).
inline JoinResult join_core_search(const std::vector<ModElement>& L1_mod,
                                    const std::vector<Element>& L2,
                                    const std::vector<ModElement>& L3_mod,
                                    const std::vector<Element>& L4,
                                    int128 T,
                                    uint64_t max_trials,
                                    uint64_t rng_seed,
                                    uint64_t mod_bits = 24) {
    JoinResult result;
    const uint64_t MOD_MASK = (mod_bits >= 64) ? ~0ULL : ((1ULL << mod_bits) - 1);

    std::mt19937_64 rng(rng_seed);

    std::vector<Element> L12, L34;

    for (uint64_t trial = 1; trial <= max_trials; ++trial) {
        uint32_t R1 = (uint32_t)(rng() & MOD_MASK);
        uint32_t R2 = (uint32_t)(((uint64_t)((T - (int128)R1) & (int128)MOD_MASK)));

        L12.clear();
        L34.clear();
        L12.reserve(L1_mod.size() / 4 + 16);
        L34.reserve(L3_mod.size() / 4 + 16);

        for (const auto& e2 : L2) {
            uint32_t target_mod = (uint32_t)((R1 - (uint32_t)((uint64_t)(e2.sum & (int128)MOD_MASK))) & MOD_MASK);
            auto range = std::equal_range(L1_mod.begin(), L1_mod.end(), ModElement{target_mod, 0, 0});
            for (auto it = range.first; it != range.second; ++it) {
                L12.push_back({it->sum + e2.sum, it->mask | e2.mask});
            }
        }
        for (const auto& e4 : L4) {
            uint32_t target_mod = (uint32_t)((R2 - (uint32_t)((uint64_t)(e4.sum & (int128)MOD_MASK))) & MOD_MASK);
            auto range = std::equal_range(L3_mod.begin(), L3_mod.end(), ModElement{target_mod, 0, 0});
            for (auto it = range.first; it != range.second; ++it) {
                L34.push_back({it->sum + e4.sum, it->mask | e4.mask});
            }
        }

        std::sort(L12.begin(), L12.end(), [](const Element& x, const Element& y) { return x.sum < y.sum; });

        for (const auto& e34 : L34) {
            int128 target_exact = T - e34.sum;
            auto it = std::lower_bound(L12.begin(), L12.end(), Element{target_exact, 0},
                                        [](const Element& x, const Element& y) { return x.sum < y.sum; });
            if (it != L12.end() && it->sum == target_exact) {
                result.found = true;
                result.mask = it->mask | e34.mask;
                result.trial_found_at = trial;
                return result;
            }
        }
    }
    return result;
}
