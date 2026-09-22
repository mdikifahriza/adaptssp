#pragma once
// FASE 1 - Weight window per blok.
//
// Kalau total subset punya bobot m dari n elemen, dan kita split n elemen
// menjadi 4 blok berukuran sama (block_size = n/4) TANPA memilih blok
// berdasarkan nilai (yaitu split "netral"/arbitrer terhadap subset target),
// maka jumlah elemen solusi yang jatuh ke satu blok tertentu mengikuti
// distribusi HYPERGEOMETRIC(populasi=n, sukses=m, sampel=block_size).
//
// mean = block_size * m / n
// var  = block_size * (m/n) * (1 - m/n) * (n - block_size) / (n - 1)
//
// Kita pakai window [mean - delta*std, mean + delta*std] (dibulatkan &
// di-clamp ke [0, block_size]) sebagai rentang bobot yang HARUS kita
// enumerasi di blok tsb, supaya representasi solusi yang sebenarnya
// (apapun split persisnya) tercakup dengan probabilitas tinggi.
#include <cmath>
#include <algorithm>

struct WeightWindow {
    int w_lo, w_hi;
};

inline WeightWindow window_for_block(int m, int n_total, int block_size, double delta_std) {
    double p = (double)m / (double)n_total;
    double mean = block_size * p;
    double var = 0.0;
    if (n_total > 1) {
        var = block_size * p * (1.0 - p) * (double)(n_total - block_size) / (double)(n_total - 1);
    }
    double sd = std::sqrt(std::max(0.0, var));
    int lo = (int)std::floor(mean - delta_std * sd);
    int hi = (int)std::ceil(mean + delta_std * sd);
    lo = std::max(0, lo);
    hi = std::min(block_size, hi);
    if (hi < lo) std::swap(lo, hi); // safety, shouldn't happen
    return {lo, hi};
}

// Peluang massa (perkiraan normal-approx) yang TIDAK tercakup oleh window,
// dipakai untuk melaporkan coverage secara jujur ke pengguna (bukan diam-diam
// berasumsi 100% aman). Dua sisi ekor di luar +-delta_std.
inline double tail_probability_outside(double delta_std) {
    // 1 - erf(delta/sqrt(2)) approx via erfc
    double x = delta_std / std::sqrt(2.0);
    return std::erfc(x); // P(|Z| > delta_std) for standard normal
}
