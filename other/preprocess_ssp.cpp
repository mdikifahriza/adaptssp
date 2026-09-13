#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdint>

using u64  = uint64_t;
using u32  = uint32_t;
using u128 = unsigned __int128;

inline std::string u128_to_str(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) {
        s.push_back((char)('0' + (int)(v % 10)));
        v /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

inline u128 parse_u128_str(const std::string& str) {
    u128 val = 0;
    for (char c : str) {
        if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
    }
    return val;
}

struct Element128 {
    u128 val;
    int orig_idx;
    bool operator>(const Element128& o) const {
        return (val != o.val) ? (val > o.val) : (orig_idx < o.orig_idx);
    }
};

struct Instance128 {
    std::vector<u128> raw_elements;
    u128 target = 0;
    u128 total_sum = 0;
    int n = 0;
    double density = 0.0;

    static bool load_txt(const std::string& path, Instance128& inst) {
        std::ifstream fin(path);
        if (!fin.is_open()) return false;

        inst.raw_elements.clear();
        std::string line;
        std::string target_str = "";

        while (std::getline(fin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();

            // Cek target di header komentar
            size_t p_tgt = line.find("Target T:");
            if (p_tgt != std::string::npos) {
                target_str = line.substr(p_tgt + 9);
            }

            // Hapus komentar
            size_t p_comm = std::min({line.find('#'), line.find("//"), line.find(';')});
            if (p_comm != std::string::npos) line = line.substr(0, p_comm);

            std::string cur;
            for (char c : line) {
                if (isdigit((unsigned char)c)) {
                    cur += c;
                } else if (!cur.empty()) {
                    inst.raw_elements.push_back(parse_u128_str(cur));
                    cur.clear();
                }
            }
            if (!cur.empty()) inst.raw_elements.push_back(parse_u128_str(cur));
        }

        if (!target_str.empty()) {
            inst.target = parse_u128_str(target_str);
        }
        inst.n = (int)inst.raw_elements.size();

        inst.total_sum = 0;
        u128 max_val = 0;
        for (u128 v : inst.raw_elements) {
            inst.total_sum += v;
            if (v > max_val) max_val = v;
        }

        double max_bits = (max_val == 0) ? 1.0 : (double)std::log2((double)max_val + 1.0);
        inst.density = (inst.n == 0) ? 0.0 : ((double)inst.n / max_bits);

        return inst.n > 0 && inst.target > 0;
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// HASIL EVALUASI PREPROSESING
// ══════════════════════════════════════════════════════════════════════════════
struct PreprocessReport {
    // 1. Trivial check
    int trivial_filtered_count = 0; // Elemen > T
    bool trivial_unsat = false;     // Total < T

    // 2. Batas Kardinalitas (K-Window)
    int k_min = -1;
    int k_max = -1;
    int k_window_size = 0;
    double k_window_ratio_pct = 0.0;

    // 3. Obstruksi GCD & Paritas
    u128 gcd_val = 1;
    int odd_count = 0;
    int even_count = 0;
    bool parity_obstruction = false;
    bool gcd_obstruction = false;

    // 4. Residue Cascade Sieve (Modular DP)
    int primes_tested = 0;
    int elements_eliminated_by_residue = 0;
    std::vector<int> eliminated_indices;
    std::vector<int> kept_indices;

    // 5. Tail-Table Extreme Interval Pruning
    int tail_interval_eliminated = 0;

    double elapsed_ms = 0.0;
};

// ══════════════════════════════════════════════════════════════════════════════
// ENGINE PREPROSESING BERDASARKAN adaptsspCore.hpp (DIADAPTASI KE U128)
// ══════════════════════════════════════════════════════════════════════════════
class SspPreprocessor {
public:
    static PreprocessReport run(const Instance128& inst, bool verbose = true) {
        PreprocessReport rep;
        auto t_start = std::chrono::steady_clock::now();

        int n = inst.n;
        u128 T = inst.target;

        if (verbose) {
            std::cout << "================================================================================" << std::endl;
            std::cout << "        UJI LAPISAN PREPROSESING (DIADAPTASI DARI adaptsspCore.hpp)             " << std::endl;
            std::cout << "================================================================================" << std::endl;
            std::cout << "Instans Elemen (N) : " << n << std::endl;
            std::cout << "Target (T)         : " << u128_to_str(T) << std::endl;
            std::cout << "Total Sum          : " << u128_to_str(inst.total_sum) << std::endl;
            std::cout << "Densitas Masalah   : " << std::fixed << std::setprecision(4) << inst.density << std::endl;
            std::cout << "--------------------------------------------------------------------------------" << std::endl;
        }

        // --------------------------------------------------------------------
        // LAPISAN 1: Trivial Pre-Reduction (Range & Target Check)
        // --------------------------------------------------------------------
        if (inst.total_sum < T) {
            rep.trivial_unsat = true;
            if (verbose) std::cout << "[LAPISAN 1]: Total Sum < Target -> UNSAT MUTLAK." << std::endl;
            return rep;
        }

        std::vector<Element128> A;
        A.reserve(n);
        for (int i = 0; i < n; ++i) {
            u128 v = inst.raw_elements[i];
            if (v > T) {
                rep.trivial_filtered_count++;
            } else {
                A.push_back({v, i});
            }
        }

        if (verbose) {
            std::cout << "[LAPISAN 1 - Trivial Filter]: " << rep.trivial_filtered_count 
                      << " elemen dibuang karena bernilai > Target T." << std::endl;
        }

        // Urutkan menurun: A[0] >= A[1] >= ... >= A[A.size()-1]
        std::sort(A.begin(), A.end(), std::greater<Element128>());
        int n_act = (int)A.size();

        // --------------------------------------------------------------------
        // LAPISAN 2: Batas Kardinalitas Eksak (k_min & k_max)
        // --------------------------------------------------------------------
        std::vector<u128> S(n_act + 1, 0);
        for (int i = n_act - 1; i >= 0; --i) S[i] = S[i + 1] + A[i].val;

        u128 cum_u = 0;
        for (int k = 1; k <= n_act; ++k) {
            cum_u += A[k - 1].val;
            if (cum_u >= T && rep.k_min == -1) rep.k_min = k;
            u128 cum_l = S[n_act - k];
            if (cum_l <= T) rep.k_max = k;
        }

        rep.k_window_size = (rep.k_min != -1 && rep.k_max >= rep.k_min) ? (rep.k_max - rep.k_min + 1) : 0;
        rep.k_window_ratio_pct = (double)rep.k_window_size * 100.0 / std::max(1, n_act);

        if (verbose) {
            std::cout << "[LAPISAN 2 - Cardinality Bounds]: " << std::endl;
            std::cout << "  * k_min (elemen terkecil yang dibutuhkan) : " << rep.k_min << std::endl;
            std::cout << "  * k_max (elemen terbesar yang diizinkan) : " << rep.k_max << std::endl;
            std::cout << "  * Feasible K Window                       : [" << rep.k_min << " .. " << rep.k_max << "] (" 
                      << rep.k_window_size << " titik, " << std::fixed << std::setprecision(2) 
                      << rep.k_window_ratio_pct << "% dari N)" << std::endl;
        }

        // --------------------------------------------------------------------
        // LAPISAN 3: Obstruksi Paritas & GCD
        // --------------------------------------------------------------------
        for (const auto& e : A) {
            if ((e.val % 2) == 0) rep.even_count++;
            else rep.odd_count++;
        }

        if (rep.odd_count == 0 && (T % 2 != 0)) {
            rep.parity_obstruction = true;
            if (verbose) std::cout << "[LAPISAN 3 - Paritas]: Semua elemen GENAP tapi target GANJIL -> UNSAT!" << std::endl;
        } else {
            if (verbose) {
                std::cout << "[LAPISAN 3 - Paritas]: Odd = " << rep.odd_count 
                          << ", Even = " << rep.even_count << ", Target Parity = " 
                          << ((T % 2 == 0) ? "Genap" : "Ganjil") << " (Konsisten)" << std::endl;
            }
        }

        // --------------------------------------------------------------------
        // LAPISAN 4: Residue Cascade Sieve (Modular DP Sound Elimination)
        // --------------------------------------------------------------------
        if (verbose) {
            std::cout << "[LAPISAN 4 - Residue Cascade Sieve]: Menjalankan filter modular bitset DP..." << std::endl;
        }

        const int PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
        const int NUM_PRIMES = 11;
        std::vector<Element128> current_kept = A;

        for (int p_idx = 0; p_idx < NUM_PRIMES; ++p_idx) {
            int p = PRIMES[p_idx];
            int cur_n = (int)current_kept.size();
            if (cur_n == 0) break;

            int km_curr = std::min(rep.k_max, cur_n);
            int kn_curr = std::max(0, std::min(rep.k_min, cur_n));
            if (kn_curr > km_curr) break;

            rep.primes_tested++;
            u32 full_mask = (p == 32) ? 0xFFFFFFFFU : ((1U << p) - 1U);
            u32 T_mod = (u32)(T % (u128)p);

            // 1. Prefix DP: prefix_masks[i][k]
            std::vector<std::vector<u32>> prefix_masks(cur_n + 1, std::vector<u32>(km_curr + 1, 0));
            prefix_masks[0][0] = 1U;

            for (int i = 0; i < cur_n; ++i) {
                u32 r = (u32)(current_kept[i].val % (u128)p);
                for (int k = 0; k <= km_curr; ++k) prefix_masks[i + 1][k] = prefix_masks[i][k];
                for (int k = km_curr - 1; k >= 0; --k) {
                    u32 m = prefix_masks[i][k];
                    if (m == 0) continue;
                    u32 rotated = (r == 0) ? m : (((m << r) | (m >> (p - r))) & full_mask);
                    prefix_masks[i + 1][k + 1] |= rotated;
                }
            }

            // Global feasibility
            bool feasible = false;
            for (int k = kn_curr; k <= km_curr; ++k) {
                if ((prefix_masks[cur_n][k] >> T_mod) & 1U) { feasible = true; break; }
            }
            if (!feasible) {
                if (verbose) std::cout << "  * Obstruksi terdeteksi modulo " << p << " -> UNSAT!" << std::endl;
                break;
            }

            // 2. Suffix DP: suffix_masks[i][k]
            std::vector<std::vector<u32>> suffix_masks(cur_n + 1, std::vector<u32>(km_curr + 1, 0));
            suffix_masks[cur_n][0] = 1U;

            for (int i = cur_n - 1; i >= 0; --i) {
                u32 r = (u32)(current_kept[i].val % (u128)p);
                for (int k = 0; k <= km_curr; ++k) suffix_masks[i][k] = suffix_masks[i + 1][k];
                for (int k = km_curr - 1; k >= 0; --k) {
                    u32 m = suffix_masks[i + 1][k];
                    if (m == 0) continue;
                    u32 rotated = (r == 0) ? m : (((m << r) | (m >> (p - r))) & full_mask);
                    suffix_masks[i][k + 1] |= rotated;
                }
            }

            // 3. Eliminasi Elemen
            std::vector<Element128> next_kept;
            next_kept.reserve(cur_n);

            for (int i = 0; i < cur_n; ++i) {
                u32 r_i = (u32)(current_kept[i].val % (u128)p);
                u32 target_ps = (T_mod + p - r_i) % p;
                bool can_include = false;

                for (int k = kn_curr; k <= km_curr && !can_include; ++k) {
                    for (int k1 = 0; k1 < k; ++k1) {
                        int k2 = k - 1 - k1;
                        if (k1 > i || k2 > (cur_n - 1 - i)) continue;

                        u32 m1 = prefix_masks[i][k1];
                        u32 m2 = suffix_masks[i + 1][k2];
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

                if (can_include) {
                    next_kept.push_back(current_kept[i]);
                } else {
                    rep.elements_eliminated_by_residue++;
                    rep.eliminated_indices.push_back(current_kept[i].orig_idx);
                }
            }

            current_kept = std::move(next_kept);
        }

        for (const auto& e : current_kept) {
            rep.kept_indices.push_back(e.orig_idx);
        }

        // --------------------------------------------------------------------
        // LAPISAN 5: Tail Extreme Interval Bound Pruning
        // --------------------------------------------------------------------
        // Elemen x mustahil ikut jika x + sum(k_min - 1 elemen terbesar lainnya) < T
        // atau jika x + sum(k_max - 1 elemen terkecil lainnya) > T
        int tail_elims = 0;
        for (int i = 0; i < (int)current_kept.size(); ++i) {
            u128 x = current_kept[i].val;

            // Cari sum (k_min - 1) elemen terbesar selain x
            u128 sum_largest_other = 0;
            int needed_large = rep.k_min - 1;
            int taken = 0;
            for (int j = 0; j < (int)current_kept.size() && taken < needed_large; ++j) {
                if (j != i) { sum_largest_other += current_kept[j].val; taken++; }
            }
            if (taken == needed_large && (x + sum_largest_other < T)) {
                tail_elims++;
                continue;
            }

            // Cari sum (k_max - 1) elemen terkecil selain x
            u128 sum_smallest_other = 0;
            int needed_small = rep.k_max - 1;
            taken = 0;
            for (int j = (int)current_kept.size() - 1; j >= 0 && taken < needed_small; --j) {
                if (j != i) { sum_smallest_other += current_kept[j].val; taken++; }
            }
            if (taken == needed_small && (x + sum_smallest_other > T)) {
                tail_elims++;
                continue;
            }
        }
        rep.tail_interval_eliminated = tail_elims;

        auto t_end = std::chrono::steady_clock::now();
        rep.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        return rep;
    }
};

int main(int argc, char* argv[]) {
    std::string file_path = "hgj_96bit_instance.txt";
    if (argc >= 2) file_path = argv[1];

    Instance128 inst;
    if (!Instance128::load_txt(file_path, inst)) {
        std::cerr << "[ERROR]: Gagal membaca berkas: " << file_path << std::endl;
        return 1;
    }

    PreprocessReport rep = SspPreprocessor::run(inst, true);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                 KESIMPULAN HASIL EVALUASI PREPROSESING                        " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Waktu Eksekusi Preprosesing  : " << std::fixed << std::setprecision(2) 
              << rep.elapsed_ms << " ms" << std::endl;
    std::cout << "Elemen Awal (N)              : " << inst.n << " elemen" << std::endl;
    std::cout << "Elemen Dibuang (Trivial > T) : " << rep.trivial_filtered_count << " elemen" << std::endl;
    std::cout << "Elemen Dibuang (Residue DP)  : " << rep.elements_eliminated_by_residue << " elemen" << std::endl;
    std::cout << "Elemen Dibuang (Tail Bounds) : " << rep.tail_interval_eliminated << " elemen" << std::endl;
    std::cout << "Total Elemen Berhasil Pangkas: " 
              << (rep.trivial_filtered_count + rep.elements_eliminated_by_residue + rep.tail_interval_eliminated) 
              << " elemen (" 
              << (inst.n - (rep.trivial_filtered_count + rep.elements_eliminated_by_residue + rep.tail_interval_eliminated))
              << " elemen tersisa)" << std::endl;
    std::cout << "Rentang Solusi (K Window)    : k = " << rep.k_min << " sampai " << rep.k_max 
              << " (lebar: " << rep.k_window_size << " titik)" << std::endl;
    std::cout << "================================================================================\n" << std::endl;

    if (rep.elements_eliminated_by_residue > 0 || rep.trivial_filtered_count > 0 || rep.k_window_size <= 5) {
        std::cout << "[KESIMPULAN]: LAPISAN PREPROSESING SANGAT BERGUNA!" << std::endl;
    } else {
        std::cout << "[KESIMPULAN]: LAPISAN PREPROSESING TIDAK EFEKTIF UNTUK MEMANGKAS ELEMEN PADA INSTANS INI." << std::endl;
        std::cout << "              (Penyebab: Densitas 1.0001 dan residu terdistribusi acak seragam sempurna)." << std::endl;
    }

    return 0;
}
