#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "bcjSimd.hpp"
#include "bcjDiskBuffer.hpp"

inline double get_system_peak_ram_mb() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0;
    }
    return 0.0;
#endif
}

inline std::string u128_to_dec(u128 v) {
    if (v == 0) return "0";
    std::string s;
    while (v > 0) {
        s.push_back((char)('0' + (int)(v % 10)));
        v /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

inline u128 parse_u128_dec(const std::string& str) {
    u128 val = 0;
    for (char c : str) {
        if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
    }
    return val;
}

// Struktur Instans Subset Sum
struct FullInstance {
    int n = 96;
    int k = 48;
    int k_min = 1;
    int k_max = 96;
    int feasible_k_count = 0;
    std::vector<u128> elements;
    u128 target = 0;
    u128 total_sum = 0;
    double density = 1.0;

    static bool load(const std::string& path, const std::string& tgt_override, FullInstance& inst) {
        std::ifstream infile(path);
        if (!infile.is_open()) return false;

        inst.elements.clear();
        std::string line;
        std::string tgt_str = tgt_override;

        while (std::getline(infile, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // Cari komentar yang mungkin berisi Target T
            size_t p_tgt = line.find("Target T:");
            if (p_tgt != std::string::npos && tgt_str.empty()) {
                tgt_str = line.substr(p_tgt + 9);
            }
            // Buang komentar
            size_t p_comm = std::min({line.find('#'), line.find("//"), line.find(';')});
            if (p_comm != std::string::npos) line = line.substr(0, p_comm);

            std::string cur;
            for (char c : line) {
                if (isdigit((unsigned char)c)) {
                    cur += c;
                } else if (!cur.empty()) {
                    inst.elements.push_back(parse_u128_dec(cur));
                    cur.clear();
                }
            }
            if (!cur.empty()) inst.elements.push_back(parse_u128_dec(cur));
        }

        inst.n = (int)inst.elements.size();
        if (!tgt_str.empty()) inst.target = parse_u128_dec(tgt_str);

        inst.total_sum = 0;
        u128 max_val = 0;
        for (u128 val : inst.elements) {
            inst.total_sum += val;
            if (val > max_val) max_val = val;
        }
        double max_bits = (max_val == 0) ? 1.0 : (double)std::log2((double)max_val + 1.0);
        inst.density = (inst.elements.empty()) ? 0.0 : ((double)inst.elements.size() / max_bits);

        // Hitung batas kardinalitas eksak k_min dan k_max
        std::vector<u128> sorted_A = inst.elements;
        std::sort(sorted_A.begin(), sorted_A.end(), std::greater<u128>());

        std::vector<u128> S(inst.n + 1, 0);
        for (int i = inst.n - 1; i >= 0; --i) S[i] = S[i + 1] + sorted_A[i];

        u128 cum_u = 0;
        inst.k_min = -1;
        inst.k_max = -1;
        for (int k_idx = 1; k_idx <= inst.n; ++k_idx) {
            cum_u += sorted_A[k_idx - 1];
            if (cum_u >= inst.target && inst.k_min == -1) inst.k_min = k_idx;
            u128 cum_l = S[inst.n - k_idx];
            if (cum_l <= inst.target) inst.k_max = k_idx;
        }
        if (inst.k_min == -1) inst.k_min = 1;
        if (inst.k_max == -1) inst.k_max = inst.n;
        inst.feasible_k_count = (inst.k_max >= inst.k_min) ? (inst.k_max - inst.k_min + 1) : 0;
        inst.k = (inst.k_min + inst.k_max) / 2;

        return inst.n >= 48;
    }

};

// Vektor Representasi Bitboard Terner {-1, 0, 1, 2}
struct BitVec {
    u128 pos_mask; // Bit koordinat bernilai +1 (atau bit dasar jika +2)
    u128 neg_mask; // Bit koordinat bernilai -1
    u128 plus2_mask; // Khusus Level 2 (BBSS): Bit koordinat bernilai +2
    u128 sum;      // Nilai jumlahan parsial <a, d>
};

// Laporan Hasil Eksekusi Solver
struct FullExecutionReport {
    bool   solved = false;
    double runtime_ms = 0.0;
    double peak_ram_mb = 0.0;
    double max_disk_spill_mb = 0.0;
    u64    trials_completed = 0;
    u64    total_candidates_evaluated = 0;
    std::vector<int>  witness_indices;
    std::vector<u128> witness_values;
    u128   witness_sum = 0;
    std::string verification_status;
};

class BCJFullSolver {
public:
    static FullExecutionReport solve(const FullInstance& inst, unsigned num_threads = 0,
                                     double time_limit_s = 0.0, bool verbose = true) {
        FullExecutionReport report;
        auto start_time = std::chrono::steady_clock::now();

        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }

        int n = inst.n;
        u128 target = inst.target;

        // Parameter teroptimasi dari classical.py (BBSS Asiacrypt 2020)
        // c3 ~ 24 bit, c2 ~ 51 bit, c1 ~ 76 bit
        const int ELL1 = 24;
        const int ELL2 = 27;
        const int ELL12 = ELL1 + ELL2; // 51 bit

        const u128 MASK1 = (ELL1 >= 128) ? ~(u128)0 : (((u128)1 << ELL1) - 1);
        const u128 MASK12 = (ELL12 >= 128) ? ~(u128)0 : (((u128)1 << ELL12) - 1);

        // Hitung konfigurasi base_plus dan base_minus secara dinamis dari jendela K
        int target_k = inst.k;
        int net_k_per_list = std::max(1, (target_k + 7) / 8); // 8 lists
        int base_minus = 2;
        int base_plus = net_k_per_list + base_minus;
        size_t base_list_size = 32768; // Ukuran dasar yang aman untuk RAM 8 GB

        if (verbose) {
            std::cout << "[INFO]: Memulai Engine BCJ/BBSS Penuh (SIMD SSE4.1 + AES-NI + Disk Spilling)..." << std::endl;
            std::cout << "[INFO]: Feasible K-Window: [" << inst.k_min << " .. " << inst.k_max 
                      << "] (Target K = " << inst.k << ", Net/List = " << net_k_per_list << ")." << std::endl;
            std::cout << "[INFO]: Constraint Bits: Level 1 = " << ELL1 << " bits, Level 2 = " << ELL12 << " bits." << std::endl;
            std::cout << "[INFO]: Base List Size = " << base_list_size << " per list (8 lists L0)." << std::endl;
        }

        std::atomic<bool> solution_found(false);
        std::atomic<u64>  total_trials(0);
        std::atomic<u64>  total_evals(0);
        std::mutex report_mutex;

        auto worker = [&](unsigned thread_id) {
            AesPrng rng(1337ULL + thread_id * 10007ULL + (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            DiskSpillManager spill_mgr;

            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;

            while (!solution_found) {
                if (time_limit_s > 0.0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration<double>(now - start_time).count() >= time_limit_s) break;
                }

                total_trials++;
                // 1. Acak urutan elemen
                std::shuffle(perm.begin(), perm.end(), std::mt19937_64(rng.next_u64()));

                // Array elemen terpermutasi
                std::vector<u128> A(n);
                for (int i = 0; i < n; ++i) A[i] = inst.elements[perm[i]];

                // 2. Bangun 8 List Dasar {-1, 0, 1}
                std::vector<std::vector<BitVec>> L0(8);
                for (int li = 0; li < 8; ++li) {
                    L0[li].reserve(base_list_size);
                    for (size_t iter = 0; iter < base_list_size && !solution_found; ++iter) {
                        BitVec bv;
                        bv.pos_mask = 0;
                        bv.neg_mask = 0;
                        bv.plus2_mask = 0;
                        bv.sum = 0;

                        // Pilih base_plus koordinat +1 dan base_minus koordinat -1
                        std::vector<int> coords(n);
                        for (int c = 0; c < n; ++c) coords[c] = c;
                        for (int c = 0; c < base_plus + base_minus; ++c) {
                            int swap_pos = c + (int)rng.next_range(n - c);
                            std::swap(coords[c], coords[swap_pos]);
                        }

                        for (int c = 0; c < base_plus; ++c) {
                            int bit = coords[c];
                            bv.pos_mask |= ((u128)1 << bit);
                            bv.sum += A[bit];
                        }
                        for (int c = 0; c < base_minus; ++c) {
                            int bit = coords[base_plus + c];
                            bv.neg_mask |= ((u128)1 << bit);
                            bv.sum -= A[bit];
                        }
                        L0[li].push_back(bv);
                    }
                }

                if (solution_found) break;

                // 3. LEVEL 1: Gabungkan 8 list L0 menjadi 4 list L1 modulo 2^ELL1
                // Residu acak c1 yang memenuhi c1[0] + c1[1] + c1[2] + c1[3] == target mod 2^ELL1
                u128 c1[4];
                c1[0] = rng.next_u128() & MASK1;
                c1[1] = rng.next_u128() & MASK1;
                c1[2] = rng.next_u128() & MASK1;
                c1[3] = ((target & MASK1) - ((c1[0] + c1[1] + c1[2]) & MASK1)) & MASK1;

                std::vector<std::vector<BitVec>> L1(4);
                for (int i = 0; i < 4 && !solution_found; ++i) {
                    join_level1(L0[2 * i], L0[2 * i + 1], ELL1, c1[i], L1[i], solution_found);
                }

                if (solution_found) break;

                // 4. LEVEL 2 (Inovasi BBSS): Gabungkan 4 list L1 menjadi 2 list L2 modulo 2^ELL12
                // Pertahankan digit +2, buang -2
                u128 c2[2];
                u128 lower_req = (c1[0] + c1[1]) & MASK1;
                c2[0] = ((rng.next_u128() & MASK12) & ~MASK1) | lower_req;
                c2[1] = ((target & MASK12) - c2[0]) & MASK12;

                std::vector<BitVec> L2_0, L2_1;
                join_level2_bbss(L1[0], L1[1], ELL12, c2[0], L2_0, solution_found);
                if (solution_found || L2_0.empty()) continue;

                join_level2_bbss(L1[2], L1[3], ELL12, c2[1], L2_1, solution_found);
                if (solution_found || L2_1.empty()) continue;

                // 5. LEVEL 3: Pencocokan Final menuju target T (harus biner murni {0, 1}^n)
                join_level3_final(L2_0, L2_1, target, n, perm, inst, solution_found, report_mutex, report);
            }
        };

        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) {
            workers.emplace_back(worker, t);
        }
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }

        auto end_time = std::chrono::steady_clock::now();
        report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        report.peak_ram_mb = get_system_peak_ram_mb();
        report.trials_completed = total_trials.load();
        report.solved = solution_found.load();

        if (report.solved) {
            report.verification_status = (report.witness_sum == target) ? "100% COCOK PRESISI" : "VERIFIKASI GAGAL";
        } else {
            report.verification_status = "TIDAK DITEMUKAN / TIMEOUT";
        }

        return report;
    }

private:
    // Join Level 1: Menggabungkan dua list L0 modulo 2^ELL1 (membuang +-2)
    static void join_level1(const std::vector<BitVec>& A, const std::vector<BitVec>& B,
                            int ell, u128 c, std::vector<BitVec>& out,
                            const std::atomic<bool>& solution_found) {
        u128 mask = (ell >= 128) ? ~(u128)0 : (((u128)1 << ell) - 1);

        // Urutkan A berdasarkan kunci residu
        std::vector<std::pair<u64, u32>> keyed_A(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            keyed_A[i] = {(u64)(A[i].sum & mask), (u32)i};
        }
        std::sort(keyed_A.begin(), keyed_A.end());

        for (const auto& y : B) {
            if (solution_found) return;
            u64 y_rem = (u64)(y.sum & mask);
            u64 needed = (u64)(((c - y_rem) & mask));

            auto lo = std::lower_bound(keyed_A.begin(), keyed_A.end(), std::make_pair(needed, 0u));
            for (auto it = lo; it != keyed_A.end() && it->first == needed; ++it) {
                const auto& x = A[it->second];

                // Cek apakah ada koordinat yang menjadi +2 atau -2
                // +2 terjadi jika pos_mask_x & pos_mask_y != 0
                // -2 terjadi jika neg_mask_x & neg_mask_y != 0
                if (!is_disjoint_u64((u64)x.pos_mask, (u64)y.pos_mask)) continue;
                if (!is_disjoint_u64((u64)(x.pos_mask >> 64), (u64)(y.pos_mask >> 64))) continue;
                if (!is_disjoint_u64((u64)x.neg_mask, (u64)y.neg_mask)) continue;
                if (!is_disjoint_u64((u64)(x.neg_mask >> 64), (u64)(y.neg_mask >> 64))) continue;

                BitVec res;
                // Koordinat +1 baru: (pos_x | pos_y) kecuali yang saling meniadakan dengan neg
                res.pos_mask = (x.pos_mask | y.pos_mask) & ~(x.neg_mask | y.neg_mask);
                res.neg_mask = (x.neg_mask | y.neg_mask) & ~(x.pos_mask | y.pos_mask);
                res.plus2_mask = 0;
                res.sum = x.sum + y.sum;
                out.push_back(res);
            }
        }
    }

    // Join Level 2 (BBSS): Menggabungkan list L1 modulo 2^ELL12 (mempertahankan +2, membuang -2)
    static void join_level2_bbss(const std::vector<BitVec>& A, const std::vector<BitVec>& B,
                                 int ell, u128 c, std::vector<BitVec>& out,
                                 const std::atomic<bool>& solution_found) {
        u128 mask = (ell >= 128) ? ~(u128)0 : (((u128)1 << ell) - 1);

        std::vector<std::pair<u64, u32>> keyed_A(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            keyed_A[i] = {(u64)(A[i].sum & mask), (u32)i};
        }
        std::sort(keyed_A.begin(), keyed_A.end());

        for (const auto& y : B) {
            if (solution_found) return;
            u64 y_rem = (u64)(y.sum & mask);
            u64 needed = (u64)(((c - y_rem) & mask));

            auto lo = std::lower_bound(keyed_A.begin(), keyed_A.end(), std::make_pair(needed, 0u));
            for (auto it = lo; it != keyed_A.end() && it->first == needed; ++it) {
                const auto& x = A[it->second];

                // BBSS: Buang jika menghasilkan -2 (neg_mask_x & neg_mask_y != 0)
                if (!is_disjoint_u64((u64)x.neg_mask, (u64)y.neg_mask)) continue;
                if (!is_disjoint_u64((u64)(x.neg_mask >> 64), (u64)(y.neg_mask >> 64))) continue;

                // BBSS KUNCI UTAMA: Digit +2 dipertahankan!
                u128 plus2 = x.pos_mask & y.pos_mask;
                u128 plus1 = (x.pos_mask ^ y.pos_mask) & ~(x.neg_mask | y.neg_mask);
                u128 minus1 = (x.neg_mask ^ y.neg_mask) & ~(x.pos_mask | y.pos_mask);

                BitVec res;
                res.pos_mask = plus1;
                res.plus2_mask = plus2;
                res.neg_mask = minus1;
                res.sum = x.sum + y.sum;
                out.push_back(res);
            }
        }
    }

    // Join Level 3 Final: Mencari kecocokan tepat target T (hasil wajib biner murni {0, 1}^n)
    static void join_level3_final(const std::vector<BitVec>& A, const std::vector<BitVec>& B,
                                  u128 target, int n, const std::vector<int>& perm,
                                  const FullInstance& inst,
                                  std::atomic<bool>& solution_found,
                                  std::mutex& report_mutex, FullExecutionReport& report) {
        // Flat Hash Map untuk lookup super cepat O(1)
        CompactHashMap map_A;
        map_A.init(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            map_A.insert((u64)A[i].sum, (u32)i);
        }

        for (const auto& y : B) {
            if (solution_found) return;
            if (y.sum > target) continue;

            u64 needed_u64 = (u64)(target - y.sum);
            u64 bucket = crc32_fast_hash(needed_u64) & map_A.mask_cap;
            u32 slot_idx = map_A.head[bucket];

            while (slot_idx != 0xFFFFFFFFU) {
                const auto& slot = map_A.slots[slot_idx];
                if (slot.key == needed_u64) {
                    const auto& x = A[slot.index];
                    if (x.sum + y.sum == target) {
                        // 1. Filter POPCNT Hardware Cepat: jumlah bit-1 wajib dalam [k_min, k_max]
                        u128 final_ones = (x.pos_mask | y.pos_mask | x.plus2_mask | y.plus2_mask) 
                                          & ~(x.neg_mask & y.pos_mask) & ~(y.neg_mask & x.pos_mask);
                        int active_bits = __builtin_popcountll((u64)final_ones) + __builtin_popcountll((u64)(final_ones >> 64));
                        if (active_bits < inst.k_min || active_bits > inst.k_max) goto next_slot;

                        // 2. Verifikasi koordinat: harus menghasilkan himpunan {0, 1}^n
                        // Setiap +2 di x harus dinetralkan oleh -1 di y: (+2 - 1 = +1)
                        if ((x.plus2_mask & ~y.neg_mask) != 0) goto next_slot;
                        // Setiap +2 di y harus dinetralkan oleh -1 di x: (+2 - 1 = +1)
                        if ((y.plus2_mask & ~x.neg_mask) != 0) goto next_slot;
                        // Tidak boleh ada sisa -1 yang tidak dinetralkan
                        u128 rem_neg_x = x.neg_mask & ~y.plus2_mask & ~y.pos_mask;
                        if (rem_neg_x != 0) goto next_slot;
                        u128 rem_neg_y = y.neg_mask & ~x.plus2_mask & ~x.pos_mask;
                        if (rem_neg_y != 0) goto next_slot;

                        // Solusi saksi eksak ditemukan!
                        {
                            std::unique_lock<std::mutex> lock(report_mutex);
                            if (!solution_found) {
                                solution_found = true;
                                report.witness_indices.clear();
                                report.witness_values.clear();
                                report.witness_sum = 0;

                                u128 final_ones = (x.pos_mask | y.pos_mask | x.plus2_mask | y.plus2_mask) & ~(x.neg_mask & y.pos_mask) & ~(y.neg_mask & x.pos_mask);
                                for (int bit = 0; bit < n; ++bit) {
                                    if ((final_ones >> bit) & 1) {
                                        int original_idx = perm[bit];
                                        report.witness_indices.push_back(original_idx);
                                        report.witness_values.push_back(inst.elements[original_idx]);
                                        report.witness_sum += inst.elements[original_idx];
                                    }
                                }
                                std::sort(report.witness_indices.begin(), report.witness_indices.end());
                            }
                            return;
                        }
                    }
                }
            next_slot:
                slot_idx = slot.next;
            }
        }
    }
};
