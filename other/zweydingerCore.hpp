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
#include <cstring>
#include <cstdint>
#include <immintrin.h>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

// Tipe data integer 128-bit
typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;

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

// PRNG Berbasis AES-NI (Instruksi Hardware CPU Celeron N5100)
class AesPrng {
private:
    __m128i state;
    __m128i key;
    uint64_t counter;
public:
    AesPrng(uint64_t seed = 0x9e3779b97f4a7c15ULL) {
        state = _mm_set_epi64x(seed ^ 0x517cc1b727220a95ULL, seed);
        key   = _mm_set_epi64x(0x9b05688c2b3e6c1fULL, 0x1e35a7bd1a9c4fedULL);
        counter = seed + 1;
    }

    inline uint64_t next_u64() {
        __m128i c = _mm_set_epi64x(counter++, counter++);
        __m128i enc = _mm_aesenc_si128(state, key);
        state = _mm_xor_si128(enc, c);
        return (uint64_t)_mm_cvtsi128_si64(state);
    }

    inline u128 next_u128() {
        uint64_t lo = next_u64();
        uint64_t hi = next_u64();
        return ((u128)hi << 64) | lo;
    }

    inline uint64_t next_range(uint64_t max_val) {
        if (max_val <= 1) return 0;
        return next_u64() % max_val;
    }
};

// Struktur Instans SSP
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
            size_t p_tgt = line.find("Target T:");
            if (p_tgt != std::string::npos && tgt_str.empty()) {
                tgt_str = line.substr(p_tgt + 9);
            }
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

        // K-Window Bounds (Batas Kardinalitas Eksak)
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

// Vektor Bitboard Terner {-1, 0, 1}
struct BitVec {
    u128 pos_mask; // Bit koordinat bernilai +1
    u128 neg_mask; // Bit koordinat bernilai -1
    u128 sum;      // Nilai jumlahan parsial <a, x>
};

// ============================================================================
// COMPACT INDEX BUCKET HASHMAP (Model SimpleHashMap FloydZ/cryptanalysislib)
// ============================================================================
// Menyimpan HANYA indeks u16 ke elemen list, bukan menduplikasi objek 48-byte!
// Ukuran total memori hanya ~128 KB (100% muat di L2/L3 Cache Celeron N5100).
// Waktu reset = memset 16 KB (selesai dalam hitungan 0.5 microsecond).
template<size_t BUCKET_COUNT = 16384, size_t BUCKET_CAP = 4>
class CompactBucketHashMap {
public:
    static constexpr size_t MASK = BUCKET_COUNT - 1;

private:
    alignas(64) u16 table[BUCKET_COUNT * BUCKET_CAP];
    alignas(64) u8  load[BUCKET_COUNT];

public:
    CompactBucketHashMap() {
        std::memset(load, 0, sizeof(load));
    }

    inline void clear() {
        std::memset(load, 0, sizeof(load));
    }

    inline bool insert(u32 key, u16 index) {
        size_t b = key & MASK;
        u8 l = load[b];
        if (l < BUCKET_CAP) {
            table[b * BUCKET_CAP + l] = index;
            load[b] = l + 1;
            return true;
        }
        return false;
    }

    inline u8 get_load(u32 key) const {
        return load[key & MASK];
    }

    inline const u16* get_bucket(u32 key) const {
        return &table[(key & MASK) * BUCKET_CAP];
    }
};

// Laporan Hasil Eksekusi
struct ZweydingerReport {
    bool   solved = false;
    double runtime_ms = 0.0;
    double peak_ram_mb = 0.0;
    u64    trials_completed = 0;
    u64    epochs_completed = 0;
    u64    candidates_l1 = 0;
    u64    candidates_l2 = 0;
    std::vector<int>  witness_indices;
    std::vector<u128> witness_values;
    u128   witness_sum = 0;
    std::string verification_status;
};

// ============================================================================
// KELAS UTAMA: ZweydingerSspSolver
// ============================================================================
// Mengimplementasikan Trade-Off Esser-Zweydinger (Eurocrypt 2022/1329):
// 1. Base-List L0 dibuat SEKALI dan disimpan di cache (Base-List Reuse).
// 2. Setiap trial HANYA menggeser target perantara (Intermediate Targets iT).
// 3. CompactBucketHashMap menyimpan HANYA indeks (ukuran 128 KB, nol alokasi heap).
// 4. Kecepatan mencapai puluhan ribu trial/detik tanpa membebani bus DRAM.
class ZweydingerSspSolver {
public:
    static ZweydingerReport solve(const FullInstance& inst, unsigned num_threads = 0,
                                  double time_limit_s = 0.0, bool verbose = true) {
        ZweydingerReport report;
        auto start_time = std::chrono::steady_clock::now();

        if (num_threads == 0) {
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        }

        int n = inst.n;
        u128 target = inst.target;

        // Skala Constraint: 14 bit Level 1, 14 bit Level 2 (total 28 bit)
        // Menjamin keseimbangan throughput dan probabilitas lolos ke Level 3
        const int ELL1 = 14;
        const int ELL2 = 14;
        const int ELL12 = ELL1 + ELL2; // 28 bit

        const u64 MASK1 = ((u64)1 << ELL1) - 1;
        const u64 MASK12 = ((u64)1 << ELL12) - 1;

        // Base List Size: 8192 per list (8 list L0 = ~3.1 MB RAM total)
        // 100% muat di L3 cache 4 MB Intel Celeron N5100!
        const size_t BASE_SIZE = 8192;

        if (verbose) {
            std::cout << "[INFO]: Memulai Engine Zweydinger-Esser (Base-List Reuse + Cache-Resident Hash)..." << std::endl;
            std::cout << "[INFO]: Feasible K-Window: [" << inst.k_min << " .. " << inst.k_max 
                      << "] (Target K = " << inst.k << ")." << std::endl;
            std::cout << "[INFO]: Constraint Bits: Level 1 = " << ELL1 << " bits, Level 2 = " << ELL12 << " bits." << std::endl;
            std::cout << "[INFO]: Base List Size = " << BASE_SIZE << " per list (Reused across trials, 0 alloc/trial)." << std::endl;
            std::cout << "[INFO]: Active Acceleration: SSE4.1, AES-NI PRNG, POPCNT, CompactBucketHashMap." << std::endl;
        }

        std::atomic<bool> solution_found(false);
        std::atomic<u64>  total_trials(0);
        std::atomic<u64>  total_epochs(0);
        std::atomic<u64>  total_l1_cands(0);
        std::atomic<u64>  total_l2_cands(0);
        std::mutex report_mutex;

        const u64 EPOCH_TRIALS = 1024; // 1024 pergeseran target per epoch permutasi

        auto worker = [&](unsigned thread_id) {
            AesPrng rng(1337ULL + thread_id * 70001ULL + (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());

            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            std::vector<u128> A(n);

            std::vector<std::vector<BitVec>> L0(8);
            CompactBucketHashMap<16384, 4> hm_L0[4];

            int half_n = n / 2; // 48 elemen per belahan koordinat
            int base_minus = 1;
            int base_plus = 7;  // net 6 per belahan -> 12 per L1 -> total 48 pada target solusi

            auto init_epoch = [&]() {
                total_epochs.fetch_add(1, std::memory_order_relaxed);

                // 1. Acak ulang seluruh permutasi 96 elemen secara menyeluruh
                std::shuffle(perm.begin(), perm.end(), std::mt19937_64(rng.next_u64()));
                for (int i = 0; i < n; ++i) A[i] = inst.elements[perm[i]];

                // 2. Bangun 8 list dasar L0 dengan Meet-in-the-Middle koordinat disjoint
                // L0 genap (0, 2, 4, 6) beroperasi di belahan pertama [0..47]
                // L0 ganjil (1, 3, 5, 7) beroperasi di belahan kedua [48..95]
                for (int li = 0; li < 8; ++li) {
                    L0[li].clear();
                    L0[li].reserve(BASE_SIZE);
                    int offset = (li % 2 == 0) ? 0 : half_n;

                    std::vector<int> coords(half_n);
                    for (int c = 0; c < half_n; ++c) coords[c] = offset + c;

                    for (size_t iter = 0; iter < BASE_SIZE; ++iter) {
                        for (int c = 0; c < base_plus + base_minus; ++c) {
                            int swap_pos = c + (int)rng.next_range(half_n - c);
                            std::swap(coords[c], coords[swap_pos]);
                        }

                        BitVec bv;
                        bv.pos_mask = 0;
                        bv.neg_mask = 0;
                        bv.sum = 0;

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

                // 3. Bangun direct-mapped index table untuk L0 genap (0, 2, 4, 6)
                for (int p = 0; p < 4; ++p) {
                    hm_L0[p].clear();
                    for (u16 i = 0; i < (u16)L0[2 * p].size(); ++i) {
                        hm_L0[p].insert((u32)(L0[2 * p][i].sum & MASK1), i);
                    }
                }
            };

            // Bangun epoch pertama
            init_epoch();

            // Compact Index HashMaps untuk Level 2 dan Level 3 (128 KB)
            CompactBucketHashMap<16384, 4> hm2;

            std::vector<BitVec> L1_0, L1_1, L1_2, L1_3;
            L1_0.reserve(2048);
            L1_1.reserve(2048);
            L1_2.reserve(2048);
            L1_3.reserve(2048);

            std::vector<BitVec> L2_0, L2_1;
            L2_0.reserve(512);
            L2_1.reserve(512);

            u64 local_trials = 0;

            while (!solution_found.load(std::memory_order_relaxed)) {
                // Cek batas waktu setiap 256 trial
                if (time_limit_s > 0.0 && (local_trials & 0xFF) == 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration<double>(now - start_time).count() >= time_limit_s) break;
                }

                local_trials++;
                if ((local_trials & 0xFF) == 0) {
                    total_trials.fetch_add(0x100, std::memory_order_relaxed);
                }

                // Regenerasi epoch baru setiap EPOCH_TRIALS (1024 trial)
                if ((local_trials & (EPOCH_TRIALS - 1)) == 0) {
                    init_epoch();
                }

                // ============================================================
                // LEVEL 1: JOIN BASE LISTS VIA PRE-BUILT HASHMAP (ZERO INSERT!)
                // ============================================================
                u64 iT1[4];
                iT1[0] = rng.next_u64() & MASK1;
                iT1[1] = rng.next_u64() & MASK1;
                iT1[2] = rng.next_u64() & MASK1;
                iT1[3] = (((u64)target & MASK1) - ((iT1[0] + iT1[1] + iT1[2]) & MASK1)) & MASK1;

                L1_0.clear();
                L1_1.clear();
                L1_2.clear();
                L1_3.clear();

                auto probe_l0_pair = [&](int p, const std::vector<BitVec>& listB,
                                         u64 req_mod, std::vector<BitVec>& out_l1) {
                    const auto& listA = L0[2 * p];
                    const auto& hm = hm_L0[p];
                    for (u16 j = 0; j < (u16)listB.size(); ++j) {
                        const auto& b_item = listB[j];
                        u32 needed = (u32)((req_mod - b_item.sum) & MASK1);
                        u8 bucket_len = hm.get_load(needed);
                        if (bucket_len == 0) continue;

                        const u16* bucket_indices = hm.get_bucket(needed);
                        for (u8 k = 0; k < bucket_len; ++k) {
                            const auto& a_item = listA[bucket_indices[k]];

                            // Koordinat A [0..47] dan B [48..95] dijamin 100% disjoint!
                            BitVec merged;
                            merged.pos_mask = a_item.pos_mask | b_item.pos_mask;
                            merged.neg_mask = a_item.neg_mask | b_item.neg_mask;
                            merged.sum      = a_item.sum + b_item.sum;
                            out_l1.push_back(merged);

                            if (out_l1.size() >= 2048) return;
                        }
                    }
                };

                probe_l0_pair(0, L0[1], iT1[0], L1_0);
                if (L1_0.empty()) continue;

                probe_l0_pair(1, L0[3], iT1[1], L1_1);
                if (L1_1.empty()) continue;

                probe_l0_pair(2, L0[5], iT1[2], L1_2);
                if (L1_2.empty()) continue;

                probe_l0_pair(3, L0[7], iT1[3], L1_3);
                if (L1_3.empty()) continue;

                total_l1_cands.fetch_add(L1_0.size() + L1_1.size() + L1_2.size() + L1_3.size(), std::memory_order_relaxed);

                // ============================================================
                // LEVEL 2: JOIN L1 LISTS INTO L2 modulo 2^ELL12
                // ============================================================
                u64 c2[2];
                u64 lower_req = (iT1[0] + iT1[1]) & MASK1;
                c2[0] = ((rng.next_u64() & MASK12) & ~MASK1) | lower_req;
                c2[1] = (((u64)target & MASK12) - c2[0]) & MASK12;

                L2_0.clear();
                L2_1.clear();

                auto join_l1_pair = [&](const std::vector<BitVec>& listA, const std::vector<BitVec>& listB,
                                        u64 req_mod, std::vector<BitVec>& out_l2) {
                    hm2.clear();
                    u64 shift_mask = (MASK12 >> ELL1);
                    for (u16 i = 0; i < (u16)listA.size(); ++i) {
                        u32 key = (u32)((listA[i].sum >> ELL1) & shift_mask);
                        hm2.insert(key, i);
                    }

                    for (u16 j = 0; j < (u16)listB.size(); ++j) {
                        const auto& b_item = listB[j];
                        u64 a_req_full = (req_mod - (b_item.sum & MASK12)) & MASK12;
                        u32 needed_key = (u32)((a_req_full >> ELL1) & shift_mask);
                        u8 bucket_len = hm2.get_load(needed_key);
                        if (bucket_len == 0) continue;

                        const u16* bucket_indices = hm2.get_bucket(needed_key);
                        for (u8 k = 0; k < bucket_len; ++k) {
                            const auto& a_item = listA[bucket_indices[k]];

                            if ((a_item.pos_mask & b_item.pos_mask) != 0) continue;
                            if ((a_item.neg_mask & b_item.neg_mask) != 0) continue;

                            BitVec merged;
                            merged.pos_mask = (a_item.pos_mask | b_item.pos_mask) & ~(a_item.neg_mask | b_item.neg_mask);
                            merged.neg_mask = (a_item.neg_mask | b_item.neg_mask) & ~(a_item.pos_mask | b_item.pos_mask);
                            merged.sum      = a_item.sum + b_item.sum;
                            out_l2.push_back(merged);

                            if (out_l2.size() >= 512) return;
                        }
                    }
                };

                join_l1_pair(L1_0, L1_1, c2[0], L2_0);
                if (L2_0.empty()) continue;

                join_l1_pair(L1_2, L1_3, c2[1], L2_1);
                if (L2_1.empty()) continue;

                total_l2_cands.fetch_add(L2_0.size() + L2_1.size(), std::memory_order_relaxed);

                // ============================================================
                // LEVEL 3: PENCARIAN FINAL & VERIFIKASI SOLUSI (STREAM JOIN)
                // ============================================================
                hm2.clear();
                for (u16 i = 0; i < (u16)L2_0.size(); ++i) {
                    u32 key = (u32)((L2_0[i].sum >> ELL12) & 0x3FFF);
                    hm2.insert(key, i);
                }

                for (const auto& y : L2_1) {
                    if (solution_found.load(std::memory_order_relaxed)) return;
                    if (y.sum > target) continue;

                    u128 needed_sum = target - y.sum;
                    u32 needed_key = (u32)((needed_sum >> ELL12) & 0x3FFF);
                    u8 bucket_len = hm2.get_load(needed_key);
                    if (bucket_len == 0) continue;

                    const u16* bucket_indices = hm2.get_bucket(needed_key);
                    for (u8 k = 0; k < bucket_len; ++k) {
                        const auto& x = L2_0[bucket_indices[k]];

                        // 1. Cek jumlahan eksak 128-bit
                        if (x.sum + y.sum != target) continue;

                        // 2. Cek koordinat biner murni {0, 1}^n:
                        // Tidak boleh ada tabrakan +2 atau -2
                        if ((x.pos_mask & y.pos_mask) != 0) continue;
                        if ((x.neg_mask & y.neg_mask) != 0) continue;
                        // Setiap -1 harus saling dinetralkan oleh +1 dari pasangan
                        if ((x.neg_mask & ~y.pos_mask) != 0) continue;
                        if ((y.neg_mask & ~x.pos_mask) != 0) continue;

                        u128 final_sol = (x.pos_mask | y.pos_mask) & ~(x.neg_mask | y.neg_mask);

                        // 3. Hardware POPCNT Gating: Batas kardinalitas K
                        int sol_k = (int)__builtin_popcountll((u64)final_sol) + 
                                    (int)__builtin_popcountll((u64)(final_sol >> 64));

                        if (sol_k < inst.k_min || sol_k > inst.k_max) continue;

                        // 4. Verifikasi jumlahan independen
                        u128 check_sum = 0;
                        std::vector<int> sol_indices;
                        std::vector<u128> sol_values;
                        for (int bit = 0; bit < n; ++bit) {
                            if ((final_sol >> bit) & 1) {
                                int orig_idx = perm[bit];
                                sol_indices.push_back(orig_idx);
                                sol_values.push_back(inst.elements[orig_idx]);
                                check_sum += inst.elements[orig_idx];
                            }
                        }

                        if (check_sum == target) {
                            std::lock_guard<std::mutex> lock(report_mutex);
                            if (!solution_found.load()) {
                                solution_found.store(true);
                                report.solved = true;
                                report.witness_indices = sol_indices;
                                report.witness_values = sol_values;
                                report.witness_sum = check_sum;
                                report.verification_status = "100% COCOK PRESISI";
                                return;
                            }
                        }
                    }
                }
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
        report.epochs_completed = total_epochs.load();
        report.candidates_l1 = total_l1_cands.load();
        report.candidates_l2 = total_l2_cands.load();

        if (report.solved) {
            report.verification_status = (report.witness_sum == target) ? "100% COCOK PRESISI" : "VERIFIKASI GAGAL";
        } else {
            report.verification_status = "TIDAK DITEMUKAN / TIMEOUT";
        }

        return report;
    }
};
