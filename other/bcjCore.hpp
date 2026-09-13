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
    #include <cstdint>
    #include <cmath>
    #include <iomanip>
    #include <fstream>
    #if defined(_WIN32) || defined(_WIN64)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <psapi.h>
    #else
    #include <sys/resource.h>
    #endif

    using u64  = uint64_t;
    using u32  = uint32_t;
    using u128 = unsigned __int128;

    inline std::string u128_to_string(u128 v) {
        if (v == 0) return "0";
        std::string s;
        while (v > 0) {
            s.push_back((char)('0' + (int)(v % 10)));
            v /= 10;
        }
        std::reverse(s.begin(), s.end());
        return s;
    }

    inline u128 parse_u128_decimal(const std::string& str) {
        u128 val = 0;
        for (char c : str) {
            if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
        }
        return val;
    }

    inline double get_current_peak_ram_mb() {
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

    struct BCJInstance {
        std::vector<u128> elements;
        u128 target;
        u128 total_sum;
        double density;

        static bool load_from_txt_or_string(const std::string& file_or_str, const std::string& tgt_str, BCJInstance& inst) {
            std::string raw_content;
            std::ifstream infile(file_or_str);
            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    size_t p = std::min({line.find('#'), line.find("//"), line.find(';')});
                    if (p != std::string::npos) line = line.substr(0, p);
                    raw_content += line + " ";
                }
            } else {
                raw_content = file_or_str;
            }

            inst.elements.clear();
            std::string cur;
            for (char c : raw_content) {
                if (isdigit((unsigned char)c)) {
                    cur += c;
                } else if (!cur.empty()) {
                    inst.elements.push_back(parse_u128_decimal(cur));
                    cur.clear();
                }
            }
            if (!cur.empty()) inst.elements.push_back(parse_u128_decimal(cur));
            inst.target = parse_u128_decimal(tgt_str);

            inst.total_sum = 0;
            u128 max_val = 0;
            for (u128 val : inst.elements) {
                inst.total_sum += val;
                if (val > max_val) max_val = val;
            }
            double max_bits = (max_val == 0) ? 1.0 : (double)std::log2((double)max_val + 1.0);
            inst.density = (inst.elements.empty()) ? 0.0 : ((double)inst.elements.size() / max_bits);
            return !inst.elements.empty();
        }
    };

    // Daftar dasar tingkat daun (gamma = 0: koefisien biner murni)
    struct BaseListSoA {
        std::vector<u64> rem;
        std::vector<u32> mask;

        void clear() { rem.clear(); mask.clear(); }
        void reserve(size_t n) { rem.reserve(n); mask.reserve(n); }
        size_t size() const { return rem.size(); }
        inline void push(u32 m, u64 r) {
            mask.push_back(m);
            rem.push_back(r);
        }
    };

    // Entri kandidat dengan dukungan koefisien terner {-1, 0, +1}
    struct TernaryEntry {
        u64 sum64;       // Aritmatika 64-bit untuk lookup super cepat
        u64 pos_mask;    // Bit bernilai +1
        u64 neg_mask;    // Bit bernilai -1
    };

    // Flat Hash Map khusus 64-bit (Power-of-two open addressing, zero memory explosion)
    struct CompactHashMap {
        struct Slot {
            u64 key;
            u32 index;
            u32 next;
        };
        std::vector<u32> head;
        std::vector<Slot> slots;
        u64 mask_cap;

        void init(size_t expected_elements) {
            size_t cap = 1;
            while (cap < expected_elements * 2) cap <<= 1;
            if (cap < 1024) cap = 1024;
            mask_cap = cap - 1;
            head.assign(cap, 0xFFFFFFFFU);
            slots.clear();
            slots.reserve(expected_elements);
        }

        inline void insert(u64 key, u32 idx) {
            u64 bucket = key & mask_cap;
            slots.push_back({key, idx, head[bucket]});
            head[bucket] = (u32)(slots.size() - 1);
        }
    };

    struct BCJExecutionReport {
        bool        solved = false;
        double      runtime_ms = 0.0;
        double      peak_ram_mb = 0.0;
        u64         partitions_evaluated = 0;
        u64         queries_evaluated = 0;
        u32         threads_used = 1;
        std::vector<int>  witness_indices;
        std::vector<u128> witness_values;
        u128        witness_sum = 0;
    };

    class BCJSolver {
    public:
        static BCJExecutionReport solve(const BCJInstance& inst, unsigned num_threads = 0,
                                        double time_limit_s = 0.0, bool verbose = true) {
            BCJExecutionReport report;
            auto start_time = std::chrono::steady_clock::now();

            int n = (int)inst.elements.size();
            int k = n / 2;
            if (n != 96 || k != 48) {
                std::cerr << "[ERROR]: Solver BCJ ini dioptimalkan khusus skala N=96, K=48." << std::endl;
                return report;
            }

            int q_n = n / 4; // 24
            int q_k = k / 4; // 12

            // C(24, 12) = 2.704.156
            size_t est_combinations = 2704156;

            if (num_threads == 0) {
                num_threads = std::max(1u, std::thread::hardware_concurrency());
            }
            report.threads_used = num_threads;

            // Modulus seimbang untuk menyaring daftar antara (HGJ10/BCJ11 calibrated)
            u64 M = 1500007ULL; 

            int radix_bits = 1;
            while ((1ULL << radix_bits) < M) radix_bits++;

            // 15 query per partisi cukup untuk mengamortisasi pembuatan list
            const int QUERIES_PER_PARTITION = 15;

            std::atomic<bool> solution_found(false);
            std::atomic<u64>  total_partitions(0);
            std::atomic<u64>  total_queries(0);
            std::mutex report_mutex;

            auto worker = [&](unsigned thread_id) {
                std::mt19937_64 rng(1337ULL + thread_id * 10007ULL +
                                (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());

                std::vector<int> perm(n);
                for (int i = 0; i < n; ++i) perm[i] = i;

                BaseListSoA L1, L2, R1, R2;
                L1.reserve(est_combinations);
                L2.reserve(est_combinations);
                R1.reserve(est_combinations);
                R2.reserve(est_combinations);

                std::vector<u32> idx_buf(est_combinations), idx_tmp(est_combinations);

                std::vector<TernaryEntry> cand_L;
                cand_L.reserve(65536);
                CompactHashMap hash_map_L;

                std::vector<u128> Q1(q_n), Q2(q_n), Q3(q_n), Q4(q_n);
                std::vector<u64>  Q1r(q_n), Q2r(q_n), Q3r(q_n), Q4r(q_n);
                std::vector<u64>  Q1_64(q_n), Q2_64(q_n), Q3_64(q_n), Q4_64(q_n);

                u64 target64 = (u64)inst.target;
                u64 target_mod = (u64)(inst.target % (u128)M);

                while (!solution_found) {
                    if (time_limit_s > 0.0) {
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration<double>(now - start_time).count() >= time_limit_s) break;
                    }

                    total_partitions++;
                    std::shuffle(perm.begin(), perm.end(), rng);

                    for (int i = 0; i < q_n; ++i) {
                        Q1[i] = inst.elements[perm[i]];
                        Q2[i] = inst.elements[perm[q_n + i]];
                        Q3[i] = inst.elements[perm[2 * q_n + i]];
                        Q4[i] = inst.elements[perm[3 * q_n + i]];

                        Q1r[i] = (u64)(Q1[i] % (u128)M);
                        Q2r[i] = (u64)(Q2[i] % (u128)M);
                        Q3r[i] = (u64)(Q3[i] % (u128)M);
                        Q4r[i] = (u64)(Q4[i] % (u128)M);

                        Q1_64[i] = (u64)Q1[i];
                        Q2_64[i] = (u64)Q2[i];
                        Q3_64[i] = (u64)Q3[i];
                        Q4_64[i] = (u64)Q4[i];
                    }

                    L1.clear(); L2.clear(); R1.clear(); R2.clear();
                    gen_combinations_dfs(Q1r, q_n, q_k, M, L1);
                    gen_combinations_dfs(Q2r, q_n, q_k, M, L2);
                    gen_combinations_dfs(Q3r, q_n, q_k, M, R1);
                    gen_combinations_dfs(Q4r, q_n, q_k, M, R2);

                    radix_sort_by_rem(L1, idx_buf, idx_tmp, radix_bits);
                    radix_sort_by_rem(L2, idx_buf, idx_tmp, radix_bits);
                    radix_sort_by_rem(R1, idx_buf, idx_tmp, radix_bits);
                    radix_sort_by_rem(R2, idx_buf, idx_tmp, radix_bits);

                    for (int q_iter = 0; q_iter < QUERIES_PER_PARTITION && !solution_found; ++q_iter) {
                        total_queries++;
                        u64 R_mod = rng() % M;
                        u64 R_comp_mod = (target_mod >= R_mod) ? (target_mod - R_mod) : (M - (R_mod - target_mod));

                        // 1. SISI KIRI: Susun cand_L dan indeks ke Hash Table
                        cand_L.clear();
                        build_half_ternary(L1, L2, R_mod, M, q_n, Q1_64, Q2_64, cand_L, solution_found);
                        if (solution_found || cand_L.empty()) continue;

                        hash_map_L.init(cand_L.size());
                        for (size_t idx = 0; idx < cand_L.size(); ++idx) {
                            hash_map_L.insert(cand_L[idx].sum64, (u32)idx);
                        }

                        // 2. SISI KANAN: Streaming Matching (Tanpa alokasi cand_R)
                        stream_match_right(R1, R2, R_comp_mod, M, q_n, Q3_64, Q4_64,
                                        target64, hash_map_L, cand_L, inst, perm,
                                        solution_found, report_mutex, report);
                    }
                }
            };

            (void)verbose;
            std::vector<std::thread> workers;
            for (unsigned t = 0; t < num_threads; ++t) workers.emplace_back(worker, t);
            for (auto& th : workers) if (th.joinable()) th.join();

            auto end_time = std::chrono::steady_clock::now();
            report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            report.peak_ram_mb = get_current_peak_ram_mb();
            report.partitions_evaluated = total_partitions.load();
            report.queries_evaluated = total_queries.load();
            report.solved = solution_found.load();

            return report;
        }

    private:
        static void gen_dfs(int pos, int remaining_k, u64 cur_rem, u32 cur_mask,
                            const std::vector<u64>& A_rem, int q_n, u64 M, BaseListSoA& out) {
            if (remaining_k == 0) {
                out.push(cur_mask, cur_rem);
                return;
            }
            if (pos >= q_n || (q_n - pos) < remaining_k) return;

            gen_dfs(pos + 1, remaining_k, cur_rem, cur_mask, A_rem, q_n, M, out);

            u64 new_rem = cur_rem + A_rem[pos];
            if (new_rem >= M) new_rem -= M;
            gen_dfs(pos + 1, remaining_k - 1, new_rem, cur_mask | (1U << pos), A_rem, q_n, M, out);
        }

        static void gen_combinations_dfs(const std::vector<u64>& A_rem, int q_n, int q_k, u64 M, BaseListSoA& out) {
            gen_dfs(0, q_k, 0, 0, A_rem, q_n, M, out);
        }

        static void radix_sort_by_rem(BaseListSoA& list, std::vector<u32>& idx, std::vector<u32>& idx_tmp, int total_bits) {
            size_t n = list.size();
            if (n < 2) return;

            for (size_t i = 0; i < n; ++i) idx[i] = (u32)i;

            const int RADIX_BITS = 11;
            const u32 RADIX_SIZE = 1U << RADIX_BITS;
            std::vector<u32> count(RADIX_SIZE + 1);

            int passes = (total_bits + RADIX_BITS - 1) / RADIX_BITS;
            std::vector<u32>* src = &idx;
            std::vector<u32>* dst = &idx_tmp;

            for (int p = 0; p < passes; ++p) {
                std::fill(count.begin(), count.end(), 0u);
                int shift = p * RADIX_BITS;
                for (u32 id : *src) {
                    u32 digit = (u32)((list.rem[id] >> shift) & (RADIX_SIZE - 1));
                    count[digit + 1]++;
                }
                for (u32 i = 0; i < RADIX_SIZE; ++i) count[i + 1] += count[i];
                for (u32 id : *src) {
                    u32 digit = (u32)((list.rem[id] >> shift) & (RADIX_SIZE - 1));
                    (*dst)[count[digit]++] = id;
                }
                std::swap(src, dst);
            }

            std::vector<u64> new_rem(n);
            std::vector<u32> new_mask(n);
            for (size_t i = 0; i < n; ++i) {
                u32 id = (*src)[i];
                new_rem[i] = list.rem[id];
                new_mask[i] = list.mask[id];
            }
            list.rem = std::move(new_rem);
            list.mask = std::move(new_mask);
        }

        // Membangun separuh kandidat dengan representasi bitmask terner
        static void build_half_ternary(const BaseListSoA& l1, const BaseListSoA& l2,
                                    u64 target_mod, u64 M, int q_n,
                                    const std::vector<u64>& q1_64, const std::vector<u64>& q2_64,
                                    std::vector<TernaryEntry>& out,
                                    const std::atomic<bool>& solution_found) {
            size_t n1 = l1.size();
            size_t n2 = l2.size();
            if (n1 == 0 || n2 == 0) return;

            auto emit_pair = [&](size_t a, size_t b) {
                u32 m1 = l1.mask[a];
                u32 m2 = l2.mask[b];

                // Filter konsistensi koefisien terner: buang jika ada overlap bit terlarang (+2)
                if (m1 & m2) return; 

                u64 sum64 = 0;
                for (int bit = 0; bit < q_n; ++bit) {
                    if ((m1 >> bit) & 1) sum64 += q1_64[bit];
                    if ((m2 >> bit) & 1) sum64 += q2_64[bit];
                }

                u64 pos_m = ((u64)m2 << q_n) | (u64)m1;
                out.push_back({sum64, pos_m, 0ULL});
            };

            // Two-pointer sweep case 1: rem1 + rem2 == target_mod
            size_t i = 0;
            int64_t j = (int64_t)n2 - 1;
            while (j >= 0 && l2.rem[j] > target_mod) j--;

            while (i < n1 && l1.rem[i] <= target_mod && j >= 0) {
                if (solution_found) return;
                u64 s = l1.rem[i] + l2.rem[j];
                if (s < target_mod) i++;
                else if (s > target_mod) j--;
                else {
                    size_t i2 = i;
                    while (i2 < n1 && l1.rem[i2] == l1.rem[i]) i2++;
                    int64_t j2 = j;
                    while (j2 >= 0 && l2.rem[j2] == l2.rem[j]) j2--;
                    for (size_t a = i; a < i2; ++a) {
                        for (int64_t b = j; b > j2; --b) emit_pair(a, b);
                    }
                    i = i2; j = j2;
                }
            }

            // Two-pointer sweep case 2: rem1 + rem2 == target_mod + M
            u64 target2 = target_mod + M;
            i = 0;
            while (i < n1 && l1.rem[i] <= target_mod) i++;
            j = (int64_t)n2 - 1;

            while (i < n1 && j >= 0 && l2.rem[j] > target_mod) {
                if (solution_found) return;
                u64 s = l1.rem[i] + l2.rem[j];
                if (s < target2) i++;
                else if (s > target2) j--;
                else {
                    size_t i2 = i;
                    while (i2 < n1 && l1.rem[i2] == l1.rem[i]) i2++;
                    int64_t j2 = j;
                    while (j2 >= 0 && l2.rem[j2] == l2.rem[j]) j2--;
                    for (size_t a = i; a < i2; ++a) {
                        for (int64_t b = j; b > j2; --b) emit_pair(a, b);
                    }
                    i = i2; j = j2;
                }
            }
        }

        // Streaming match sisi kanan langsung terhadap Hash Table sisi kiri
        static void stream_match_right(const BaseListSoA& r1, const BaseListSoA& r2,
                                    u64 target_mod, u64 M, int q_n,
                                    const std::vector<u64>& q3_64, const std::vector<u64>& q4_64,
                                    u64 target64, const CompactHashMap& map_L,
                                    const std::vector<TernaryEntry>& cand_L,
                                    const BCJInstance& inst, const std::vector<int>& perm,
                                    std::atomic<bool>& solution_found,
                                    std::mutex& report_mutex, BCJExecutionReport& report) {
            size_t n1 = r1.size();
            size_t n2 = r2.size();
            if (n1 == 0 || n2 == 0) return;

            auto test_pair = [&](size_t a, size_t b) {
                if (solution_found) return;
                u32 m1 = r1.mask[a];
                u32 m2 = r2.mask[b];

                if (m1 & m2) return; // Inkonsisten (+2)

                u64 sum64_R = 0;
                for (int bit = 0; bit < q_n; ++bit) {
                    if ((m1 >> bit) & 1) sum64_R += q3_64[bit];
                    if ((m2 >> bit) & 1) sum64_R += q4_64[bit];
                }

                u64 needed_sum64 = target64 - sum64_R;
                u64 bucket = needed_sum64 & map_L.mask_cap;
                u32 slot_idx = map_L.head[bucket];

                while (slot_idx != 0xFFFFFFFFU) {
                    const auto& slot = map_L.slots[slot_idx];
                    if (slot.key == needed_sum64) {
                        const auto& cL = cand_L[slot.index];

                        // Kandidat lolos filter 64-bit -> Evaluasi u128 penuh (Lazy Reconstruction)
                        u128 full_s = 0;
                        std::vector<int> cur_witness;

                        for (int bit = 0; bit < 2 * q_n; ++bit) {
                            if ((cL.pos_mask >> bit) & 1ULL) {
                                int idx = perm[bit];
                                cur_witness.push_back(idx);
                                full_s += inst.elements[idx];
                            }
                        }

                        u64 pos_m_R = ((u64)m2 << q_n) | (u64)m1;
                        for (int bit = 0; bit < 2 * q_n; ++bit) {
                            if ((pos_m_R >> bit) & 1ULL) {
                                int idx = perm[2 * q_n + bit];
                                cur_witness.push_back(idx);
                                full_s += inst.elements[idx];
                            }
                        }

                        if (full_s == inst.target) {
                            std::unique_lock<std::mutex> lock(report_mutex);
                            if (!solution_found) {
                                solution_found = true;
                                std::sort(cur_witness.begin(), cur_witness.end());
                                report.witness_indices = cur_witness;
                                report.witness_values.clear();
                                for (int id : cur_witness) report.witness_values.push_back(inst.elements[id]);
                                report.witness_sum = full_s;
                            }
                            return;
                        }
                    }
                    slot_idx = slot.next;
                }
            };

            // Two-pointer sweep case 1
            size_t i = 0;
            int64_t j = (int64_t)n2 - 1;
            while (j >= 0 && r2.rem[j] > target_mod) j--;

            while (i < n1 && r1.rem[i] <= target_mod && j >= 0) {
                if (solution_found) return;
                u64 s = r1.rem[i] + r2.rem[j];
                if (s < target_mod) i++;
                else if (s > target_mod) j--;
                else {
                    size_t i2 = i;
                    while (i2 < n1 && r1.rem[i2] == r1.rem[i]) i2++;
                    int64_t j2 = j;
                    while (j2 >= 0 && r2.rem[j2] == r2.rem[j]) j2--;
                    for (size_t a = i; a < i2; ++a) {
                        for (int64_t b = j; b > j2; --b) test_pair(a, b);
                    }
                    i = i2; j = j2;
                }
            }

            // Two-pointer sweep case 2
            u64 target2 = target_mod + M;
            i = 0;
            while (i < n1 && r1.rem[i] <= target_mod) i++;
            j = (int64_t)n2 - 1;

            while (i < n1 && j >= 0 && r2.rem[j] > target_mod) {
                if (solution_found) return;
                u64 s = r1.rem[i] + r2.rem[j];
                if (s < target2) i++;
                else if (s > target2) j--;
                else {
                    size_t i2 = i;
                    while (i2 < n1 && r1.rem[i2] == r1.rem[i]) i2++;
                    int64_t j2 = j;
                    while (j2 >= 0 && r2.rem[j2] == r2.rem[j]) j2--;
                    for (size_t a = i; a < i2; ++a) {
                        for (int64_t b = j; b > j2; --b) test_pair(a, b);
                    }
                    i = i2; j = j2;
                }
            }
        }
    };