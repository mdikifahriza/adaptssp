    #include <iostream>
    #include <vector>
    #include <string>
    #include <sstream>
    #include <fstream>
    #include <algorithm>
    #include <chrono>
    #include <thread>
    #include <atomic>
    #include <mutex>
    #include <condition_variable>
    #include <random>
    #include <cstdint>
    #include <cmath>
    #include <iomanip>
    #include <numeric>
    #include <cstring>
    #include <limits>

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

    // ============================================================================
    // Util u128
    // ============================================================================
    inline std::string u128_to_string(u128 v) {
        if (v == 0) return "0";
        std::string s;
        while (v > 0) { s.push_back((char)('0' + (int)(v % 10))); v /= 10; }
        std::reverse(s.begin(), s.end());
        return s;
    }
    inline std::ostream& operator<<(std::ostream& os, u128 v) { return os << u128_to_string(v); }
    inline u128 parse_u128_decimal(const std::string& str) {
        u128 val = 0;
        for (char c : str) if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
        return val;
    }
    inline double get_current_peak_ram_mb() {
    #if defined(_WIN32) || defined(_WIN64)
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            return pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
        return 0.0;
    #else
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
        return 0.0;
    #endif
    }

    // ============================================================================
    // Instance + Preprocessing
    // ============================================================================
    struct Instance {
        std::vector<u128> elements;
        u128 target = 0;
        u128 total_sum = 0;
        double density = 0.0;
        int k_min = -1, k_max = -1;

        static bool load(const std::string& file_or_str, const std::string& tgt_str, Instance& inst) {
            std::string raw;
            std::ifstream infile(file_or_str);
            if (infile.is_open()) {
                std::string line;
                while (std::getline(infile, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    size_t p = std::min({line.find('#'), line.find("//"), line.find(';')});
                    if (p != std::string::npos) line = line.substr(0, p);
                    raw += line + " ";
                }
            } else {
                raw = file_or_str;
            }
            inst.elements.clear();
            std::string cur;
            for (char c : raw) {
                if (isdigit((unsigned char)c)) cur += c;
                else if (!cur.empty()) { inst.elements.push_back(parse_u128_decimal(cur)); cur.clear(); }
            }
            if (!cur.empty()) inst.elements.push_back(parse_u128_decimal(cur));
            if (!tgt_str.empty()) inst.target = parse_u128_decimal(tgt_str);

            inst.total_sum = 0;
            u128 max_val = 0;
            for (u128 v : inst.elements) { inst.total_sum += v; if (v > max_val) max_val = v; }
            double max_bits = (max_val == 0) ? 1.0 : std::log2((double)max_val + 1.0);
            inst.density = inst.elements.empty() ? 0.0 : ((double)inst.elements.size() / max_bits);

            inst.compute_k_window();
            return !inst.elements.empty();
        }

        void compute_k_window() {
            int n = (int)elements.size();
            std::vector<u128> s = elements;
            std::sort(s.begin(), s.end());

            std::vector<u128> pref(n + 1, 0), suff(n + 1, 0);
            for (int i = 0; i < n; ++i) pref[i + 1] = pref[i] + s[i];
            for (int i = 0; i < n; ++i) suff[i + 1] = suff[i] + s[n - 1 - i];

            k_min = -1; k_max = -1;
            for (int k = 1; k <= n; ++k) {
                u128 min_sum = pref[k];
                u128 max_sum = suff[k];
                if (target >= min_sum && target <= max_sum) {
                    if (k_min == -1) k_min = k;
                    k_max = k;
                }
            }
        }

        double k_window_ratio() const {
            if (elements.empty() || k_min == -1) return 0.0;
            return (double)(k_max - k_min + 1) / (double)elements.size();
        }
    };

    // ============================================================================
    // Witness + Report
    // ============================================================================
    struct Witness {
        std::vector<int> indices;
        std::vector<u128> values;
        u128 sum = 0;
    };

    struct Report {
        bool solved = false;
        double runtime_ms = 0.0;
        u64 partitions_evaluated = 0;
        u64 modular_queries_evaluated = 0;
        u64 base_combinations_generated = 0;
        u64 half_candidates_generated = 0;
        u64 false_positives_rejected = 0;
        double peak_ram_mb = 0.0;
        unsigned threads_used = 0;
        Witness witness;
        std::string verification_msg;
    };

    inline bool verify_witness(const std::vector<u128>& elements, u128 target, int expected_k,
                            Witness& w, std::string& err_msg) {
        if (w.indices.empty()) { err_msg = "Saksi kosong"; return false; }
        std::vector<int> sorted_idx = w.indices;
        std::sort(sorted_idx.begin(), sorted_idx.end());
        for (size_t i = 1; i < sorted_idx.size(); ++i) {
            if (sorted_idx[i] == sorted_idx[i - 1]) {
                err_msg = "Duplikasi indeks terdeteksi: " + std::to_string(sorted_idx[i]);
                return false;
            }
        }
        for (int idx : w.indices) {
            if (idx < 0 || idx >= (int)elements.size()) {
                err_msg = "Indeks di luar rentang: " + std::to_string(idx);
                return false;
            }
        }
        u128 acc = 0;
        w.values.clear();
        for (int idx : w.indices) {
            acc += elements[idx];
            w.values.push_back(elements[idx]);
        }
        w.sum = acc;
        if (acc != target) {
            err_msg = "Jumlah saksi (" + u128_to_string(acc) + ") != target (" + u128_to_string(target) + ")";
            return false;
        }
        if (expected_k > 0 && (int)w.indices.size() != expected_k) {
            err_msg = "Ukuran k (" + std::to_string(w.indices.size()) + ") != expected (" + std::to_string(expected_k) + ")";
            return false;
        }
        err_msg = "OK: verified independently, sum == target, indices unique.";
        return true;
    }

    // ============================================================================
    // ENGINE: HGJ With Full Recommendations 1 to 4
    // Rec 1: Bounded R_mod trial budget per partition
    // Rec 2: Direct Modulo-M1 bucketing / Inverted Index
    // Rec 3: Soft-window relaxation (q_k +/- 1 per quarter)
    // Rec 4: Dual-modulus filtering (M1 + M2=65521) with inverted candidate index
    // ============================================================================
    namespace EngineHGJ {

        struct FastPrng {
            u64 state;
            FastPrng(u64 seed = 13371337ULL) : state(seed ? seed : 0x853c49e6748fea9bULL) {}
            inline u64 next_u64() {
                u64 z = (state += 0x9e3779b97f4a7c15ULL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                return z ^ (z >> 31);
            }
            inline u32 next_range(u32 limit) {
                if (limit <= 1) return 0;
                return (u32)(((u64)(u32)next_u64() * (u64)limit) >> 32);
            }
        };

        struct BaseListSoA2 {
            std::vector<u32> rem1;
            std::vector<u32> rem2;
            std::vector<u32> mask;
            std::vector<u128> sum;

            inline void clear() {
                rem1.clear(); rem2.clear(); mask.clear(); sum.clear();
            }
            inline size_t size() const { return rem1.size(); }
            inline void reserve(size_t cap) {
                rem1.reserve(cap); rem2.reserve(cap); mask.reserve(cap); sum.reserve(cap);
            }
            inline void push_back(u32 r1, u32 r2, u32 m, u128 s) {
                rem1.push_back(r1); rem2.push_back(r2); mask.push_back(m); sum.push_back(s);
            }
        };

        struct HalfEntry2 {
            u128 sum;
            u64 mask_half;
            u32 rem2;
        };

        inline u128 sum_smallest_k(const std::vector<u128>& arr, int k) {
            if (k <= 0) return 0;
            if (k >= (int)arr.size()) {
                u128 s = 0; for (u128 x : arr) s += x; return s;
            }
            std::vector<u128> tmp = arr;
            std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
            u128 s = 0; for (int i = 0; i < k; ++i) s += tmp[i];
            return s;
        }

        inline u128 sum_largest_k(const std::vector<u128>& arr, int k) {
            if (k <= 0) return 0;
            if (k >= (int)arr.size()) {
                u128 s = 0; for (u128 x : arr) s += x; return s;
            }
            std::vector<u128> tmp = arr;
            std::nth_element(tmp.begin(), tmp.begin() + (tmp.size() - k), tmp.end());
            u128 s = 0; for (size_t i = tmp.size() - k; i < tmp.size(); ++i) s += tmp[i];
            return s;
        }

        static void gen_combinations_soft(const std::vector<u128>& Q,
                                        const std::vector<u32>& Qr1,
                                        const std::vector<u32>& Qr2,
                                        int q_n, int W, u32 M1, u32 M2,
                                        BaseListSoA2& list) {
            if (W < 0 || W > q_n) return;
            if (W == 0) {
                list.push_back(0, 0, 0, 0);
                return;
            }
            u32 mask = (1U << W) - 1U;
            u32 limit = 1U << q_n;
            while (mask < limit) {
                u128 sum = 0;
                u32 r1 = 0, r2 = 0;
                u32 temp = mask;
                while (temp) {
                    int b = __builtin_ctz(temp);
                    sum += Q[b];
                    r1 += Qr1[b];
                    r2 += Qr2[b];
                    temp &= temp - 1;
                }
                r1 %= M1;
                r2 %= M2;
                list.push_back(r1, r2, mask, sum);

                u32 c = mask & -mask;
                u32 r = mask + c;
                if (r == 0) break;
                mask = (((r ^ mask) >> 2) / c) | r;
            }
        }

        static void radix_sort_by_rem1(BaseListSoA2& list,
                                    std::vector<u32>& idx,
                                    std::vector<u32>& idx_tmp,
                                    std::vector<u32>& buf_r1,
                                    std::vector<u32>& buf_r2,
                                    std::vector<u32>& buf_mask,
                                    std::vector<u128>& buf_sum,
                                    int total_bits) {
            size_t n = list.size();
            if (n < 2) return;
            idx.resize(n);
            for (size_t i = 0; i < n; ++i) idx[i] = (u32)i;
            idx_tmp.resize(n);

            const int RADIX_BITS = 13;
            const u32 RADIX_SIZE = 1U << RADIX_BITS;
            std::vector<u32> count(RADIX_SIZE + 1);
            int passes = (total_bits + RADIX_BITS - 1) / RADIX_BITS;
            if (passes < 1) passes = 1;

            std::vector<u32>* src = &idx;
            std::vector<u32>* dst = &idx_tmp;
            for (int p = 0; p < passes; ++p) {
                std::fill(count.begin(), count.end(), 0u);
                int shift = p * RADIX_BITS;
                for (u32 id : *src) count[((list.rem1[id] >> shift) & (RADIX_SIZE - 1)) + 1]++;
                for (u32 i = 0; i < RADIX_SIZE; ++i) count[i + 1] += count[i];
                for (u32 id : *src) {
                    u32 digit = (u32)((list.rem1[id] >> shift) & (RADIX_SIZE - 1));
                    (*dst)[count[digit]++] = id;
                }
                std::swap(src, dst);
            }

            buf_r1.resize(n); buf_r2.resize(n); buf_mask.resize(n); buf_sum.resize(n);
            for (size_t i = 0; i < n; ++i) {
                u32 id = (*src)[i];
                buf_r1[i] = list.rem1[id];
                buf_r2[i] = list.rem2[id];
                buf_mask[i] = list.mask[id];
                buf_sum[i] = list.sum[id];
            }
            list.rem1 = buf_r1;
            list.rem2 = buf_r2;
            list.mask = buf_mask;
            list.sum = buf_sum;
        }

        static void merge_quarter_pair(const BaseListSoA2& list1, const BaseListSoA2& list2,
                                    const std::vector<u32>& active1,
                                    const std::vector<u32>& head1, const std::vector<u32>& count1,
                                    const std::vector<u32>& head2, const std::vector<u32>& count2,
                                    u32 target_mod1, u32 M1, u32 M2,
                                    u128 lower_bound, u128 upper_bound,
                                    std::vector<HalfEntry2>& out_cand, int q_n) {
            const u128* const l1_sum = list1.sum.data();
            const u32*  const l1_mask = list1.mask.data();
            const u32*  const l1_rem2 = list1.rem2.data();
            const u128* const l2_sum = list2.sum.data();
            const u32*  const l2_mask = list2.mask.data();
            const u32*  const l2_rem2 = list2.rem2.data();

            for (u32 r1 : active1) {
                u32 r2 = (target_mod1 >= r1) ? (target_mod1 - r1) : (target_mod1 + M1 - r1);
                u32 c2 = count2[r2];
                if (c2 == 0) continue;
                u32 s1 = head1[r1], c1 = count1[r1];
                u32 s2 = head2[r2];
                for (u32 a = 0; a < c1; ++a) {
                    u128 sum1 = l1_sum[s1 + a];
                    if (sum1 > upper_bound) continue;
                    u32 m1 = l1_mask[s1 + a];
                    u32 rem2_1 = l1_rem2[s1 + a];
                    for (u32 b = 0; b < c2; ++b) {
                        u128 half_sum = sum1 + l2_sum[s2 + b];
                        if (half_sum < lower_bound || half_sum > upper_bound) continue;
                        u32 rem2 = rem2_1 + l2_rem2[s2 + b];
                        if (rem2 >= M2) rem2 -= M2;
                        out_cand.push_back({half_sum, ((u64)l2_mask[s2 + b] << q_n) | (u64)m1, rem2});
                    }
                }
            }
        }

        // Pilih q_n terbesar (<= n/4) yang bikin total memori base-list
        // (3 window x 4 list x num_threads x C(q_n,q_k) x bytes_per_entry) tetap
        // di bawah mem_budget_mb. Turun bertahap dari n/4 sampai muat.
        static int choose_q_n(int n_full, int k_half, unsigned num_threads,
                            double mem_budget_mb, int bytes_per_entry = 32) {
            double budget_bytes = mem_budget_mb * 1024.0 * 1024.0;
            int n_head_quarter_max = n_full / 4;
            for (int q_n = n_head_quarter_max; q_n >= 2; --q_n) {
                int q_k = std::max(1, (int)std::llround((double)q_n * k_half / (double)n_full));
                q_k = std::min(q_k, q_n);
                double est = 1.0;
                for (int i = 0; i < q_k; ++i) est = est * (double)(q_n - i) / (double)(i + 1);
                double total_mem = 3.0 * 4.0 * (double)num_threads * est * (double)bytes_per_entry;
                if (total_mem <= budget_bytes) return q_n;
            }
            return 2;
        }

        static Report solve(const Instance& inst_in, unsigned num_threads, double time_limit_s,
                            bool proof_mode, bool verbose, double mem_budget_mb = 1500.0) {
            Report report;
            auto start_time = std::chrono::steady_clock::now();
            Instance inst = inst_in;
            int n = (int)inst.elements.size();
            int k = n / 2;
            if (num_threads == 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
            int q_n = choose_q_n(n, k, num_threads, mem_budget_mb);
            int q_k = std::max(1, (int)std::llround((double)q_n * k / (double)n));
            q_k = std::min(q_k, q_n);

            size_t est_combinations = 1;
            for (int i = 0; i < q_k; ++i) est_combinations = est_combinations * (q_n - i) / (i + 1);

            report.threads_used = num_threads;

            u32 M1;
            if (est_combinations <= 50) M1 = 127;
            else if (est_combinations <= 500) M1 = 1009;
            else if (est_combinations <= 2000) M1 = 4093;
            else if (est_combinations <= 6000) M1 = 8191;
            else if (est_combinations <= 20000) M1 = 16381;
            else M1 = 32749;

            const u32 M2 = 65521;

            int radix_bits_total = 1;
            while ((1ULL << radix_bits_total) < M1) radix_bits_total++;

            if (verbose) {
                std::cout << "[ENGINE] Mode: HGJ FULL (Rec 1: Bounded Trials, Rec 2: Direct Bucketing,\n"
                        << "                         Rec 3: Soft-Window q_k+/-1, Rec 4: Dual-Modulus M1/M2)\n";
                std::cout << "[INFO] n=" << n << " k=" << k << " q_n=" << q_n << " q_k=" << q_k << "\n";
                std::cout << "[INFO] C(q_n,q_k)=" << est_combinations << "  M1=" << M1 << "  M2=" << M2
                        << "  threads=" << num_threads << "\n";
            }

            std::atomic<bool> solution_found(false);
            std::atomic<u64>  total_partitions(0);
            std::atomic<u64>  total_queries(0);
            std::atomic<u64>  total_base_entries(0);
            std::atomic<u64>  total_half_candidates(0);
            std::atomic<u64>  total_false_positives(0);
            std::mutex report_mutex;
            std::condition_variable cv_done;
            Witness best_witness;

            auto worker = [&](unsigned thread_id) {
                FastPrng rng(1337ULL + thread_id * 10007ULL +
                    (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
                std::vector<int> perm(n);
                for (int i = 0; i < n; ++i) perm[i] = i;

                BaseListSoA2 L1[3], L2[3], R1[3], R2[3];
                for (int w = 0; w < 3; ++w) {
                    L1[w].reserve(est_combinations); L2[w].reserve(est_combinations);
                    R1[w].reserve(est_combinations); R2[w].reserve(est_combinations);
                }

                std::vector<u32> idx_buf, idx_tmp;
                std::vector<u32> buf_r1, buf_r2, buf_mask;
                std::vector<u128> buf_sum;

                std::vector<u32> head_L1[3], count_L1[3], active_L1[3];
                std::vector<u32> head_L2[3], count_L2[3];
                std::vector<u32> head_R1[3], count_R1[3], active_R1[3];
                std::vector<u32> head_R2[3], count_R2[3];
                for (int w = 0; w < 3; ++w) {
                    head_L1[w].assign(M1, 0); count_L1[w].assign(M1, 0); active_L1[w].reserve(est_combinations);
                    head_L2[w].assign(M1, 0); count_L2[w].assign(M1, 0);
                    head_R1[w].assign(M1, 0); count_R1[w].assign(M1, 0); active_R1[w].reserve(est_combinations);
                    head_R2[w].assign(M1, 0); count_R2[w].assign(M1, 0);
                }

                std::vector<HalfEntry2> cand_L, cand_R;
                size_t est_cand = std::min((size_t)262144, std::max((size_t)4096, (est_combinations * est_combinations * 3) / (size_t)M1 * 2));
                cand_L.reserve(est_cand); cand_R.reserve(est_cand);

                std::vector<int> head_cand_R(M2, -1);
                std::vector<int> next_cand_R;
                next_cand_R.reserve(est_cand);
                std::vector<u32> active_cand_R;
                active_cand_R.reserve(est_cand);

                std::vector<u128> Q1(q_n), Q2(q_n), Q3(q_n), Q4(q_n);
                std::vector<u32>  Q1r1(q_n), Q2r1(q_n), Q3r1(q_n), Q4r1(q_n);
                std::vector<u32>  Q1r2(q_n), Q2r2(q_n), Q3r2(q_n), Q4r2(q_n);
                std::vector<u128> Left_elem(2 * q_n), Right_elem(2 * q_n);

                u64 local_partitions = 0, local_queries = 0, local_base_entries = 0;
                auto flush_counters = [&]() {
                    if (local_partitions) total_partitions += local_partitions;
                    if (local_queries) total_queries += local_queries;
                    if (local_base_entries) total_base_entries += local_base_entries;
                    local_partitions = local_queries = local_base_entries = 0;
                };

                u128 min_L = 0, max_L = 0;
                int reuse_counter = 0;
                const int MAX_REUSE = 4;

                while (!solution_found.load(std::memory_order_relaxed)) {
                    if (time_limit_s > 0.0) {
                        double el = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start_time).count();
                        if (el >= time_limit_s) break;
                    }

                    local_partitions++;
                    bool rebuild_left = (reuse_counter == 0);

                    if (rebuild_left) {
                        for (int i = n - 1; i > 0; --i) {
                            int j = (int)rng.next_range(i + 1);
                            std::swap(perm[i], perm[j]);
                        }
                        for (int i = 0; i < q_n; ++i) {
                            Q1[i] = inst.elements[perm[i]];
                            Q2[i] = inst.elements[perm[q_n + i]];
                            Left_elem[i] = Q1[i]; Left_elem[q_n + i] = Q2[i];
                        }
                    } else {
                        for (int i = n - 1; i > 2 * q_n; --i) {
                            int j = 2 * q_n + (int)rng.next_range(i - 2 * q_n + 1);
                            std::swap(perm[i], perm[j]);
                        }
                    }

                    for (int i = 0; i < q_n; ++i) {
                        Q3[i] = inst.elements[perm[2 * q_n + i]];
                        Q4[i] = inst.elements[perm[3 * q_n + i]];
                        Right_elem[i] = Q3[i]; Right_elem[q_n + i] = Q4[i];
                    }

                    int half_k = 2 * q_k;
                    if (rebuild_left) {
                        min_L = sum_smallest_k(Left_elem, half_k);
                        max_L = sum_largest_k(Left_elem, half_k);
                    }
                    u128 min_R = sum_smallest_k(Right_elem, half_k);
                    u128 max_R = sum_largest_k(Right_elem, half_k);

                    u128 lower_bound_L = (inst.target > max_R) ? (inst.target - max_R) : 0;
                    u128 upper_bound_L = (inst.target >= min_R) ? (inst.target - min_R) : 0;
                    u128 lower_bound_R = (inst.target > max_L) ? (inst.target - max_L) : 0;
                    u128 upper_bound_R = (inst.target >= min_L) ? (inst.target - min_L) : 0;

                    if (rebuild_left) {
                        for (int i = 0; i < q_n; ++i) {
                            Q1r1[i] = (u32)(Q1[i] % (u128)M1); Q1r2[i] = (u32)(Q1[i] % (u128)M2);
                            Q2r1[i] = (u32)(Q2[i] % (u128)M1); Q2r2[i] = (u32)(Q2[i] % (u128)M2);
                        }
                    }
                    for (int i = 0; i < q_n; ++i) {
                        Q3r1[i] = (u32)(Q3[i] % (u128)M1); Q3r2[i] = (u32)(Q3[i] % (u128)M2);
                        Q4r1[i] = (u32)(Q4[i] % (u128)M1); Q4r2[i] = (u32)(Q4[i] % (u128)M2);
                    }

                    int target_weights[3] = {q_k - 1, q_k, q_k + 1};
                    for (int w = 0; w < 3; ++w) {
                        int W = target_weights[w];
                        if (rebuild_left) {
                            L1[w].clear(); L2[w].clear();
                            gen_combinations_soft(Q1, Q1r1, Q1r2, q_n, W, M1, M2, L1[w]);
                            gen_combinations_soft(Q2, Q2r1, Q2r2, q_n, W, M1, M2, L2[w]);
                            local_base_entries += (u64)(L1[w].size() + L2[w].size());

                            radix_sort_by_rem1(L1[w], idx_buf, idx_tmp, buf_r1, buf_r2, buf_mask, buf_sum, radix_bits_total);
                            radix_sort_by_rem1(L2[w], idx_buf, idx_tmp, buf_r1, buf_r2, buf_mask, buf_sum, radix_bits_total);

                            active_L1[w].clear();
                            for (size_t i = 0; i < L1[w].size(); ++i) {
                                u32 r = L1[w].rem1[i];
                                if (count_L1[w][r] == 0) {
                                    head_L1[w][r] = (u32)i;
                                    active_L1[w].push_back(r);
                                }
                                count_L1[w][r]++;
                            }
                            for (size_t i = 0; i < L2[w].size(); ++i) {
                                u32 r = L2[w].rem1[i];
                                if (count_L2[w][r] == 0) head_L2[w][r] = (u32)i;
                                count_L2[w][r]++;
                            }
                        }

                        R1[w].clear(); R2[w].clear();
                        gen_combinations_soft(Q3, Q3r1, Q3r2, q_n, W, M1, M2, R1[w]);
                        gen_combinations_soft(Q4, Q4r1, Q4r2, q_n, W, M1, M2, R2[w]);
                        local_base_entries += (u64)(R1[w].size() + R2[w].size());

                        radix_sort_by_rem1(R1[w], idx_buf, idx_tmp, buf_r1, buf_r2, buf_mask, buf_sum, radix_bits_total);
                        radix_sort_by_rem1(R2[w], idx_buf, idx_tmp, buf_r1, buf_r2, buf_mask, buf_sum, radix_bits_total);

                        active_R1[w].clear();
                        for (size_t i = 0; i < R1[w].size(); ++i) {
                            u32 r = R1[w].rem1[i];
                            if (count_R1[w][r] == 0) {
                                head_R1[w][r] = (u32)i;
                                active_R1[w].push_back(r);
                            }
                            count_R1[w][r]++;
                        }
                        for (size_t i = 0; i < R2[w].size(); ++i) {
                            u32 r = R2[w].rem1[i];
                            if (count_R2[w][r] == 0) head_R2[w][r] = (u32)i;
                            count_R2[w][r]++;
                        }
                    }

                    u32 target_mod1 = (u32)(inst.target % (u128)M1);
                    u32 target_mod2 = (u32)(inst.target % (u128)M2);

                    u32 max_trials = (M1 <= 2048) ? M1 : std::min(M1, (u32)2048);
                    u32 r_start = (u32)rng.next_range(M1);
                    u32 r_step = 997 % M1; if (r_step == 0) r_step = 1;

                    for (u32 trial = 0; trial < max_trials && !solution_found; ++trial) {
                        local_queries++;
                        u32 R_mod = (M1 <= 2048) ? trial : ((r_start + trial * r_step) % M1);
                        u32 R_comp_mod = (target_mod1 >= R_mod)
                            ? (target_mod1 - R_mod)
                            : (M1 - (R_mod - target_mod1));

                        cand_L.clear();
                        merge_quarter_pair(L1[1], L2[1], active_L1[1], head_L1[1], count_L1[1], head_L2[1], count_L2[1],
                                        R_mod, M1, M2, lower_bound_L, upper_bound_L, cand_L, q_n);
                        if (L1[0].size() > 0 && L2[2].size() > 0) {
                            merge_quarter_pair(L1[0], L2[2], active_L1[0], head_L1[0], count_L1[0], head_L2[2], count_L2[2],
                                            R_mod, M1, M2, lower_bound_L, upper_bound_L, cand_L, q_n);
                        }
                        if (L1[2].size() > 0 && L2[0].size() > 0) {
                            merge_quarter_pair(L1[2], L2[0], active_L1[2], head_L1[2], count_L1[2], head_L2[0], count_L2[0],
                                            R_mod, M1, M2, lower_bound_L, upper_bound_L, cand_L, q_n);
                        }
                        total_half_candidates.fetch_add((u64)cand_L.size(), std::memory_order_relaxed);
                        if (solution_found.load(std::memory_order_relaxed) || cand_L.empty()) continue;

                        cand_R.clear();
                        merge_quarter_pair(R1[1], R2[1], active_R1[1], head_R1[1], count_R1[1], head_R2[1], count_R2[1],
                                        R_comp_mod, M1, M2, lower_bound_R, upper_bound_R, cand_R, q_n);
                        if (R1[0].size() > 0 && R2[2].size() > 0) {
                            merge_quarter_pair(R1[0], R2[2], active_R1[0], head_R1[0], count_R1[0], head_R2[2], count_R2[2],
                                            R_comp_mod, M1, M2, lower_bound_R, upper_bound_R, cand_R, q_n);
                        }
                        if (R1[2].size() > 0 && R2[0].size() > 0) {
                            merge_quarter_pair(R1[2], R2[0], active_R1[2], head_R1[2], count_R1[2], head_R2[0], count_R2[0],
                                            R_comp_mod, M1, M2, lower_bound_R, upper_bound_R, cand_R, q_n);
                        }
                        total_half_candidates.fetch_add((u64)cand_R.size(), std::memory_order_relaxed);
                        if (solution_found.load(std::memory_order_relaxed) || cand_R.empty()) continue;

                        if (next_cand_R.size() < cand_R.size()) next_cand_R.resize(cand_R.size() + 4096);
                        active_cand_R.clear();
                        for (size_t i = 0; i < cand_R.size(); ++i) {
                            u32 r2 = cand_R[i].rem2;
                            if (head_cand_R[r2] == -1) {
                                active_cand_R.push_back(r2);
                            }
                            next_cand_R[i] = head_cand_R[r2];
                            head_cand_R[r2] = (int)i;
                        }

                        u64 target_u64 = (u64)inst.target;
                        for (const auto& cL : cand_L) {
                            if (solution_found.load(std::memory_order_relaxed)) break;
                            if (cL.sum > inst.target) continue;
                            u32 req_r2 = (target_mod2 >= cL.rem2) ? (target_mod2 - cL.rem2) : (target_mod2 + M2 - cL.rem2);
                            u64 cL_sum_u64 = (u64)cL.sum;
                            int cL_pop = __builtin_popcountll(cL.mask_half);

                            for (int idx = head_cand_R[req_r2]; idx != -1; idx = next_cand_R[idx]) {
                                if ((cL_sum_u64 + (u64)cand_R[idx].sum) != target_u64) continue;
                                if (cL_pop + __builtin_popcountll(cand_R[idx].mask_half) != k) continue;
                                if (cL.sum + cand_R[idx].sum != inst.target) continue;

                                Witness cand;
                                for (int b = 0; b < q_n; ++b) {
                                    if ((cL.mask_half >> b) & 1ULL) {
                                        int id = perm[b];
                                        cand.indices.push_back(id); cand.values.push_back(inst.elements[id]);
                                    }
                                    if ((cL.mask_half >> (q_n + b)) & 1ULL) {
                                        int id = perm[q_n + b];
                                        cand.indices.push_back(id); cand.values.push_back(inst.elements[id]);
                                    }
                                }
                                for (int b = 0; b < q_n; ++b) {
                                    if ((cand_R[idx].mask_half >> b) & 1ULL) {
                                        int id = perm[2 * q_n + b];
                                        cand.indices.push_back(id); cand.values.push_back(inst.elements[id]);
                                    }
                                    if ((cand_R[idx].mask_half >> (q_n + b)) & 1ULL) {
                                        int id = perm[3 * q_n + b];
                                        cand.indices.push_back(id); cand.values.push_back(inst.elements[id]);
                                    }
                                }

                                std::string msg;
                                bool ok = verify_witness(inst.elements, inst.target, k, cand, msg);
                                if (!ok) {
                                    total_false_positives.fetch_add(1, std::memory_order_relaxed);
                                    continue;
                                }

                                std::unique_lock<std::mutex> lock(report_mutex);
                                if (!solution_found.load(std::memory_order_relaxed)) {
                                    solution_found = true;
                                    best_witness = cand;
                                    report.verification_msg = msg;
                                    cv_done.notify_all();
                                }
                                for (u32 r : active_cand_R) head_cand_R[r] = -1;
                                active_cand_R.clear();
                                flush_counters();
                                return;
                            }
                        }

                        for (u32 r : active_cand_R) head_cand_R[r] = -1;
                        active_cand_R.clear();
                    }

                    for (int w = 0; w < 3; ++w) {
                        for (u32 r : active_R1[w]) count_R1[w][r] = 0;
                        for (size_t i = 0; i < R2[w].size(); ++i) count_R2[w][R2[w].rem1[i]] = 0;
                    }
                    reuse_counter++;
                    if (reuse_counter == MAX_REUSE) {
                        for (int w = 0; w < 3; ++w) {
                            for (u32 r : active_L1[w]) count_L1[w][r] = 0;
                            for (size_t i = 0; i < L2[w].size(); ++i) count_L2[w][L2[w].rem1[i]] = 0;
                        }
                        reuse_counter = 0;
                    }

                    flush_counters();
                }
                flush_counters();
            };

            std::vector<std::thread> workers;
            for (unsigned t = 0; t < num_threads; ++t) {
                workers.emplace_back(worker, t);
            }

            auto monitor_start = std::chrono::steady_clock::now();
            while (!solution_found.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - monitor_start).count();

                if (time_limit_s > 0.0 && elapsed >= time_limit_s) break;

                static double last_print = 0.0;
                if (verbose && (elapsed - last_print >= 10.0)) {
                    last_print = elapsed;
                    u64 p = total_partitions.load(std::memory_order_relaxed);
                    u64 q = total_queries.load(std::memory_order_relaxed);
                    u64 fp = total_false_positives.load(std::memory_order_relaxed);
                    double rate = (elapsed > 0) ? ((double)p / elapsed) : 0.0;
                    double fp_rate = (p > 0) ? ((double)fp / (double)p * 100.0) : 0.0;

                    double p_hit = 1.0 / 10.0;
                    double est_needed = (p_hit > 0.0) ? (std::log(0.05) / std::log(1.0 - p_hit)) : 0.0;
                    double rem_part = std::max(0.0, est_needed - (double)p);
                    double eta_s = (rate > 0.0) ? (rem_part / rate) : 0.0;

                    std::cout << "[PROGRESS] t=" << std::fixed << std::setprecision(1) << elapsed << "s"
                            << "  partitions=" << p
                            << "  queries=" << q
                            << "  rate=" << std::setprecision(2) << rate << " part/s"
                            << "  k_window_ratio=" << std::setprecision(3) << inst.k_window_ratio()
                            << "  false_positives=" << fp << " (" << std::setprecision(3) << fp_rate << "%/partisi)"
                            << "  | min. partisi tambahan (95% CI)~=" << (u64)rem_part
                            << " (~" << std::setprecision(1) << eta_s << "s)\n";
                }
            }

            for (auto& th : workers) if (th.joinable()) th.join();

            auto end_time = std::chrono::steady_clock::now();
            report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            report.solved = solution_found.load();
            report.partitions_evaluated = total_partitions.load();
            report.modular_queries_evaluated = total_queries.load();
            report.base_combinations_generated = total_base_entries.load();
            report.half_candidates_generated = total_half_candidates.load();
            report.false_positives_rejected = total_false_positives.load();
            report.peak_ram_mb = get_current_peak_ram_mb();
            report.witness = best_witness;
            return report;
        }
    }

    // ============================================================================
    // Adaptive Solver Router
    // ============================================================================
    struct Solver {
        static Report solve(const Instance& inst, unsigned num_threads, double time_limit_s,
                            bool proof_mode, bool verbose, double mem_budget_mb = 1500.0) {
            if (verbose) std::cout << "[ENGINE] Mode: HGJ Full (EngineHGJ, all N)\n";
            return EngineHGJ::solve(inst, num_threads, time_limit_s, proof_mode, verbose, mem_budget_mb);
        }
    };

    // ============================================================================
    // CLI Main
    // ============================================================================
    int main(int argc, char* argv[]) {
        if (argc < 2) {
            std::cout << "Usage: adaptssp2.exe <file_or_string> [target] [threads] [timelimit_s] [--calibrate_s=N] [--proof]\n";
            return 1;
        }

        std::string file_path = argv[1];
        std::string tgt_str = "";
        unsigned num_threads = 0;
        double time_limit_s = 0.0;
        double calibrate_s = 0.0;
        bool proof_mode = false;
        double mem_budget_mb = 1500.0;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--calibrate_s=", 0) == 0) {
                calibrate_s = std::atof(arg.substr(14).c_str());
            } else if (arg.rfind("--mem_mb=", 0) == 0) {
                mem_budget_mb = std::atof(arg.substr(9).c_str());
            } else if (arg == "--proof") {
                proof_mode = true;
            } else if (i == 2) tgt_str = arg;
            else if (i == 3) num_threads = std::stoul(arg);
            else if (i == 4) time_limit_s = std::atof(argv[i]);
        }

        Instance inst;
        if (!Instance::load(file_path, tgt_str, inst)) {
            std::cerr << "[ERROR] Gagal membaca instance dari: " << file_path << "\n";
            return 1;
        }

        std::cout << "================================================================================\n";
        std::cout << " HGJ FULL SOLVER (EngineHGJ - ALL N)\n";
        std::cout << "================================================================================\n";
        std::cout << "N elemen        : " << inst.elements.size() << "\n";
        std::cout << "Target          : " << inst.target << "\n";
        std::cout << "Densitas        : " << std::fixed << std::setprecision(4) << inst.density << "\n";
        std::cout << "K window (exact): [" << inst.k_min << " .. " << inst.k_max << "]"
                << "  (ratio=" << std::setprecision(3) << inst.k_window_ratio() << ")\n";

        if (inst.k_max < inst.k_min) {
            std::cout << "[UNSAT TERBUKTI DARI PREPROCESSING]\n";
            return 1;
        }

        if (calibrate_s > 0) {
            std::cout << "--------------------------------------------------------------------------------\n";
            std::cout << "[CALIBRATION] " << std::fixed << std::setprecision(1) << calibrate_s
                    << " detik, memakai pipeline HGJ yang sama.\n";
            Report cal = Solver::solve(inst, num_threads, calibrate_s, false, false, mem_budget_mb);
            if (cal.solved) {
                std::cout << "[CALIBRATION] SOLVED sebelum long-run.\n";
                std::cout << "Runtime : " << std::fixed << std::setprecision(2) << cal.runtime_ms / 1000.0 << " s\n";
                std::cout << "Partisi : " << cal.partitions_evaluated << "\n";
                std::cout << "Peak RAM: " << std::fixed << std::setprecision(2) << cal.peak_ram_mb << " MB\n";
                return 0;
            }
        }

        std::cout << "Mem budget      : " << std::fixed << std::setprecision(0) << mem_budget_mb << " MB\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "[RUNNING SOLVER]\n";

        Report report = Solver::solve(inst, num_threads, time_limit_s, proof_mode, true, mem_budget_mb);

        std::cout << "\n================================================================================\n";
        std::cout << "                              LAPORAN EKSEKUSI\n";
        std::cout << "================================================================================\n";
        std::cout << "Status               : " << (report.solved ? "SOLVED (verified)" : "NOT SOLVED") << "\n";
        std::cout << "Runtime              : " << std::fixed << std::setprecision(2) << report.runtime_ms / 1000.0 << " detik\n";
        std::cout << "Partisi dievaluasi   : " << report.partitions_evaluated << "\n";
        std::cout << "Query modular        : " << report.modular_queries_evaluated << "\n";
        std::cout << "Kombinasi base total : " << report.base_combinations_generated << "\n";
        std::cout << "Half candidates      : " << report.half_candidates_generated << "\n";
        std::cout << "False-positive       : " << report.false_positives_rejected << "\n";
        std::cout << "Peak RAM             : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB\n";
        std::cout << "Threads              : " << report.threads_used << "\n";
        std::cout << "Pesan verifikasi     : " << report.verification_msg << "\n";

        if (report.solved) {
            std::cout << "SAKSI (" << report.witness.indices.size() << " indeks): [";
            for (size_t i = 0; i < report.witness.indices.size(); ++i)
                std::cout << report.witness.indices[i] << (i + 1 < report.witness.indices.size() ? ", " : "");
            std::cout << "]\nJumlah saksi         : " << report.witness.sum << "\n";
            std::cout << "Target               : " << inst.target << "\n";
            std::cout << "Cocok?               : " << (report.witness.sum == inst.target ? "YA, 100% PRESISI" : "TIDAK (bug!)") << "\n";
        }

        std::cout << "================================================================================\n";
        return report.solved ? 0 : 1;
    }