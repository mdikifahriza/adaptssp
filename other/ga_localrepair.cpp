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
#include <functional>

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
    int generations_completed = 0;
    u64 fitness_evaluations = 0;
    u64 crossover_operations = 0;
    u64 mutation_bit_flips = 0;
    u64 mutated_individuals = 0;
    u128 best_fitness = 0;
    double average_fitness = 0.0;
    double diversity_percent = 0.0;
    int best_k = 0;
    // --- new: local repair diagnostics ---
    u64 local_repair_attempts = 0;
    u64 local_repair_1swap_success = 0;
    u64 local_repair_2swap_success = 0;
    int population_used = 0;
    int elitism_used = 0;
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
// ENGINE: Genetic Algorithm (Heuristic Subset Sum) + Local Repair + k-window
// ============================================================================
namespace EngineGA {

    struct Individual {
        std::vector<uint8_t> chromosome;
        u128 sum = 0;
        u128 fitness = 0;
    };

    class GeneticAlgorithm {
    public:
        GeneticAlgorithm(const std::vector<u128>& values, u128 target,
                         int population_size, int max_generations,
                         double crossover_rate, double mutation_rate,
                         int elitism, uint64_t seed,
                         int k_min = -1, int k_max = -1,
                         bool diagnostic = false, int diagnostic_interval = 100)
            : a(values), target(target),
              P(population_size), G(max_generations),
              crossover_rate(crossover_rate),
              mutation_rate(mutation_rate > 0.0 ? mutation_rate : 1.0 / (double)values.size()),
              elitism(elitism), rng(seed),
              k_min(k_min), k_max(k_max),
              diagnostic(diagnostic), diagnostic_interval(std::max(1, diagnostic_interval)) {
            // Guard rail: kalau k_window tidak valid, jangan batasi popcount.
            if (this->k_min <= 0 || this->k_max <= 0 || this->k_min > (int)a.size() || this->k_max > (int)a.size()) {
                this->k_min = 1;
                this->k_max = (int)a.size();
            }
            // Local repair (r dibuang, c ditambah, r boleh != c -- popcount ikut berubah).
            // Susun daftar (r,c) yang dicoba, diurutkan dari yang termurah, dan hanya
            // masukkan pasangan yang estimasi biayanya masih dalam anggaran komputasi.
            double n_ones_est = (double)k_max;
            double n_zeros_est = (double)(a.size() - k_min);
            std::vector<std::pair<int,int>> candidates = {
                {1,0},{0,1},{1,1},{2,0},{0,2},{2,1},{1,2},{2,2},
                {3,1},{1,3},{3,2},{2,3},{3,3},{4,1},{1,4},{4,2},{2,4}
            };
            repair_rc_list.clear();
            for (auto& rc : candidates) {
                double combos = choose(n_ones_est, rc.first) * choose(n_zeros_est, rc.second);
                if (combos <= 2'000'000.0) repair_rc_list.push_back(rc);
            }
            if (repair_rc_list.empty()) repair_rc_list.push_back({1, 0});
        }

        Report solve(double time_limit_s) {
            Report report;
            auto start_time = std::chrono::steady_clock::now();
            fitness_evaluations = 0;
            crossover_operations = 0;
            mutation_bit_flips = 0;
            mutated_individuals = 0;
            local_repair_attempts = 0;
            local_repair_1swap_success = 0;
            local_repair_2swap_success = 0;

            auto elapsed_s = [&]() -> double {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            };

            if (diagnostic) {
                std::cout << "\n[GA DIAGNOSTIC CONFIG]\n";
                std::cout << "N                 : " << a.size() << "\n";
                std::cout << "Population        : " << P << "\n";
                std::cout << "Max generations   : " << G << " (dikontrol time_limit_s, bukan hard cap)\n";
                std::cout << "Crossover rate    : " << crossover_rate << "\n";
                std::cout << "Mutation rate     : " << mutation_rate << "\n";
                std::cout << "Elitism           : " << elitism << "\n";
                std::cout << "k_window          : [" << k_min << " .. " << k_max << "]\n";
                std::cout << "Repair (r,c) pairs: " << repair_rc_list.size() << " kombinasi dicoba per repair\n";
                std::cout << "Diagnostic every  : " << diagnostic_interval << " generations\n";
            }

            std::vector<Individual> population;
            population.reserve(P);

            for (int i = 0; i < P; ++i) {
                Individual ind;
                init_biased(ind);
                evaluate(ind);
                if (try_finish_if_solved(ind, 0, start_time, report)) return report;
                population.push_back(std::move(ind));
            }

            if (diagnostic && !population.empty()) print_diagnostic(population, 0, start_time);

            int generation = 1;
            for (;; ++generation) {
                if (time_limit_s > 0.0 && elapsed_s() >= time_limit_s) break;
                if (G > 0 && generation > G) break;

                std::sort(population.begin(), population.end(),
                          [](const Individual& x, const Individual& y) {
                              return x.fitness < y.fitness;
                          });

                if (population[0].sum == target) {
                    finish(population[0], generation, start_time, report);
                    return report;
                }

                // --- Local repair terhadap individu terbaik generasi ini ---
                // Menutup gap kecil yang tidak bisa dijembatani satu bit flip biasa.
                Individual repaired = population[0];
                if (local_repair(repaired)) {
                    finish(repaired, generation, start_time, report);
                    return report;
                }

                std::vector<Individual> next_gen;
                next_gen.reserve(P);

                for (int i = 0; i < elitism && i < P; ++i)
                    next_gen.push_back(population[i]);

                std::uniform_int_distribution<int> sel_d(0, P - 1);
                std::uniform_real_distribution<double> uni(0.0, 1.0);

                while ((int)next_gen.size() < P) {
                    const Individual& p1 = tournament(population, sel_d);
                    const Individual& p2 = tournament(population, sel_d);

                    Individual c1 = p1;
                    Individual c2 = p2;

                    if (uni(rng) < crossover_rate) crossover(p1, p2, c1, c2);

                    mutate(c1);
                    repair_k_window(c1);
                    evaluate(c1);
                    if (try_finish_if_solved(c1, generation, start_time, report)) return report;
                    next_gen.push_back(std::move(c1));

                    if ((int)next_gen.size() < P) {
                        mutate(c2);
                        repair_k_window(c2);
                        evaluate(c2);
                        if (try_finish_if_solved(c2, generation, start_time, report)) return report;
                        next_gen.push_back(std::move(c2));
                    }
                }

                population = std::move(next_gen);
                if (diagnostic && (generation % diagnostic_interval == 0))
                    print_diagnostic(population, generation, start_time);
            }

            std::sort(population.begin(), population.end(),
                      [](const Individual& x, const Individual& y) {
                          return x.fitness < y.fitness;
                      });

            // Upaya terakhir: local repair pada individu terbaik sebelum menyerah.
            Individual final_best = population[0];
            local_repair(final_best);

            finish(final_best, generation, start_time, report);
            return report;
        }

    private:
        const std::vector<u128>& a;
        u128 target;
        int P;
        int G;
        double crossover_rate;
        double mutation_rate;
        int elitism;
        std::mt19937_64 rng;
        int k_min, k_max;
        bool diagnostic;
        int diagnostic_interval;
        std::vector<std::pair<int,int>> repair_rc_list;

        static double choose(double n, int r) {
            if (r < 0 || n < r) return 0.0;
            double res = 1.0;
            for (int i = 0; i < r; ++i) res *= (n - i) / (i + 1);
            return res;
        }
        u64 fitness_evaluations = 0;
        u64 crossover_operations = 0;
        u64 mutation_bit_flips = 0;
        u64 mutated_individuals = 0;
        u64 local_repair_attempts = 0;
        u64 local_repair_1swap_success = 0;
        u64 local_repair_2swap_success = 0;

        // Inisialisasi chromosome dengan popcount di dalam k_window (bukan uniform random murni).
        void init_biased(Individual& ind) {
            int n = (int)a.size();
            ind.chromosome.assign(n, 0);
            std::uniform_int_distribution<int> kdist(k_min, k_max);
            int k = kdist(rng);
            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (int i = 0; i < k; ++i) ind.chromosome[idx[i]] = 1;
        }

        // Jika popcount keluar dari [k_min, k_max] setelah mutasi/crossover, perbaiki.
        void repair_k_window(Individual& ind) {
            int n = (int)ind.chromosome.size();
            int cnt = 0;
            for (uint8_t b : ind.chromosome) cnt += b;
            if (cnt >= k_min && cnt <= k_max) return;

            std::vector<int> ones, zeros;
            ones.reserve(n); zeros.reserve(n);
            for (int i = 0; i < n; ++i) (ind.chromosome[i] ? ones : zeros).push_back(i);

            if (cnt < k_min) {
                std::shuffle(zeros.begin(), zeros.end(), rng);
                int need = k_min - cnt;
                for (int i = 0; i < need && i < (int)zeros.size(); ++i)
                    ind.chromosome[zeros[i]] = 1;
            } else if (cnt > k_max) {
                std::shuffle(ones.begin(), ones.end(), rng);
                int excess = cnt - k_max;
                for (int i = 0; i < excess && i < (int)ones.size(); ++i)
                    ind.chromosome[ones[i]] = 0;
            }
        }

        void evaluate(Individual& ind) {
            ++fitness_evaluations;
            u128 sum = 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (ind.chromosome[i]) sum += a[i];
            ind.sum = sum;
            ind.fitness = (sum >= target) ? (sum - target) : (target - sum);
        }

        // Coba tutup gap dengan membuang r elemen dan menambah c elemen (r boleh != c,
        // jadi popcount BOLEH berubah). Temuan empiris: kasus paling umum justru
        // (r=0,c=1) atau (r=1,c=0) -- cuma nambah/buang SATU elemen tanpa pasangan
        // penukar -- bukan swap simetris m-untuk-m seperti versi awal.
        bool local_repair(Individual& ind) {
            if (ind.sum == target) return true;
            ++local_repair_attempts;

            std::vector<int> ones, zeros;
            for (size_t i = 0; i < ind.chromosome.size(); ++i)
                (ind.chromosome[i] ? ones : zeros).push_back((int)i);

            for (auto& rc : repair_rc_list) {
                int r = rc.first, c = rc.second;
                std::vector<int> combI, combJ;
                if (try_rc_swap(ind.sum, ones, zeros, r, c, combI, combJ)) {
                    for (int idx : combI) ind.chromosome[ones[idx]] = 0;
                    for (int idx : combJ) ind.chromosome[zeros[idx]] = 1;
                    u128 removed = 0, added = 0;
                    for (int idx : combI) removed += a[ones[idx]];
                    for (int idx : combJ) added += a[zeros[idx]];
                    ind.sum = ind.sum - removed + added;
                    ind.fitness = 0;
                    if (r == 0 && c == 1) ++local_repair_1swap_success;
                    else ++local_repair_2swap_success;
                    return true;
                }
            }
            return false;
        }

        // Cari r indeks dari 'ones' untuk dibuang dan c indeks dari 'zeros' untuk ditambah
        // sedemikian sum - removedSum + addedSum == target. Exhaustive, dibatasi ukuran
        // kombinasi lewat repair_rc_list yang disusun di constructor.
        bool try_rc_swap(u128 sum, const std::vector<int>& ones, const std::vector<int>& zeros,
                          int r, int c, std::vector<int>& outCombI, std::vector<int>& outCombJ) {
            int ks = (int)ones.size(), zs = (int)zeros.size();
            if (ks < r || zs < c) return false;
            std::vector<int> combI(r), combJ(c);

            auto search_zeros = [&](u128 need) -> bool {
                std::function<bool(int,int,u128)> recurJ = [&](int start2, int depth2, u128 addedSum)->bool {
                    if (depth2 == c) return addedSum == need;
                    for (int q = start2; q < zs; ++q) {
                        combJ[depth2] = q;
                        if (recurJ(q + 1, depth2 + 1, addedSum + a[zeros[q]])) return true;
                    }
                    return false;
                };
                return recurJ(0, 0, 0);
            };

            if (r == 0) {
                if (target < sum) return false;
                if (search_zeros(target - sum)) { outCombI = combI; outCombJ = combJ; return true; }
                return false;
            }

            std::function<bool(int,int,u128)> recurI = [&](int start, int depth, u128 removedSum)->bool {
                if (depth == r) {
                    if (target + removedSum < sum) return false;
                    u128 need = target + removedSum - sum;
                    return search_zeros(need);
                }
                for (int p = start; p < ks; ++p) {
                    combI[depth] = p;
                    if (recurI(p + 1, depth + 1, removedSum + a[ones[p]])) return true;
                }
                return false;
            };

            if (recurI(0, 0, 0)) { outCombI = combI; outCombJ = combJ; return true; }
            return false;
        }

        const Individual& tournament(const std::vector<Individual>& population,
                                     std::uniform_int_distribution<int>& dist) {
            int x = dist(rng), y = dist(rng), z = dist(rng);
            int best = x;
            if (population[y].fitness < population[best].fitness) best = y;
            if (population[z].fitness < population[best].fitness) best = z;
            return population[best];
        }

        void crossover(const Individual& p1, const Individual& p2,
                       Individual& c1, Individual& c2) {
            ++crossover_operations;
            std::uniform_int_distribution<size_t> dist(0, a.size() - 1);
            size_t point = dist(rng);
            for (size_t i = point; i < a.size(); ++i) {
                c1.chromosome[i] = p2.chromosome[i];
                c2.chromosome[i] = p1.chromosome[i];
            }
        }

        void mutate(Individual& ind) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            u64 flips = 0;
            for (auto& bit : ind.chromosome)
                if (dist(rng) < mutation_rate) { bit ^= 1; ++flips; }
            if (flips > 0) ++mutated_individuals;
            mutation_bit_flips += flips;
        }

        bool try_finish_if_solved(Individual& ind, int generation,
                                   std::chrono::steady_clock::time_point start_time,
                                   Report& report) {
            if (ind.sum == target) {
                finish(ind, generation, start_time, report);
                return true;
            }
            return false;
        }

        void print_diagnostic(const std::vector<Individual>& population, int generation,
                              std::chrono::steady_clock::time_point start_time) {
            if (population.empty()) return;
            u128 best = population[0].fitness;
            long double avg = 0.0L;
            u64 ones = 0;
            u64 total_bits = (u64)population.size() * (u64)a.size();
            for (const auto& ind : population) {
                if (ind.fitness < best) best = ind.fitness;
                avg += (long double)ind.fitness;
                for (uint8_t bit : ind.chromosome) ones += bit ? 1 : 0;
            }
            double diversity = 0.0;
            if (total_bits > 0) {
                long double p1 = (long double)ones / (long double)total_bits;
                long double p0 = 1.0L - p1;
                diversity = (double)(2.0L * std::min(p0, p1) * 100.0L);
            }
            int best_k = 0;
            for (uint8_t bit : population[0].chromosome) best_k += bit ? 1 : 0;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            std::cout << "[GA GEN " << generation << "] "
                      << "best_fitness=" << best
                      << " avg_fitness=" << std::fixed << std::setprecision(2) << (double)(avg / population.size())
                      << " best_k=" << best_k
                      << " diversity=" << diversity << "%"
                      << " evals=" << fitness_evaluations
                      << " crossovers=" << crossover_operations
                      << " mutation_flips=" << mutation_bit_flips
                      << " mutated_individuals=" << mutated_individuals
                      << " repair_attempts=" << local_repair_attempts
                      << " repair_1swap_ok=" << local_repair_1swap_success
                      << " repair_2swap_ok=" << local_repair_2swap_success
                      << " elapsed=" << elapsed << "s\n";
        }

        void finish(const Individual& ind, int generation, std::chrono::steady_clock::time_point start_time,
                    Report& report) {
            auto end_time = std::chrono::steady_clock::now();
            report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            report.peak_ram_mb = get_current_peak_ram_mb();
            report.threads_used = 1;
            report.solved = (ind.sum == target);
            report.generations_completed = generation;
            report.fitness_evaluations = fitness_evaluations;
            report.crossover_operations = crossover_operations;
            report.mutation_bit_flips = mutation_bit_flips;
            report.mutated_individuals = mutated_individuals;
            report.local_repair_attempts = local_repair_attempts;
            report.local_repair_1swap_success = local_repair_1swap_success;
            report.local_repair_2swap_success = local_repair_2swap_success;
            report.population_used = P;
            report.elitism_used = elitism;
            report.best_fitness = ind.fitness;
            report.best_k = 0;
            for (uint8_t bit : ind.chromosome) report.best_k += bit ? 1 : 0;
            if (report.solved) {
                Witness w;
                for (size_t i = 0; i < ind.chromosome.size(); ++i)
                    if (ind.chromosome[i]) w.indices.push_back((int)i);
                std::string msg;
                if (verify_witness(a, target, -1, w, msg)) {
                    report.witness = w;
                    report.verification_msg = msg;
                } else {
                    report.verification_msg = msg;
                }
            } else {
                report.verification_msg = "GA selesai tanpa solusi (heuristic, tidak lengkap; sudah dicoba local repair).";
            }
        }
    };

    // Population & elitism adaptif terhadap N (tetap dibatasi supaya tidak meledak).
    static int adaptive_population(size_t n) {
        // Dikalibrasi ke data eksperimen manual: N=16 -> 500, N=24 -> ~2000 (4x, cocok temuan 5%->30%).
        double scale = std::pow(2.0, ((double)n - 16.0) / 4.0);
        int pop = (int)std::round(500.0 * scale);
        pop = std::max(200, std::min(pop, 20000)); // batas atas supaya tidak meledak durasi per-generasi
        return pop;
    }
    static int adaptive_elitism(int population) {
        int e = (int)std::round(population * 0.01); // 1% dari populasi
        return std::max(2, e);
    }

    static Report solve(const Instance& inst, unsigned num_threads, double time_limit_s, bool diagnostic = false, int diagnostic_interval = 100) {
        int pop = adaptive_population(inst.elements.size());
        int elit = adaptive_elitism(pop);
        // Generations: bukan lagi hard cap 10.000. Diberi batas atas sangat besar,
        // sehingga yang benar-benar mengontrol durasi adalah time_limit_s (argumen CLI).
        int max_gen = 100000000;
        GeneticAlgorithm ga(
            inst.elements, inst.target,
            pop, max_gen, 0.8, 1.0 / (double)inst.elements.size(), elit,
            (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
            inst.k_min, inst.k_max,
            diagnostic, diagnostic_interval);
        return ga.solve(time_limit_s);
    }
}

// ============================================================================
// Solver Router (selalu EngineGA)
// ============================================================================
struct Solver {
    static Report solve(const Instance& inst, unsigned num_threads, double time_limit_s,
                        bool proof_mode, bool verbose, bool diagnostic = false, int diagnostic_interval = 100) {
        if (verbose) std::cout << "[ENGINE] Mode: Genetic Algorithm + Local Repair + k-window (EngineGA v2)\n";
        return EngineGA::solve(inst, num_threads, time_limit_s, diagnostic, diagnostic_interval);
    }
};

// ============================================================================
// CLI Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: adaptssp4.exe <file_or_string> [target] [threads] [timelimit_s] [--calibrate_s=N] [--proof] [--diagnostic] [--diag_interval=N]\n";
        return 1;
    }

    std::string file_path = argv[1];
    std::string tgt_str = "";
    unsigned num_threads = 0;
    double time_limit_s = 0.0;
    double calibrate_s = 0.0;
    bool proof_mode = false;
    bool diagnostic = false;
    int diagnostic_interval = 100;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--calibrate_s=", 0) == 0) {
            calibrate_s = std::atof(arg.substr(14).c_str());
        } else if (arg == "--proof") {
            proof_mode = true;
        } else if (arg == "--diagnostic") {
            diagnostic = true;
        } else if (arg.rfind("--diag_interval=", 0) == 0) {
            diagnostic_interval = std::max(1, std::atoi(arg.substr(16).c_str()));
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
    std::cout << " GENETIC ALGORITHM SOLVER + LOCAL REPAIR + K-WINDOW (EngineGA v2)\n";
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
                  << " detik, memakai pipeline GA yang sama.\n";
        Report cal = Solver::solve(inst, num_threads, calibrate_s, false, false);
        if (cal.solved) {
            std::cout << "[CALIBRATION] SOLVED sebelum long-run.\n";
            std::cout << "Runtime : " << std::fixed << std::setprecision(2) << cal.runtime_ms / 1000.0 << " s\n";
            std::cout << "Partisi : " << cal.partitions_evaluated << "\n";
            std::cout << "Peak RAM: " << std::fixed << std::setprecision(2) << cal.peak_ram_mb << " MB\n";
            return 0;
        }
    }

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "[RUNNING SOLVER]\n";

    Report report = Solver::solve(inst, num_threads, time_limit_s, proof_mode, true, diagnostic, diagnostic_interval);

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
    std::cout << "Population dipakai   : " << report.population_used << "\n";
    std::cout << "Elitism dipakai      : " << report.elitism_used << "\n";
    std::cout << "Generasi selesai     : " << report.generations_completed << "\n";
    std::cout << "Fitness evaluations  : " << report.fitness_evaluations << "\n";
    std::cout << "Best fitness         : " << report.best_fitness << "\n";
    std::cout << "Best k               : " << report.best_k << "\n";
    std::cout << "Crossover operations : " << report.crossover_operations << "\n";
    std::cout << "Mutation bit flips   : " << report.mutation_bit_flips << "\n";
    std::cout << "Mutated individuals  : " << report.mutated_individuals << "\n";
    std::cout << "Local repair attempts: " << report.local_repair_attempts << "\n";
    std::cout << "  - sukses 1-swap    : " << report.local_repair_1swap_success << "\n";
    std::cout << "  - sukses 2-swap    : " << report.local_repair_2swap_success << "\n";
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
