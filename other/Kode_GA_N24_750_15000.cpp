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
// ENGINE: Genetic Algorithm (Heuristic Subset Sum)
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
                         int elitism, uint64_t seed)
            : a(values), target(target),
              P(population_size), G(max_generations),
              crossover_rate(crossover_rate),
              mutation_rate(mutation_rate > 0.0 ? mutation_rate : 1.0 / (double)values.size()),
              elitism(elitism), rng(seed) {}

        Report solve(double time_limit_s) {
            Report report;
            auto start_time = std::chrono::steady_clock::now();

            auto elapsed_s = [&]() -> double {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            };

            std::vector<Individual> population;
            population.reserve(P);

            for (int i = 0; i < P; ++i) {
                Individual ind;
                ind.chromosome.resize(a.size());
                for (size_t j = 0; j < a.size(); ++j)
                    ind.chromosome[j] = (uint8_t)(rng() & 1ULL);
                evaluate(ind);
                if (ind.sum == target) {
                    finish(ind, 0, start_time, report);
                    return report;
                }
                population.push_back(std::move(ind));
            }

            for (int generation = 1; generation <= G; ++generation) {
                if (time_limit_s > 0.0 && elapsed_s() >= time_limit_s) break;

                std::sort(population.begin(), population.end(),
                          [](const Individual& x, const Individual& y) {
                              return x.fitness < y.fitness;
                          });

                if (population[0].sum == target) {
                    finish(population[0], generation, start_time, report);
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
                    evaluate(c1);
                    if (c1.sum == target) {
                        finish(c1, generation, start_time, report);
                        return report;
                    }
                    next_gen.push_back(std::move(c1));

                    if ((int)next_gen.size() < P) {
                        mutate(c2);
                        evaluate(c2);
                        if (c2.sum == target) {
                            finish(c2, generation, start_time, report);
                            return report;
                        }
                        next_gen.push_back(std::move(c2));
                    }
                }

                population = std::move(next_gen);
            }

            std::sort(population.begin(), population.end(),
                      [](const Individual& x, const Individual& y) {
                          return x.fitness < y.fitness;
                      });

            finish(population[0], G, start_time, report);
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

        void evaluate(Individual& ind) {
            u128 sum = 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (ind.chromosome[i]) sum += a[i];
            ind.sum = sum;
            ind.fitness = (sum >= target) ? (sum - target) : (target - sum);
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
            std::uniform_int_distribution<size_t> dist(0, a.size() - 1);
            size_t point = dist(rng);
            for (size_t i = point; i < a.size(); ++i) {
                c1.chromosome[i] = p2.chromosome[i];
                c2.chromosome[i] = p1.chromosome[i];
            }
        }

        void mutate(Individual& ind) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (auto& bit : ind.chromosome)
                if (dist(rng) < mutation_rate) bit ^= 1;
        }

        void finish(const Individual& ind, int generation, std::chrono::steady_clock::time_point start_time,
                    Report& report) {
            auto end_time = std::chrono::steady_clock::now();
            report.runtime_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            report.peak_ram_mb = get_current_peak_ram_mb();
            report.threads_used = 1;
            report.solved = (ind.sum == target);
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
                report.verification_msg = "GA selesai tanpa solusi (heuristic, tidak lengkap).";
            }
        }
    };

    static Report solve(const Instance& inst, unsigned num_threads, double time_limit_s) {
        GeneticAlgorithm ga(
            inst.elements, inst.target,
            750, 15000, 0.8, 0.09375, 3,
            (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
        return ga.solve(time_limit_s);
    }
}

// ============================================================================
// Solver Router (selalu EngineGA)
// ============================================================================
struct Solver {
    static Report solve(const Instance& inst, unsigned num_threads, double time_limit_s,
                        bool proof_mode, bool verbose) {
        if (verbose) std::cout << "[ENGINE] Mode: Genetic Algorithm (EngineGA, all N)\n";
        return EngineGA::solve(inst, num_threads, time_limit_s);
    }
};

// ============================================================================
// CLI Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: adaptssp3.exe <file_or_string> [target] [threads] [timelimit_s] [--calibrate_s=N] [--proof]\n";
        return 1;
    }

    std::string file_path = argv[1];
    std::string tgt_str = "";
    unsigned num_threads = 0;
    double time_limit_s = 0.0;
    double calibrate_s = 0.0;
    bool proof_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--calibrate_s=", 0) == 0) {
            calibrate_s = std::atof(arg.substr(14).c_str());
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
    std::cout << " GENETIC ALGORITHM SOLVER (EngineGA - ALL N)\n";
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

    Report report = Solver::solve(inst, num_threads, time_limit_s, proof_mode, true);

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