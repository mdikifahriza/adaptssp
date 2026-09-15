#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using u64  = uint64_t;
using u128 = unsigned __int128;

// ============================================================================
// Util
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
// Instance
// ============================================================================
struct Instance {
    std::vector<u128> elements;
    u128 target = 0;

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
        return !inst.elements.empty();
    }
};

struct Witness {
    std::vector<int> indices;
    u128 sum = 0;
};

inline bool verify_witness(const std::vector<u128>& elements, u128 target, Witness& w, std::string& msg) {
    if (w.indices.empty()) { msg = "Saksi kosong"; return false; }
    std::vector<int> sorted_idx = w.indices;
    std::sort(sorted_idx.begin(), sorted_idx.end());
    for (size_t i = 1; i < sorted_idx.size(); ++i)
        if (sorted_idx[i] == sorted_idx[i-1]) { msg = "Duplikasi indeks"; return false; }
    u128 acc = 0;
    for (int idx : w.indices) {
        if (idx < 0 || idx >= (int)elements.size()) { msg = "Indeks di luar rentang"; return false; }
        acc += elements[idx];
    }
    w.sum = acc;
    if (acc != target) { msg = "Sum (" + u128_to_string(acc) + ") != target (" + u128_to_string(target) + ")"; return false; }
    msg = "OK: verified, sum == target, indices unik.";
    return true;
}

struct Report {
    bool solved = false;
    double runtime_ms = 0.0;
    double peak_ram_mb = 0.0;
    Witness witness;
    std::string verification_msg;
    // tail-mitm info
    int m_tail = 0, n_head = 0;
    u64 tail_table_size = 0;
    double tail_build_ms = 0.0;
    // ga info
    int generations_completed = 0;
    u64 fitness_evaluations = 0;
    u128 best_fitness = 0; // jarak residual terbaik ke entri tabel tail terdekat
    int population_used = 0;
};

// ============================================================================
// ENGINE: Head-GA + Tail-MITM
// ============================================================================
namespace EngineHeadGA {

    // Pilih ukuran tail m secara adaptif berdasarkan budget memori.
    // Tiap entri tabel tail ~ 24 byte (u128 sum + u64 mask, dibulatkan).
    inline int choose_m(int n, double mem_budget_mb) {
        double max_entries = (mem_budget_mb * 1024.0 * 1024.0) / 24.0;
        int m = (int)std::floor(std::log2(std::max(1.0, max_entries)));
        m = std::min(m, 27);       // guard: >2^27 entri mulai berat untuk di-generate & sort
        m = std::min(m, n);        // tail tidak boleh lebih besar dari N
        return std::max(0, m);
    }

    struct TailTable {
        std::vector<u128> sums;   // terurut ascending
        std::vector<u64>  masks;  // mask sejajar dengan sums (SEBELUM sort, lalu ikut diurutkan)
        int m = 0;

        void build(const std::vector<u128>& a, int offset, int m_) {
            m = m_;
            u64 total = (m == 0) ? 1ULL : (1ULL << m);
            std::vector<std::pair<u128,u64>> entries(total);
            u128 sum = 0;
            u64 prev_gray = 0;
            entries[0] = {0, 0};
            for (u64 i = 1; i < total; ++i) {
                u64 gray = i ^ (i >> 1);
                u64 diff = gray ^ prev_gray;
                int bit = __builtin_ctzll(diff);
                if (gray & diff) sum += a[offset + bit]; else sum -= a[offset + bit];
                entries[i] = {sum, gray};
                prev_gray = gray;
            }
            std::sort(entries.begin(), entries.end(),
                      [](const std::pair<u128,u64>& x, const std::pair<u128,u64>& y) { return x.first < y.first; });
            sums.resize(total); masks.resize(total);
            for (u64 i = 0; i < total; ++i) { sums[i] = entries[i].first; masks[i] = entries[i].second; }
        }

        // Cari entri dengan sum PERSIS sama dengan need. Return index atau -1.
        long long find_exact(u128 need) const {
            size_t lo = 0, hi = sums.size();
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (sums[mid] < need) lo = mid + 1; else hi = mid;
            }
            if (lo < sums.size() && sums[lo] == need) return (long long)lo;
            return -1;
        }

        // Jarak terkecil |sums[i] - need| di sekitar posisi lower_bound(need).
        u128 nearest_distance(u128 need) const {
            size_t lo = 0, hi = sums.size();
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (sums[mid] < need) lo = mid + 1; else hi = mid;
            }
            u128 best = ~(u128)0;
            if (lo < sums.size()) best = std::min(best, sums[lo] >= need ? sums[lo]-need : need-sums[lo]);
            if (lo > 0) best = std::min(best, sums[lo-1] >= need ? sums[lo-1]-need : need-sums[lo-1]);
            return best;
        }
    };

    struct Individual {
        std::vector<uint8_t> chromosome; // panjang = n_head
        u128 head_sum = 0;
        u128 fitness = 0; // jarak residual ke tabel tail terdekat (0 = exact solve)
    };

    class HeadGA {
    public:
        HeadGA(const std::vector<u128>& a, u128 target, int n_head, const TailTable& tail,
               int population_size, double crossover_rate, double mutation_rate, int elitism, uint64_t seed)
            : a(a), target(target), n_head(n_head), tail(tail),
              P(population_size), crossover_rate(crossover_rate),
              mutation_rate(mutation_rate), elitism(elitism), rng(seed) {}

        Report run(double time_limit_s) {
            Report report;
            auto start = std::chrono::steady_clock::now();
            auto elapsed_s = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };

            std::vector<Individual> pop(P);
            for (auto& ind : pop) {
                ind.chromosome.resize(n_head);
                for (int i = 0; i < n_head; ++i) ind.chromosome[i] = (uint8_t)(rng() & 1ULL);
                evaluate(ind);
                if (ind.fitness == 0 && try_close(ind, report)) { finish(report, start); return report; }
            }

            int generation = 0;
            for (;; ++generation) {
                if (time_limit_s > 0.0 && elapsed_s() >= time_limit_s) break;

                std::sort(pop.begin(), pop.end(), [](const Individual& x, const Individual& y) { return x.fitness < y.fitness; });
                if (pop[0].fitness == 0 && try_close(pop[0], report)) { report.generations_completed = generation; finish(report, start); return report; }

                std::vector<Individual> next;
                next.reserve(P);
                for (int i = 0; i < elitism && i < P; ++i) next.push_back(pop[i]);

                std::uniform_int_distribution<int> sel(0, P - 1);
                std::uniform_real_distribution<double> uni(0.0, 1.0);

                while ((int)next.size() < P) {
                    const Individual& p1 = tournament(pop, sel);
                    const Individual& p2 = tournament(pop, sel);
                    Individual c1 = p1, c2 = p2;
                    if (n_head > 0 && uni(rng) < crossover_rate) crossover(p1, p2, c1, c2);
                    mutate(c1); evaluate(c1);
                    if (c1.fitness == 0 && try_close(c1, report)) { report.generations_completed = generation; finish(report, start); return report; }
                    next.push_back(std::move(c1));
                    if ((int)next.size() < P) {
                        mutate(c2); evaluate(c2);
                        if (c2.fitness == 0 && try_close(c2, report)) { report.generations_completed = generation; finish(report, start); return report; }
                        next.push_back(std::move(c2));
                    }
                }
                pop = std::move(next);
            }

            std::sort(pop.begin(), pop.end(), [](const Individual& x, const Individual& y) { return x.fitness < y.fitness; });
            report.generations_completed = generation;
            report.fitness_evaluations = fitness_evaluations;
            report.best_fitness = pop[0].fitness;
            report.solved = false;
            report.verification_msg = "GA (head) tidak menutup gap sampai waktu habis. Best residual-distance=" + u128_to_string(pop[0].fitness);
            finish(report, start);
            return report;
        }

    private:
        const std::vector<u128>& a;
        u128 target;
        int n_head;
        const TailTable& tail;
        int P;
        double crossover_rate, mutation_rate;
        int elitism;
        std::mt19937_64 rng;
        u64 fitness_evaluations = 0;

        void evaluate(Individual& ind) {
            ++fitness_evaluations;
            u128 s = 0;
            for (int i = 0; i < n_head; ++i) if (ind.chromosome[i]) s += a[i];
            ind.head_sum = s;
            if (target >= s) {
                u128 need = target - s;
                ind.fitness = tail.nearest_distance(need);
            } else {
                // head_sum melebihi target: tail tidak bisa "mengurangi", jadi mustahil ditutup dari sini.
                ind.fitness = (s - target) + 1; // penalti > 0 supaya tidak dianggap solved
            }
        }

        // Kalau fitness==0, pastikan exact match beneran ada (bukan cuma nearest==0 karena sama2 nol residual & tail sum 0)
        bool try_close(Individual& ind, Report& report) {
            if (target < ind.head_sum) return false;
            u128 need = target - ind.head_sum;
            long long idx = tail.find_exact(need);
            if (idx < 0) return false;
            Witness w;
            for (int i = 0; i < n_head; ++i) if (ind.chromosome[i]) w.indices.push_back(i);
            u64 mask = tail.masks[idx];
            for (int b = 0; b < tail.m; ++b) if (mask & (1ULL << b)) w.indices.push_back(n_head + b);
            std::string msg;
            if (verify_witness(a, target, w, msg)) {
                report.witness = w;
                report.verification_msg = msg;
                report.solved = true;
                report.fitness_evaluations = fitness_evaluations;
                report.best_fitness = 0;
                return true;
            }
            return false;
        }

        const Individual& tournament(const std::vector<Individual>& pop, std::uniform_int_distribution<int>& dist) {
            int x = dist(rng), y = dist(rng), z = dist(rng);
            int best = x;
            if (pop[y].fitness < pop[best].fitness) best = y;
            if (pop[z].fitness < pop[best].fitness) best = z;
            return pop[best];
        }

        void crossover(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
            std::uniform_int_distribution<int> dist(0, n_head - 1);
            int point = dist(rng);
            for (int i = point; i < n_head; ++i) { c1.chromosome[i] = p2.chromosome[i]; c2.chromosome[i] = p1.chromosome[i]; }
        }

        void mutate(Individual& ind) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (auto& bit : ind.chromosome) if (dist(rng) < mutation_rate) bit ^= 1;
        }

        void finish(Report& report, std::chrono::steady_clock::time_point start) {
            auto end = std::chrono::steady_clock::now();
            report.runtime_ms = std::chrono::duration<double, std::milli>(end - start).count();
            report.peak_ram_mb = get_current_peak_ram_mb();
        }
    };

    static Report solve(const Instance& inst, double time_limit_s, double mem_budget_mb, bool diagnostic) {
        Report report;
        auto t0 = std::chrono::steady_clock::now();
        int n = (int)inst.elements.size();
        int m = choose_m(n, mem_budget_mb);
        int n_head = n - m;

        report.m_tail = m;
        report.n_head = n_head;

        auto t_tail_start = std::chrono::steady_clock::now();
        TailTable tail;
        tail.build(inst.elements, n_head, m); // tail = m elemen TERAKHIR
        report.tail_table_size = tail.sums.size();
        report.tail_build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tail_start).count();

        if (diagnostic) {
            std::cout << "[HEAD-GA+TAIL-MITM CONFIG]\n";
            std::cout << "N total       : " << n << "\n";
            std::cout << "Tail size (m) : " << m << "  (2^" << m << " = " << tail.sums.size() << " entri, "
                      << std::fixed << std::setprecision(1) << (tail.sums.size() * 24.0 / (1024.0*1024.0)) << " MB, "
                      << report.tail_build_ms << " ms build)\n";
            std::cout << "Head size     : " << n_head << " (dijelajah GA)\n";
        }

        if (n_head == 0) {
            // Semua elemen masuk tail -> ini murni MITM exact, cek langsung.
            long long idx = tail.find_exact(inst.target);
            report.population_used = 0;
            if (idx >= 0) {
                Witness w;
                u64 mask = tail.masks[idx];
                for (int b = 0; b < m; ++b) if (mask & (1ULL << b)) w.indices.push_back(b);
                std::string msg;
                verify_witness(inst.elements, inst.target, w, msg);
                report.witness = w; report.verification_msg = msg; report.solved = true;
            } else {
                report.verification_msg = "MITM murni: tidak ada subset yang sum-nya persis target.";
                report.solved = false;
            }
            report.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            report.peak_ram_mb = get_current_peak_ram_mb();
            return report;
        }

        // GA di head, closing presisi didelegasikan ke tail table.
        int population = std::max(200, std::min(20000, (int)std::round(500.0 * std::pow(2.0, (n_head - 16.0) / 4.0))));
        int elitism = std::max(2, (int)std::round(population * 0.01));
        double mutation_rate = 1.0 / std::max(1, n_head);
        HeadGA ga(inst.elements, inst.target, n_head, tail, population, 0.8, mutation_rate, elitism,
                  (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());

        double remaining = time_limit_s > 0.0
            ? std::max(0.0, time_limit_s - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count())
            : 0.0;
        Report ga_report = ga.run(remaining);
        ga_report.m_tail = m;
        ga_report.n_head = n_head;
        ga_report.tail_table_size = tail.sums.size();
        ga_report.tail_build_ms = report.tail_build_ms;
        ga_report.population_used = population;
        ga_report.runtime_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return ga_report;
    }
}

// ============================================================================
// CLI
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: headga_tailmitm <file_or_string> [target] [timelimit_s] [mem_budget_mb] [--diagnostic]\n";
        return 1;
    }
    std::string file_path = argv[1];
    std::string tgt_str = (argc > 2) ? argv[2] : "";
    double time_limit_s = (argc > 3) ? std::atof(argv[3]) : 0.0;
    double mem_budget_mb = (argc > 4) ? std::atof(argv[4]) : 1500.0;
    bool diagnostic = false;
    for (int i = 5; i < argc; ++i) if (std::string(argv[i]) == "--diagnostic") diagnostic = true;

    Instance inst;
    if (!Instance::load(file_path, tgt_str, inst)) {
        std::cerr << "[ERROR] Gagal membaca instance dari: " << file_path << "\n";
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << " HEAD-GA + TAIL-MITM HYBRID SOLVER\n";
    std::cout << "================================================================================\n";
    std::cout << "N elemen : " << inst.elements.size() << "\n";
    std::cout << "Target   : " << inst.target << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    Report report = EngineHeadGA::solve(inst, time_limit_s, mem_budget_mb, diagnostic);

    std::cout << "\n================================================================================\n";
    std::cout << "                              LAPORAN EKSEKUSI\n";
    std::cout << "================================================================================\n";
    std::cout << "Status               : " << (report.solved ? "SOLVED (verified)" : "NOT SOLVED") << "\n";
    std::cout << "Runtime              : " << std::fixed << std::setprecision(3) << report.runtime_ms / 1000.0 << " detik\n";
    std::cout << "Tail size (m)        : " << report.m_tail << " (" << report.tail_table_size << " entri, build " << report.tail_build_ms << " ms)\n";
    std::cout << "Head size            : " << report.n_head << "\n";
    std::cout << "Population dipakai   : " << report.population_used << "\n";
    std::cout << "Generasi selesai     : " << report.generations_completed << "\n";
    std::cout << "Fitness evaluations  : " << report.fitness_evaluations << "\n";
    std::cout << "Best fitness (gap)   : " << report.best_fitness << "\n";
    std::cout << "Peak RAM             : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB\n";
    std::cout << "Pesan verifikasi     : " << report.verification_msg << "\n";

    if (report.solved) {
        std::cout << "SAKSI (" << report.witness.indices.size() << " indeks): [";
        for (size_t i = 0; i < report.witness.indices.size(); ++i)
            std::cout << report.witness.indices[i] << (i + 1 < report.witness.indices.size() ? ", " : "");
        std::cout << "]\nJumlah saksi : " << report.witness.sum << "\n";
        std::cout << "Target       : " << inst.target << "\n";
        std::cout << "Cocok?       : " << (report.witness.sum == inst.target ? "YA, 100% PRESISI" : "TIDAK (bug!)") << "\n";
    }
    std::cout << "================================================================================\n";
    return report.solved ? 0 : 1;
}
