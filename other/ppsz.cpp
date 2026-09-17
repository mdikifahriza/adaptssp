// ============================================================================
// ssp_via_3sat_ppsz.cpp
//
// Pipeline:
//   1) Reduksi instance Subset Sum Problem (SSP) -> formula 3-CNF
//      (Tseitin encoding atas sirkuit ripple-carry adder, dengan folding
//       konstanta sehingga term "x_i * a_i" tidak butuh gerbang MUX).
//   2) Selesaikan formula 3-CNF dengan algoritma keluarga PPSZ
//      (Paturi-Pudlak-Saks-Zane): urutan variabel acak + "bounded implication".
//      NOTE JUJUR: implementasi ini memakai D=1 (hanya klausa yang menjadi unit
//      terhadap prefix yang sudah diproses) -- ini persis kasus dasar PPZ,
//      cikal-bakal PPSZ. PPSZ asli memakai D>1 (resolusi terbatas lebar D atas
//      beberapa klausa sekaligus), yang jauh lebih rumit untuk diimplementasi
//      secara umum. D=1 di sini sudah cukup untuk mendemonstrasikan pipeline
//      reduksi + solving + decoding secara end-to-end dan tetap benar secara
//      Monte Carlo (probabilitas sukses > 0 pada setiap percobaan jika formula
//      satisfiable, dan repetisi menaikkan probabilitas ke arah 1).
//   3) Decode assignment 3-SAT kembali ke subset elemen SSP, lalu verifikasi
//      independen (sum(subset) == target).
//
// Kompilasi:
//   g++ -O2 -std=c++17 -pthread ssp_via_3sat_ppsz.cpp -o ssp_via_3sat_ppsz
// ============================================================================

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
#include <random>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <array>

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
    for (char c : str) if (c >= '0' && c <= '9') val = val * 10 + (u128)(c - '0');
    return val;
}
inline int bits_needed(u128 v) {
    int b = 0;
    while (v > 0) { v >>= 1; b++; }
    return std::max(b, 1);
}
inline bool get_bit_u128(u128 v, int idx) {
    return (bool)((v >> idx) & (u128)1);
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

// ============================================================================
// Witness + verifikasi independen (tidak bergantung pada solver)
// ============================================================================
struct Witness {
    std::vector<int> indices;
    std::vector<u128> values;
    u128 sum = 0;
};

inline bool verify_witness(const std::vector<u128>& elements, u128 target,
                            Witness& w, std::string& err_msg) {
    if (w.indices.empty()) { err_msg = "Saksi kosong (subset kosong, sum=0)"; }
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
    err_msg = "OK: diverifikasi independen dari solver, sum == target, indeks unik.";
    return true;
}

// ============================================================================
// TAHAP 1: REDUKSI SSP -> 3-CNF  (Tseitin encoding, folding konstanta)
// ============================================================================
namespace ReduceSSPto3SAT {

    // Sebuah "signal" dalam sirkuit: literal variabel CNF, atau konstanta 0/1.
    struct Signal {
        bool isConst;
        bool constVal;   // dipakai jika isConst
        int  lit;        // dipakai jika !isConst; variabel v -> literal +v, negasinya -v

        static Signal Const(bool v) { return Signal{true, v, 0}; }
        static Signal Var(int v)    { return Signal{false, false, v}; }
    };

    struct CNF {
        int numVars = 0;
        std::vector<std::array<int,3>> clauses3; // klausa selalu <=3 literal (0 = slot kosong/duplikat)
        std::vector<int> unitClauses;             // klausa satu literal (hasil folding / target bit)

        int newVar() { return ++numVars; }
        void addClause3(int a, int b, int c) { clauses3.push_back({a, b, c}); }
        void addUnit(int lit) { unitClauses.push_back(lit); }
    };

    inline Signal negateSig(const Signal& s) {
        if (s.isConst) return Signal::Const(!s.constVal);
        return Signal{false, false, -s.lit};
    }

    // s = a XOR b  (Tseitin, 4 klausa 3-literal jika keduanya variabel;
    //               folding langsung jika salah satu konstan)
    inline Signal encodeXOR(const Signal& a, const Signal& b, CNF& cnf) {
        if (a.isConst && b.isConst) return Signal::Const(a.constVal ^ b.constVal);
        if (a.isConst) return a.constVal ? negateSig(b) : b;
        if (b.isConst) return b.constVal ? negateSig(a) : a;
        int s = cnf.newVar();
        int la = a.lit, lb = b.lit;
        cnf.addClause3(-la, -lb, -s);
        cnf.addClause3( la,  lb, -s);
        cnf.addClause3( la, -lb,  s);
        cnf.addClause3(-la,  lb,  s);
        return Signal::Var(s);
    }

    // s = a AND b (3 klausa 3-literal / folding)
    inline Signal encodeAND(const Signal& a, const Signal& b, CNF& cnf) {
        if (a.isConst) return a.constVal ? b : Signal::Const(false);
        if (b.isConst) return b.constVal ? a : Signal::Const(false);
        int s = cnf.newVar();
        int la = a.lit, lb = b.lit;
        cnf.addClause3(-s, la, 0);
        cnf.addClause3(-s, lb, 0);
        cnf.addClause3(s, -la, -lb);
        return Signal::Var(s);
    }

    // s = a OR b (3 klausa 3-literal / folding)
    inline Signal encodeOR(const Signal& a, const Signal& b, CNF& cnf) {
        if (a.isConst) return a.constVal ? Signal::Const(true) : b;
        if (b.isConst) return b.constVal ? Signal::Const(true) : a;
        int s = cnf.newVar();
        int la = a.lit, lb = b.lit;
        cnf.addClause3(s, -la, 0);
        cnf.addClause3(s, -lb, 0);
        cnf.addClause3(-s, la, lb);
        return Signal::Var(s);
    }

    // full adder: (sum, cout) = a + b + cin
    inline void fullAdder(const Signal& a, const Signal& b, const Signal& cin,
                           CNF& cnf, Signal& sumOut, Signal& coutOut) {
        Signal t = encodeXOR(a, b, cnf);
        Signal s = encodeXOR(t, cin, cnf);
        Signal u = encodeAND(a, b, cnf);
        Signal v = encodeAND(t, cin, cnf);
        Signal co = encodeOR(u, v, cnf);
        sumOut = s;
        coutOut = co;
    }

    struct Result {
        CNF cnf;
        std::vector<int> selectorVar; // selectorVar[i] = id variabel x_i (1-based var id CNF), untuk elemen ke-i
        int totalWidth = 0;
    };

    // Reduksi utama: bangun sirkuit "sum(x_i * a_i) == target" lalu Tseitin-encode.
    Result reduce(const Instance& inst) {
        Result R;
        int n = (int)inst.elements.size();
        R.cnf.numVars = 0;

        // Variabel seleksi x_1..x_n
        R.selectorVar.resize(n);
        for (int i = 0; i < n; ++i) R.selectorVar[i] = R.cnf.newVar();

        // Lebar bit: cukup untuk elemen terbesar & target, plus margin carry dari n penjumlahan.
        u128 maxVal = inst.target;
        for (u128 v : inst.elements) if (v > maxVal) maxVal = v;
        int baseBits = bits_needed(maxVal);
        int carryMargin = bits_needed((u128)n) + 1;
        R.totalWidth = baseBits + carryMargin;

        std::vector<Signal> acc(R.totalWidth, Signal::Const(false));

        for (int i = 0; i < n; ++i) {
            std::vector<Signal> term(R.totalWidth, Signal::Const(false));
            for (int j = 0; j < baseBits; ++j) {
                if (get_bit_u128(inst.elements[i], j)) {
                    // bit a_i == 1  =>  term_bit_j = x_i  (tidak butuh gerbang MUX,
                    // karena AND(x_i, 1) = x_i secara langsung)
                    term[j] = Signal::Var(R.selectorVar[i]);
                }
                // bit a_i == 0  =>  term_bit_j = 0 (konstanta, folding otomatis di adder)
            }

            Signal carry = Signal::Const(false);
            std::vector<Signal> newAcc(R.totalWidth);
            for (int j = 0; j < R.totalWidth; ++j) {
                Signal sumBit, coutBit;
                fullAdder(acc[j], term[j], carry, R.cnf, sumBit, coutBit);
                newAcc[j] = sumBit;
                carry = coutBit;
            }
            acc = std::move(newAcc);
        }

        // Paksa acc == target (bit demi bit) via klausa unit.
        for (int j = 0; j < R.totalWidth; ++j) {
            bool tgtBit = get_bit_u128(inst.target, j);
            const Signal& s = acc[j];
            if (s.isConst) {
                if (s.constVal != tgtBit) {
                    // UNSAT struktural: paksa formula kosong yang tak terpenuhi
                    R.cnf.addUnit(1);
                    R.cnf.addUnit(-1);
                }
                // jika cocok, tidak perlu klausa apa pun
            } else {
                R.cnf.addUnit(tgtBit ? s.lit : -s.lit);
            }
        }

        return R;
    }
}

// ============================================================================
// TAHAP 2: SOLVER PPSZ (D=1 / "PPZ-style bounded implication")
// ============================================================================
namespace PPSZ {

    using CNF = ReduceSSPto3SAT::CNF;

    struct FlatCNF {
        int numVars = 0;
        // setiap klausa disimpan sebagai <=3 literal (0 = slot kosong)
        std::vector<std::array<int,3>> clauses;
        std::vector<std::vector<int>> varToClauses; // adjacency: variabel -> indeks klausa yang memuatnya

        static FlatCNF fromCNF(const CNF& c) {
            FlatCNF f;
            f.numVars = c.numVars;
            f.clauses = c.clauses3;
            for (int u : c.unitClauses) f.clauses.push_back({u, 0, 0});
            f.varToClauses.assign(f.numVars + 1, {});
            for (size_t ci = 0; ci < f.clauses.size(); ++ci) {
                for (int lit : f.clauses[ci]) {
                    if (lit == 0) continue;
                    int v = std::abs(lit);
                    if (f.varToClauses[v].empty() || f.varToClauses[v].back() != (int)ci)
                        f.varToClauses[v].push_back((int)ci);
                }
            }
            return f;
        }
    };

    struct FastPrng {
        u64 state;
        FastPrng(u64 seed) : state(seed ? seed : 0x853c49e6748fea9bULL) {}
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
        inline int next_bit() { return (int)(next_u64() & 1ULL); }
    };

    // Satu percobaan PPSZ (D=1): urutan acak, "bounded implication" terhadap
    // klausa yang menjadi unit relatif pada prefix yang sudah diproses;
    // jika tidak terpaksa, tebak acak. Verifikasi penuh di akhir.
    //
    // CATATAN PRAKTIS: urutan seragam-acak murni atas SEMUA variabel (termasuk
    // ratusan variabel bantu Tseitin) membuat sebagian besar percobaan gagal
    // hanya karena variabel bantu ditebak sebelum input pembentuknya diproses
    // (urutan tidak menghormati struktur sirkuit), padahal secara definisi
    // variabel bantu itu SELALU D=1-implied begitu semua inputnya diketahui.
    // Supaya tetap efisien untuk n>~10, permutasi dibatasi: variabel SELEKSI
    // (x_i, id 1..numSelectors) diacak bebas -- ini variabel keputusan yang
    // sesungguhnya -- sedangkan variabel bantu diproses menurut urutan
    // topologis pembuatannya di sirkuit (id menaik), sehingga definisinya
    // selalu berhasil dipaksa oleh implication D=1. Ini varian praktis dari
    // keluarga PPSZ, bukan PPSZ murni dengan urutan seragam-acak atas semua
    // variabel (yang diperlukan untuk klaim bound teoretis 1.307^n).
    inline bool attempt(const FlatCNF& f, int numSelectors, FastPrng& rng,
                         std::vector<int>& perm, std::vector<int>& posInPerm,
                         std::vector<int8_t>& assigned /* -1,0,1 */) {
        int n = f.numVars;
        for (int i = 0; i < numSelectors; ++i) perm[i] = i + 1;
        for (int i = numSelectors - 1; i > 0; --i) {
            int j = (int)rng.next_range((u32)(i + 1));
            std::swap(perm[i], perm[j]);
        }
        for (int i = numSelectors; i < n; ++i) perm[i] = i + 1; // aux vars: urutan topologis
        for (int i = 0; i < n; ++i) posInPerm[perm[i]] = i;
        std::fill(assigned.begin(), assigned.end(), (int8_t)-1);

        for (int step = 0; step < n; ++step) {
            int v = perm[step];
            int forced = -1; // -1 = belum ada paksaan, 0/1 = nilai yang dipaksa

            for (int ci : f.varToClauses[v]) {
                const auto& cl = f.clauses[ci];
                bool allOthersProcessed = true;
                bool clauseSatisfiedByOther = false;
                int vLit = 0;
                for (int lit : cl) {
                    if (lit == 0) continue;
                    int var = std::abs(lit);
                    if (var == v) { vLit = lit; continue; }
                    if (posInPerm[var] >= step) { allOthersProcessed = false; break; }
                    int8_t val = assigned[var]; // sudah pasti terisi (0/1) karena posisinya < step
                    bool litTrue = (lit > 0) == (val == 1);
                    if (litTrue) { clauseSatisfiedByOther = true; break; }
                }
                if (!allOthersProcessed || clauseSatisfiedByOther || vLit == 0) continue;
                // klausa ini "unit" terhadap v pada prefix yang sudah diproses
                int required = (vLit > 0) ? 1 : 0;
                if (forced == -1) forced = required;
                else if (forced != required) return false; // konflik -> percobaan ini gagal
            }

            assigned[v] = (int8_t)(forced != -1 ? forced : rng.next_bit());
        }

        // verifikasi penuh (independen dari mekanisme paksaan di atas)
        for (const auto& cl : f.clauses) {
            bool sat = false;
            for (int lit : cl) {
                if (lit == 0) continue;
                int var = std::abs(lit);
                bool val = (assigned[var] == 1);
                if ((lit > 0) == val) { sat = true; break; }
            }
            if (!sat) return false;
        }
        return true;
    }

    struct Report {
        bool solved = false;
        std::vector<int8_t> assignment; // index by variable id 1..numVars
        u64 attempts = 0;
        double runtime_ms = 0.0;
        double peak_ram_mb = 0.0;
        unsigned threads_used = 0;
    };

    Report solve(const CNF& cnfIn, int numSelectors, unsigned num_threads, double time_limit_s, bool verbose) {
        Report report;
        auto start_time = std::chrono::steady_clock::now();
        FlatCNF f = FlatCNF::fromCNF(cnfIn);

        if (num_threads == 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
        report.threads_used = num_threads;

        std::atomic<bool> solutionFound(false);
        std::atomic<u64> totalAttempts(0);
        std::mutex resultMutex;
        std::vector<int8_t> bestAssignment;

        auto worker = [&](unsigned tid) {
            FastPrng rng(0x9E3779B97F4A7C15ULL ^ ((u64)tid * 0xD1B54A32D192ED03ULL) ^
                         (u64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::vector<int> perm(f.numVars), posInPerm(f.numVars + 1);
            std::vector<int8_t> assigned(f.numVars + 1, -1);

            u64 localAttempts = 0;
            while (!solutionFound.load(std::memory_order_relaxed)) {
                if (time_limit_s > 0.0) {
                    double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (el >= time_limit_s) break;
                }
                localAttempts++;
                bool ok = attempt(f, numSelectors, rng, perm, posInPerm, assigned);
                if (ok) {
                    std::lock_guard<std::mutex> lock(resultMutex);
                    if (!solutionFound.load(std::memory_order_relaxed)) {
                        solutionFound = true;
                        bestAssignment = assigned;
                    }
                    break;
                }
                if ((localAttempts & 0xFFF) == 0) totalAttempts += 0x1000;
            }
            totalAttempts += (localAttempts & 0xFFF);
        };

        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) workers.emplace_back(worker, t);

        auto monitorStart = std::chrono::steady_clock::now();
        double lastPrint = 0.0;
        while (!solutionFound.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - monitorStart).count();
            if (time_limit_s > 0.0 && elapsed >= time_limit_s) break;
            if (verbose && elapsed - lastPrint >= 5.0) {
                lastPrint = elapsed;
                std::cout << "[PPSZ] t=" << std::fixed << std::setprecision(1) << elapsed
                          << "s  percobaan~=" << totalAttempts.load() << "\n";
            }
        }
        for (auto& th : workers) if (th.joinable()) th.join();

        report.runtime_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_time).count();
        report.solved = solutionFound.load();
        report.attempts = totalAttempts.load();
        report.peak_ram_mb = get_current_peak_ram_mb();
        report.assignment = bestAssignment;
        return report;
    }
}

// ============================================================================
// TAHAP 3: DECODE assignment 3-SAT -> solusi SSP
// ============================================================================
Witness decodeToSubsetSum(const std::vector<int8_t>& assignment,
                           const std::vector<int>& selectorVar,
                           const std::vector<u128>& elements) {
    Witness w;
    for (size_t i = 0; i < selectorVar.size(); ++i) {
        int var = selectorVar[i];
        if (!assignment.empty() && assignment[var] == 1) {
            w.indices.push_back((int)i);
            w.values.push_back(elements[i]);
        }
    }
    return w;
}

// ============================================================================
// CLI Main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ssp_via_3sat_ppsz <file_or_string> [target] [threads] [timelimit_s]\n";
        return 1;
    }

    std::string file_path = argv[1];
    std::string tgt_str = (argc > 2) ? argv[2] : "";
    unsigned num_threads = (argc > 3) ? (unsigned)std::stoul(argv[3]) : 0;
    double time_limit_s  = (argc > 4) ? std::atof(argv[4]) : 0.0;

    Instance inst;
    if (!Instance::load(file_path, tgt_str, inst)) {
        std::cerr << "[ERROR] Gagal membaca instance dari: " << file_path << "\n";
        return 1;
    }

    std::cout << "================================================================================\n";
    std::cout << " SSP -> 3-SAT (Tseitin) -> PPSZ (D=1) -> decode\n";
    std::cout << "================================================================================\n";
    std::cout << "N elemen : " << inst.elements.size() << "\n";
    std::cout << "Target   : " << inst.target << "\n";

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "[TAHAP 1] Mereduksi SSP -> 3-CNF ...\n";
    auto t0 = std::chrono::steady_clock::now();
    ReduceSSPto3SAT::Result red = ReduceSSPto3SAT::reduce(inst);
    double reduceMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    size_t numClauses = red.cnf.clauses3.size() + red.cnf.unitClauses.size();
    std::cout << "  Lebar bit akumulator : " << red.totalWidth << "\n";
    std::cout << "  Variabel 3-SAT       : " << red.cnf.numVars
              << " (" << inst.elements.size() << " variabel seleksi + "
              << (red.cnf.numVars - (int)inst.elements.size()) << " variabel bantu)\n";
    std::cout << "  Klausa 3-CNF         : " << numClauses << "\n";
    std::cout << "  Waktu reduksi        : " << std::fixed << std::setprecision(2) << reduceMs << " ms\n";

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "[TAHAP 2] Menjalankan PPSZ (D=1) pada formula hasil reduksi ...\n";
    PPSZ::Report sol = PPSZ::solve(red.cnf, (int)inst.elements.size(), num_threads, time_limit_s, true);

    std::cout << "\n================================================================================\n";
    std::cout << "                              LAPORAN EKSEKUSI\n";
    std::cout << "================================================================================\n";
    std::cout << "Status 3-SAT (PPSZ)  : " << (sol.solved ? "SATISFIABLE (assignment ditemukan)" : "TIDAK ditemukan (habis waktu/percobaan)") << "\n";
    std::cout << "Runtime PPSZ         : " << std::fixed << std::setprecision(2) << sol.runtime_ms / 1000.0 << " detik\n";
    std::cout << "Percobaan (approx)   : " << sol.attempts << "\n";
    std::cout << "Peak RAM             : " << std::fixed << std::setprecision(2) << sol.peak_ram_mb << " MB\n";
    std::cout << "Threads              : " << sol.threads_used << "\n";

    if (sol.solved) {
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "[TAHAP 3] Decode assignment 3-SAT -> solusi SSP ...\n";
        Witness w = decodeToSubsetSum(sol.assignment, red.selectorVar, inst.elements);
        std::string msg;
        bool ok = verify_witness(inst.elements, inst.target, w, msg);

        std::cout << "SAKSI (" << w.indices.size() << " indeks): [";
        for (size_t i = 0; i < w.indices.size(); ++i)
            std::cout << w.indices[i] << (i + 1 < w.indices.size() ? ", " : "");
        std::cout << "]\n";
        std::cout << "Jumlah saksi         : " << w.sum << "\n";
        std::cout << "Target               : " << inst.target << "\n";
        std::cout << "Verifikasi independen: " << msg << "\n";
        std::cout << "Cocok?               : " << (ok ? "YA, solusi SSP valid" : "TIDAK (bug pada reduksi/decode!)") << "\n";
        std::cout << "================================================================================\n";
        return ok ? 0 : 1;
    } else {
        std::cout << "================================================================================\n";
        return 1;
    }
}