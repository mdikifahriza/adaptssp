// ssp_pipeline_AG.cpp
// ============================================================================
// SUBSET SUM PREPROCESSING / PRUNING PIPELINE — Grup A sampai G
// Grup A: Structural Gate
// Grup B: Magnitude-Based Pruning (B1 tail-extreme, B3 forced include/exclude)
// Grup C: Residue-Based Pruning (di-gate via probe murah p=2,3)
// Grup D: Quick-Win eligibility report (INFORMASI SAJA, tidak menjalankan solve)
// Grup E: Metadata & kalibrasi untuk solver lanjutan (HGJ/BCJ/MITM/dll)
// Grup F: Profil Kepadatan Per-k (breakdown estimasi Monte-Carlo per nilai k,
//         bukan agregat) -- menunjukkan k mana yang paling "layak dicoba dulu"
// Grup G: Exact Residue-Count DP Per-k (menghitung PERSIS, bukan estimasi,
//         berapa banyak subset ukuran-k yang lolos filter sum ≡ T (mod M)
//         untuk modulus M yang bisa diperbesar; dipakai untuk memvalidasi/
//         menyangkal apakah residue pruning (Grup C) masih bisa diperdalam)
//
// TIDAK ADA exact-solver di file ini — tujuannya murni menunjukkan pemangkasan
// yang terjadi + metadata untuk memilih solver (HGJ/BCJ/dissection/dll).
// ============================================================================

#include <bits/stdc++.h>
using namespace std;

using u128 = unsigned __int128;

// ---------------------------------------------------------------------------
// Helper u128
// ---------------------------------------------------------------------------
static string u128_to_string(u128 v) {
    if (v == 0) return "0";
    string s;
    while (v > 0) { s.push_back('0' + (int)(v % 10)); v /= 10; }
    reverse(s.begin(), s.end());
    return s;
}
static double u128_to_double(u128 v) {
    uint64_t hi = (uint64_t)(v >> 64);
    uint64_t lo = (uint64_t)(v & ~0ULL);
    return (double)hi * 18446744073709551616.0 /*2^64*/ + (double)lo;
}
static u128 gcd128(u128 a, u128 b) { while (b) { a %= b; swap(a, b); } return a; }
static u128 parse_u128(const string& s) {
    u128 v = 0;
    for (char c : s) { if (c < '0' || c > '9') continue; v = v * 10 + (u128)(c - '0'); }
    return v;
}

// ---------------------------------------------------------------------------
// Struktur dasar
// ---------------------------------------------------------------------------
struct Element { u128 val; int orig_idx; };

enum class StructureKind { None, Superincreasing, GcdReduced, ParityForced, NarrowKWindow, Bimodal, Flat };
static string structureName(StructureKind k) {
    switch (k) {
        case StructureKind::Superincreasing: return "Superincreasing";
        case StructureKind::GcdReduced:      return "GcdReduced (obstruksi GCD)";
        case StructureKind::ParityForced:    return "ParityForced (obstruksi paritas)";
        case StructureKind::NarrowKWindow:   return "NarrowKWindow";
        case StructureKind::Bimodal:         return "Bimodal (dual-cluster)";
        case StructureKind::Flat:            return "Flat/Unstructured (khas density~1 acak)";
        default: return "None";
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
class Pipeline {
public:
    vector<Element> A;                 // selalu terjaga sorted descending by val
    u128 T_orig, T_eff;
    u128 total_sum = 0, min_val = 0, max_val = 0, gcd_val = 0;
    int  n_orig;
    int  odd_count = 0, even_count = 0;
    double density = 0;
    bool complement_applied = false;
    int  k_min = -1, k_max = -1;
    long long feasible_k_count = 0;
    double window_ratio_pct = 0;
    StructureKind structure_kind = StructureKind::None;

    bool unsat_detected = false;
    string unsat_reason;

    vector<u128> forced_include_witness_val;
    vector<int>  forced_include_witness_idx;

    int eliminated_B1_total = 0;
    int forced_include_total = 0;

    bool residue_uniform = false;
    double residue_yield_p2 = 0, residue_yield_p3 = 0;
    int residue_k_eliminated = 0;

    double density_eff = 0;
    string regime_tag = "UNKNOWN";
    double monte_carlo_expected_solutions = 0;
    string solution_count_tag = "UNKNOWN";
    double alpha_recommended = 0;

    // Grup F: kandidat k dgn share tertinggi (peak densitas)
    int    peak_k = -1;
    double peak_k_share_pct = 0;

    // Grup G: modulus yang dipakai & apakah residu terbukti seragam di modulus besar
    long long grupG_modulus = 0;
    bool      grupG_uniform_confirmed = false;

    Pipeline(const vector<u128>& raw, u128 target) : T_orig(target) {
        n_orig = (int)raw.size();
        A.reserve(raw.size());
        for (int i = 0; i < (int)raw.size(); ++i) A.push_back({raw[i], i});
        sort(A.begin(), A.end(), [](const Element& a, const Element& b) { return a.val > b.val; });
        T_eff = target;
    }

    void run() {
        printHeader();
        stageA();
        if (unsat_detected) { printUnsat(); return; }
        stageB();
        if (unsat_detected) { printUnsat(); return; }
        stageC();
        stageD();
        stageE();
        stageF();
        stageG();
        printSummary();
    }

private:
    // ---- util prefix/suffix pada array A saat ini (sorted desc) ----
    vector<u128> prefixSum() const {
        int n = (int)A.size(); vector<u128> P(n + 1, 0);
        for (int i = 0; i < n; ++i) P[i + 1] = P[i] + A[i].val;
        return P; // P[i] = jumlah i elemen TERBESAR
    }
    vector<u128> suffixSum() const {
        int n = (int)A.size(); vector<u128> S(n + 1, 0);
        for (int i = n - 1; i >= 0; --i) S[i] = S[i + 1] + A[i].val;
        return S; // S[i] = jumlah A[i..n-1]  (S[n-m] = jumlah m elemen TERKECIL)
    }

    static bool largestExcl(int idx, int m, const vector<u128>& P, const vector<Element>& A, u128& out) {
        int n = (int)A.size();
        if (m < 0 || m > n - 1) return false;
        if (idx >= m) out = P[m];
        else out = P[m + 1] - A[idx].val;
        return true;
    }
    static bool smallestExcl(int idx, int m, const vector<u128>& S, const vector<Element>& A, u128& out) {
        int n = (int)A.size();
        if (idx < n - m) out = S[n - m];
        else out = S[n - m - 1] - A[idx].val;
        return true;
    }

    void computeCardinalityBounds(u128 target, const vector<u128>& P, const vector<u128>& S,
                                   int& kmin, int& kmax, long long& fcount, double& ratio) {
        int n = (int)A.size();
        kmin = -1; kmax = -1;
        if (target == 0) { kmin = 0; kmax = 0; fcount = 1; ratio = 0; return; }
        if (n == 0) { fcount = 0; ratio = 0; return; }
        for (int k = 1; k <= n; ++k) {
            if (P[k] >= target && kmin == -1) kmin = k;
            if (S[n - k] <= target) kmax = k;
        }
        fcount = (kmin != -1 && kmax != -1 && kmax >= kmin) ? (kmax - kmin + 1) : 0;
        ratio = 100.0 * (double)fcount / max(1, n);
    }

    void printHeader() {
        cout << "======================================================================\n";
        cout << " SUBSET-SUM PREPROCESSING PIPELINE -- GRUP A s/d G\n";
        cout << " TANPA exact solver -- laporan pemangkasan & metadata saja\n";
        cout << "======================================================================\n";
        cout << "n = " << n_orig << "\n";
        cout << "T (original) = " << u128_to_string(T_orig) << "\n\n";
    }
    void printUnsat() {
        cout << "\n>>> UNSAT terdeteksi pada tahap preprocessing: " << unsat_reason << "\n";
        cout << ">>> STOP. Tidak perlu lanjut ke solver apapun.\n";
    }

    // ================= GRUP A: STRUCTURAL GATE =================
    void stageA() {
        cout << "---------------------- GRUP A: STRUCTURAL GATE ----------------------\n";
        int n = (int)A.size();
        total_sum = 0; gcd_val = 0; odd_count = 0; even_count = 0;
        min_val = A.back().val; max_val = A.front().val;
        for (auto& e : A) {
            total_sum += e.val;
            if ((uint64_t)(e.val & 1) == 1) odd_count++; else even_count++;
            gcd_val = gcd128(gcd_val, e.val);
        }
        double log2max = max_val > 0 ? log2(u128_to_double(max_val)) : 0.0;
        density = log2max > 0 ? (double)n / log2max : 0.0;

        cout << "A1. total_sum=" << u128_to_string(total_sum)
             << "  min_val=" << u128_to_string(min_val) << "  max_val=" << u128_to_string(max_val) << "\n";
        cout << "    gcd=" << u128_to_string(gcd_val) << "  odd=" << odd_count << "  even=" << even_count
             << "  density=n/log2(max)=" << fixed << setprecision(4) << density << "\n";

        if (T_orig == 0) { cout << "A2. Target=0 -> trivial SAT.\n"; unsat_detected = false; return; }
        if (T_orig > total_sum) { unsat_detected = true; unsat_reason = "target > total_sum"; }
        else if (gcd_val > 1 && (T_orig % gcd_val != 0)) { unsat_detected = true; unsat_reason = "obstruksi GCD"; }
        else if (odd_count == 0 && (T_orig % 2 != 0)) { unsat_detected = true; unsat_reason = "obstruksi paritas"; }
        cout << "A2. Trivial checks: " << (unsat_detected ? ("UNSAT -> " + unsat_reason) : "lolos") << "\n";
        if (unsat_detected) return;

        if (T_orig * 2 > total_sum) { T_eff = total_sum - T_orig; complement_applied = true; }
        else { T_eff = T_orig; complement_applied = false; }
        cout << "A3. Complement Reduction: " << (complement_applied ? "DITERAPKAN" : "tidak perlu")
             << " -> T_eff=" << u128_to_string(T_eff) << "\n";

        auto P = prefixSum(); auto S = suffixSum();
        computeCardinalityBounds(T_eff, P, S, k_min, k_max, feasible_k_count, window_ratio_pct);
        if (feasible_k_count == 0) { unsat_detected = true; unsat_reason = "cardinality window kosong"; cout << "A4. UNSAT\n"; return; }
        cout << "A4. Cardinality window: k in [" << k_min << "," << k_max << "] ("
             << feasible_k_count << " nilai k, " << setprecision(2) << window_ratio_pct << "% dari n)\n";

        classifyStructure();
        cout << "A5. Struktur: " << structureName(structure_kind) << "\n\n";
    }

    void classifyStructure() {
        int n = (int)A.size();
        structure_kind = StructureKind::None;
        if (n == 0) return;
        { u128 cum = 0; bool full = true;
          for (int i = n - 1; i >= 0; --i) { if (!(A[i].val > cum)) { full = false; break; } cum += A[i].val; }
          if (full) { structure_kind = StructureKind::Superincreasing; return; } }
        if (gcd_val > 1) { structure_kind = StructureKind::GcdReduced; return; }
        if (odd_count == 0 && (T_eff % 2 != 0)) { structure_kind = StructureKind::ParityForced; return; }
        if (feasible_k_count > 0 && (feasible_k_count <= max(3, n / 10) || window_ratio_pct <= 15.0)) {
            structure_kind = StructureKind::NarrowKWindow; return; }
        double minD = u128_to_double(min_val), maxD = u128_to_double(max_val);
        if (n >= 10 && minD > 0 && (maxD / minD >= 20.0)) {
            double max_ratio = 1.0;
            for (int i = 0; i < n - 1; ++i) {
                double denom = max(1.0, u128_to_double(A[i + 1].val));
                double r = u128_to_double(A[i].val) / denom;
                if (r > max_ratio) max_ratio = r;
            }
            if (max_ratio >= 8.0) { structure_kind = StructureKind::Bimodal; return; }
        }
        structure_kind = StructureKind::Flat;
    }

    // ================= GRUP B: MAGNITUDE-BASED PRUNING =================
    void stageB() {
        cout << "---------------------- GRUP B: MAGNITUDE-BASED PRUNING ----------------------\n";
        int pass = 0;
        while (pass < 3) {
            pass++;
            int before = (int)A.size();
            auto P = prefixSum(); auto S = suffixSum();
            int n = (int)A.size();
            vector<char> eliminate(n, 0), forceIncl(n, 0);
            bool contradiction = false;

            for (int idx = 0; idx < n; ++idx) {
                u128 xv = A[idx].val;
                bool coverableIncl = false, reachableExcl = false;
                for (int k = k_min; k <= k_max; ++k) {
                    int m = k - 1;
                    u128 maxIncl, minIncl;
                    if (largestExcl(idx, m, P, A, maxIncl) && smallestExcl(idx, m, S, A, minIncl)) {
                        u128 lo = xv + minIncl, hi = xv + maxIncl;
                        if (T_eff >= lo && T_eff <= hi) coverableIncl = true;
                    }
                    u128 maxExcl, minExcl;
                    if (largestExcl(idx, k, P, A, maxExcl) && smallestExcl(idx, k, S, A, minExcl)) {
                        if (T_eff >= minExcl && T_eff <= maxExcl) reachableExcl = true;
                    }
                    if (coverableIncl && reachableExcl) break;
                }
                if (!coverableIncl && !reachableExcl) { contradiction = true; break; }
                if (!coverableIncl) eliminate[idx] = 1;
                else if (!reachableExcl) forceIncl[idx] = 1;
            }
            if (contradiction) {
                unsat_detected = true; unsat_reason = "elemen tak bisa disertakan/dikecualikan -> UNSAT";
                cout << "  Pass " << pass << ": UNSAT (" << unsat_reason << ")\n"; return;
            }
            vector<Element> next; int b1 = 0, b3 = 0;
            for (int idx = 0; idx < n; ++idx) {
                if (forceIncl[idx]) {
                    T_eff -= A[idx].val;
                    forced_include_witness_val.push_back(A[idx].val);
                    forced_include_witness_idx.push_back(A[idx].orig_idx);
                    b3++;
                } else if (eliminate[idx]) b1++;
                else next.push_back(A[idx]);
            }
            A = next;
            eliminated_B1_total += b1; forced_include_total += b3;
            if (A.empty()) {
                if (T_eff == 0) cout << "  Pass " << pass << ": array habis, T_eff=0 -> SAT\n";
                else { unsat_detected = true; unsat_reason = "array habis tapi T_eff>0"; }
            } else {
                auto P2 = prefixSum(); auto S2 = suffixSum();
                computeCardinalityBounds(T_eff, P2, S2, k_min, k_max, feasible_k_count, window_ratio_pct);
                if (feasible_k_count == 0) { unsat_detected = true; unsat_reason = "k-window kosong stlh pass " + to_string(pass); }
            }
            cout << "  Pass " << pass << ": B1(elim)=" << b1 << " B3(forced)=" << b3 << " sisa n=" << A.size();
            if (!A.empty() && !unsat_detected) cout << ", k in [" << k_min << "," << k_max << "], T_eff=" << u128_to_string(T_eff);
            cout << "\n";
            if (unsat_detected) return;
            if ((int)A.size() == before) { cout << "  (tidak ada progres, hentikan loop)\n"; break; }
        }
        cout << "Grup B total: eliminasi=" << eliminated_B1_total << " forced-include=" << forced_include_total
             << " sisa n_eff=" << A.size() << "\n\n";
    }

    // ================= GRUP C: RESIDUE-BASED PRUNING (di-gate p=2,3) =================
    void stageC() {
        cout << "---------------------- GRUP C: RESIDUE-BASED PRUNING (di-gate) ----------------------\n";
        if (A.empty()) { cout << "(array kosong, skip)\n\n"; return; }
        int orig_window = (int)feasible_k_count;
        residue_k_eliminated = 0;
        for (int p : {2, 3}) {
            int n = (int)A.size();
            vector<uint32_t> dp(n + 1, 0); dp[0] = 1u;
            for (auto& e : A) {
                int r = (int)(e.val % (u128)p);
                for (int k = n - 1; k >= 0; --k) {
                    if (!dp[k]) continue;
                    uint32_t shifted = 0;
                    for (int res = 0; res < p; ++res) if (dp[k] & (1u << res)) shifted |= (1u << ((res + r) % p));
                    dp[k + 1] |= shifted;
                }
            }
            int Tmod = (int)(T_eff % (u128)p);
            int local_elim = 0;
            for (int k = k_min; k <= k_max; ++k) { if (k < 0 || k > n) continue; if (!((dp[k] >> Tmod) & 1u)) local_elim++; }
            double yield = orig_window > 0 ? 100.0 * local_elim / orig_window : 0;
            cout << "C1. Probe p=" << p << ": k-infeasible=" << local_elim << "/" << feasible_k_count
                 << " (" << fixed << setprecision(2) << yield << "%)\n";
            if (p == 2) residue_yield_p2 = yield; else residue_yield_p3 = yield;
            if (yield < 1.0) { cout << "    -> residu seragam, hentikan probe.\n"; residue_uniform = true; break; }
            int newmin = -1, newmax = -1;
            for (int k = k_min; k <= k_max; ++k) if ((dp[k] >> Tmod) & 1u) { if (newmin == -1) newmin = k; newmax = k; }
            if (newmin == -1) { unsat_detected = true; unsat_reason = "residue sieve p=" + to_string(p) + " hapus seluruh window"; cout << "    -> UNSAT\n"; return; }
            k_min = newmin; k_max = newmax; feasible_k_count = k_max - k_min + 1; residue_k_eliminated += local_elim;
            cout << "    -> window jadi k in [" << k_min << "," << k_max << "]\n";
        }
        cout << "\n";
    }

    // ================= GRUP D: QUICK-WIN ELIGIBILITY (info saja) =================
    void stageD() {
        cout << "---------------------- GRUP D: QUICK-WIN ELIGIBILITY (info saja) ----------------------\n";
        int n = (int)A.size();
        bool d1 = (k_max <= 5);
        cout << "D1. Small-K MITM eligible? " << (d1 ? "YA" : "TIDAK") << " (k_max=" << k_max << ")\n";
        long long cost = (long long)n * (long long)max(1, k_max - k_min + 1);
        cout << "D2. Boundary-Swap: estimasi biaya ~O(n*window)=" << cost << " operasi (murah, disarankan dicoba dulu)\n\n";
    }

    // ================= GRUP E: METADATA & KALIBRASI =================
    void stageE() {
        cout << "---------------------- GRUP E: METADATA & KALIBRASI ----------------------\n";
        int n = (int)A.size();
        double log2max = max_val > 0 ? log2(u128_to_double(max_val)) : 0;
        density_eff = (n > 0 && log2max > 0) ? (double)n / log2max : 0;
        regime_tag = (fabs(density_eff - 1.0) <= 0.05) ? "HARD_DENSITY_1_REGIME"
                   : (density_eff < 1.0 ? "LOW_DENSITY_REGIME" : "HIGH_DENSITY_REGIME");

        auto P = prefixSum(); auto S = suffixSum();
        vector<double> logTerms;
        for (int k = k_min; k <= k_max && k <= n; ++k) {
            if (k < 0) continue;
            u128 maxk = P[k], mink = S[n - k];
            double range = u128_to_double(maxk) - u128_to_double(mink) + 1.0;
            if (range < 1.0) range = 1.0;
            double logC = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
            logTerms.push_back(logC - log(range));
        }
        double expected = 0;
        if (!logTerms.empty()) {
            double mx = *max_element(logTerms.begin(), logTerms.end());
            double s = 0; for (double lt : logTerms) s += exp(lt - mx);
            expected = exp(mx) * s;
        }
        monte_carlo_expected_solutions = expected;
        solution_count_tag = (expected < 1.0) ? "UNIQUE_SOLUTION_EXPECTED" : "MULTI_SOLUTION_LIKELY";
        double k_avg = (k_min + k_max) / 2.0;
        alpha_recommended = n > 0 ? k_avg / n : 0;

        cout << "E1. n_eff=" << n << " T_eff=" << u128_to_string(T_eff)
             << " k=[" << k_min << "," << k_max << "]\n";
        cout << "E2. density_eff=" << fixed << setprecision(4) << density_eff << " -> " << regime_tag << "\n";
        cout << "E3. residue_uniform=" << (residue_uniform ? "TRUE" : "FALSE") << "\n";
        cout << "E4. Monte Carlo expected #solusi=" << scientific << expected << fixed << " -> " << solution_count_tag << "\n";
        cout << "E5. alpha_recommended=" << setprecision(4) << alpha_recommended << "\n\n";
    }

    // ================= GRUP F: PROFIL KEPADATAN PER-k =================
    // Memecah estimasi Monte-Carlo Grup E (yang cuma agregat) menjadi angka
    // per nilai k, supaya solver tahu k mana yang paling "layak dicoba dulu"
    // (mis. sebagai urutan prioritas untuk MITM/DFS/branch-and-bound).
    void stageF() {
        cout << "---------------------- GRUP F: PROFIL KEPADATAN PER-k ----------------------\n";
        int n = (int)A.size();
        if (n == 0 || k_min < 0) { cout << "(tidak ada window k valid, skip)\n\n"; return; }
        auto P = prefixSum(); auto S = suffixSum();

        vector<int> ks;
        vector<double> logExp;
        for (int k = k_min; k <= k_max && k <= n; ++k) {
            if (k < 0) continue;
            u128 maxk = P[k], mink = S[n - k];
            double range = u128_to_double(maxk) - u128_to_double(mink) + 1.0;
            if (range < 1.0) range = 1.0;
            double logC = lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
            ks.push_back(k);
            logExp.push_back(logC - log(range));
        }
        if (ks.empty()) { cout << "(window kosong, skip)\n\n"; return; }

        double mx = *max_element(logExp.begin(), logExp.end());
        double denom = 0; for (double v : logExp) denom += exp(v - mx);

        peak_k = -1; peak_k_share_pct = -1;
        for (size_t i = 0; i < ks.size(); ++i) {
            double share = exp(logExp[i] - mx) / denom * 100.0;
            double expk = exp(logExp[i]);
            cout << "  k=" << setw(2) << ks[i]
                 << "  E[#solusi|k]~" << scientific << setprecision(3) << expk << fixed
                 << "  share=" << setprecision(2) << share << "%";
            if (share > peak_k_share_pct) { peak_k_share_pct = share; peak_k = ks[i]; }
            cout << "\n";
        }
        cout << "F1. Rekomendasi urutan prioritas k: mulai dari k=" << peak_k
             << " (share tertinggi=" << setprecision(2) << peak_k_share_pct
             << "%), lalu melebar ke k tetangganya.\n\n";
    }

    // ================= GRUP G: EXACT RESIDUE-COUNT DP PER-k =================
    // Berbeda dari Grup C (yang cuma cek feasible/tidak per residu kecil p=2,3),
    // di sini kita hitung PERSIS berapa banyak subset ukuran-k yang lolos filter
    // sum ≡ T (mod M) untuk modulus M yang jauh lebih besar. Kalau rasio lolos
    // untuk SEMUA k persis ~1/M, berarti residu instance ini benar-benar seragam
    // bahkan di modulus besar -> menambah probe residu lagi TIDAK akan berguna,
    // dan usaha komputasi sebaiknya dialihkan ke pendekatan lain (magnitude/
    // kombinatorial/representation technique seperti HGJ/BCJ).
    void stageG(long long M = 50021 /* modulus prima, bisa diperbesar */) {
        cout << "---------------------- GRUP G: EXACT RESIDUE-COUNT DP PER-k (mod " << M << ") ----------------------\n";
        int n = (int)A.size();
        if (n == 0 || k_min < 0) { cout << "(tidak ada window k valid, skip)\n\n"; return; }
        grupG_modulus = M;

        long long Tmod = (long long)(T_eff % (u128)M);

        // dp[k][r] = banyaknya subset ukuran-k dengan sum % M == r.
        // Pakai double (bukan integer) karena nilai bisa jauh melebihi 2^63
        // (mis. C(88,44) ~ 2.6e25) -- kita hanya butuh MAGNITUDO relatifnya,
        // bukan nilai pastinya sampai digit terakhir.
        vector<vector<double>> dp(n + 1, vector<double>(M, 0.0));
        dp[0][0] = 1.0;
        for (auto& e : A) {
            long long r = (long long)(e.val % (u128)M);
            for (int k = n - 1; k >= 0; --k) {
                for (long long res = 0; res < M; ++res) {
                    double v = dp[k][res];
                    if (v == 0.0) continue;
                    long long nr = res + r; if (nr >= M) nr -= M;
                    dp[k + 1][nr] += v;
                }
            }
        }

        double sum_ratio = 0; int count_ratio = 0;
        bool all_close_to_uniform = true;
        for (int k = k_min; k <= k_max && k <= n; ++k) {
            double cnt = dp[k][Tmod];
            double Cnk = exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0));
            double ratio = (Cnk > 0) ? cnt / Cnk : 0;
            double theoretical = 1.0 / (double)M;
            double dev_pct = theoretical > 0 ? fabs(ratio - theoretical) / theoretical * 100.0 : 0;
            cout << "  k=" << setw(2) << k
                 << "  C(n,k)=" << scientific << setprecision(3) << Cnk << fixed
                 << "  lolos_mod=" << scientific << setprecision(3) << cnt << fixed
                 << "  rasio=" << setprecision(6) << ratio
                 << "  (teori=" << setprecision(6) << theoretical
                 << ", deviasi=" << setprecision(2) << dev_pct << "%)\n";
            sum_ratio += ratio; count_ratio++;
            if (dev_pct > 5.0) all_close_to_uniform = false; // toleransi 5%
        }
        grupG_uniform_confirmed = all_close_to_uniform;
        cout << "G1. Kesimpulan: residu " << (grupG_uniform_confirmed ? "TERBUKTI SERAGAM" : "TIDAK seragam")
             << " bahkan di modulus M=" << M << ".\n";
        if (grupG_uniform_confirmed) {
            cout << "    -> Menambah probe residu (p lebih besar / M lebih besar) TIDAK akan menyempitkan\n"
                 << "       apa pun lebih jauh. Alihkan usaha ke pruning magnitude/kombinatorial atau\n"
                 << "       langsung ke solver representation-technique (HGJ/BCJ).\n";
        } else {
            cout << "    -> Ada deviasi terdeteksi -> residue pruning MASIH bisa diperdalam (naikkan M\n"
                 << "       atau gabungkan beberapa modulus koprima via CRT untuk mempersempit window k lagi).\n";
        }
        cout << "\n";
    }

    void printSummary() {
        cout << "======================================================================\n";
        cout << " RINGKASAN UNTUK PEMILIHAN SOLVER (Grup A-G)\n";
        cout << "======================================================================\n";
        cout << "n_orig=" << n_orig << " -> n_eff=" << A.size()
             << " (B1=" << eliminated_B1_total << ", B3=" << forced_include_total
             << ", residue k-narrow=" << residue_k_eliminated << ")\n";
        cout << "k-window akhir: [" << k_min << ", " << k_max << "]\n";
        cout << "Struktur: " << structureName(structure_kind) << "\n";
        cout << "Regime: " << regime_tag << "\n";
        cout << "Residu seragam (Grup C, p<=3): " << (residue_uniform ? "YA" : "TIDAK") << "\n";
        cout << "Residu seragam (Grup G, mod " << grupG_modulus << "): " << (grupG_uniform_confirmed ? "YA (terbukti)" : "TIDAK") << "\n";
        cout << "k prioritas tertinggi (Grup F): k=" << peak_k << " (share=" << setprecision(2) << peak_k_share_pct << "%)\n";
        cout << "Estimasi jumlah solusi: " << scientific << monte_carlo_expected_solutions << fixed
             << " -> " << solution_count_tag << "\n";
        cout << "alpha_recommended: " << setprecision(4) << alpha_recommended << "\n";
        cout << "======================================================================\n";
    }
};

// ---------------------------------------------------------------------------
// main: ganti array & target di bawah dengan instance ASLI kamu, atau baca
// dari file (satu angka desimal per baris) lewat argumen argv[1].
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    u128 T = parse_u128("10227952868229738527976647602");
    vector<string> raw_str = {
        "162482945245137653673851293","230532858025543336293974267","181648769856125575167861072",
        "164578759355716168703444073","222407672658605386588277358","216279783582555360616714163",
        "222965563550029152509839606","240838358759887821898472746","156753787899533751173436811",
        "204154669427527354381579019","260044939594563940393763706","221378880434588920842457993",
        "258913477941435961317255779","272324303850133066898633709","236607745847281418066819953",
        "193374331877540031681584742","271893237190362335035661644","245476791870541087984336872",
        "266663122959203804772103983","244303749045391385614060174","186000934555745500297441016",
        "295066191230814255051699159","267652605802252621034829615","264693462793680400014357225",
        "237368140310514137911410862","205313114606579210760228308","238287299311386514911833970",
        "222710115573231127762788804","172054474943871567992468172","164677050984639135375090937",
        "278896993552283077824707406","220038519916322766436608670","252130590857988911692095726",
        "309247095806991700130181487","198959119112004085568630962","231070006268555028706832350",
        "222618985061253360237091991","197554114339884194560261604","182877598756119210114437288",
        "174403387787630333389876456","158295715692187368067698413","190194559324398944149507495",
        "189266219909048409339073171","203689952206084251687506696","270463231395808671341205129",
        "154919499916530255038595614","305958643970118257181224628","249913515220445430697953337",
        "229288639579650985051340599","176162750054827656343052001","194477734749262692689492404",
        "236773555539035189000793520","220290887943642569728246943","216995740986211888342549290",
        "278224628108609434106974950","192191488190832606407809460","174557558817727892633861661",
        "176713794720217127278619318","172963048203601012107579502","257005182487248148567338491",
        "228401378052425220343706899","304971011653249804672846139","195691040074461158707837242",
        "301027096059057655826604378","301120551108980067685175592","213671521806784221851164573",
        "285833744803803865229647571","279347753489493026862655982","231697914430013039713882718",
        "285308543603551655006066301","297918961765612442214797942","183605940323447025615449000",
        "280521558658183264890740160","220894087564992823654808087","272028991250417018768374480",
        "275572201650584157739106599","243022856609974356845411533","213513041175013490328764281",
        "170260792523705748168402874","191845610859345506576257686","180115145812660773723079294",
        "252654234982467469694907955","217957656987145831329622229","251977704211332035553470811",
        "277232284660736474222331357","252594613896286954152827842","296575000812967085900705667",
        "177415584363438276834504013"
    };
    if (argc > 1) {
        ifstream fin(argv[1]);
        if (fin) { raw_str.clear(); string line; while (getline(fin, line)) if (!line.empty()) raw_str.push_back(line); }
    }
    vector<u128> raw; raw.reserve(raw_str.size());
    for (auto& s : raw_str) raw.push_back(parse_u128(s));

    cout << "[INSTANCE] n=" << raw.size() << "\n\n";
    Pipeline pipe(raw, T);
    pipe.run();
    return 0;
}