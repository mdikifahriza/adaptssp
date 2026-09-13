#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdint>

typedef unsigned __int128 u128;
typedef uint64_t u64;

struct HalfEntry {
    u128 sum;
    u64  mask;
};

inline bool compareHalf(const HalfEntry& a, const HalfEntry& b) {
    return a.sum < b.sum;
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

bool load_instance(const std::string& filepath, int& out_n, u128& out_target, std::vector<u128>& out_elements) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) return false;

    out_elements.clear();
    std::string line;
    std::string tgt_str;

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
                out_elements.push_back(parse_u128_dec(cur));
                cur.clear();
            }
        }
        if (!cur.empty()) out_elements.push_back(parse_u128_dec(cur));
    }

    out_n = (int)out_elements.size();
    if (!tgt_str.empty()) out_target = parse_u128_dec(tgt_str);
    return out_n > 0;
}

struct BenchResult {
    int n;
    u128 target;
    bool solved;
    double runtime_ms;
    u64 witness_mask;
    int witness_k;
};

BenchResult solve_mitm(const std::vector<u128>& A, u128 target) {
    BenchResult res;
    res.n = (int)A.size();
    res.target = target;
    res.solved = false;
    res.runtime_ms = 0.0;
    res.witness_mask = 0;
    res.witness_k = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    int n1 = res.n / 2;
    int n2 = res.n - n1;
    size_t size1 = (size_t)1 << n1;

    std::vector<HalfEntry> left_list(size1);
    left_list[0] = {0, 0};

    for (size_t i = 1; i < size1; ++i) {
        int bit = __builtin_ctzll((u64)i);
        size_t prev = i ^ (1ULL << bit);
        left_list[i].sum = left_list[prev].sum + A[bit];
        left_list[i].mask = (u64)i;
    }

    std::sort(left_list.begin(), left_list.end(), compareHalf);

    size_t size2 = (size_t)1 << n2;
    std::vector<u128> right_sums(size2, 0);

    for (size_t j = 1; j < size2; ++j) {
        int bit = __builtin_ctzll((u64)j);
        size_t prev = j ^ (1ULL << bit);
        right_sums[j] = right_sums[prev] + A[n1 + bit];

        u128 s2 = right_sums[j];
        if (s2 > target) continue;
        u128 needed = target - s2;

        HalfEntry dummy{needed, 0};
        auto it = std::lower_bound(left_list.begin(), left_list.end(), dummy, compareHalf);
        if (it != left_list.end() && it->sum == needed) {
            auto t_end = std::chrono::high_resolution_clock::now();
            res.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            res.solved = true;
            res.witness_mask = it->mask | ((u64)j << n1);
            res.witness_k = __builtin_popcountll(res.witness_mask);
            return res;
        }
    }

    HalfEntry dummy{target, 0};
    auto it = std::lower_bound(left_list.begin(), left_list.end(), dummy, compareHalf);
    if (it != left_list.end() && it->sum == target) {
        auto t_end = std::chrono::high_resolution_clock::now();
        res.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        res.solved = true;
        res.witness_mask = it->mask;
        res.witness_k = __builtin_popcountll(res.witness_mask);
        return res;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    res.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return res;
}

int main() {
    std::vector<int> ns = {10, 15, 20, 25, 30, 35, 40};

    std::cout << "========================================================================================" << std::endl;
    std::cout << "        BENCHMARK PERTUMBUHAN WAKTU SUBSET SUM DENSITAS 1.0 (N = 10 s.d. 40)           " << std::endl;
    std::cout << "        Exact Meet-in-the-Middle Solver (Intel Celeron N5100 4 Cores)                  " << std::endl;
    std::cout << "========================================================================================" << std::endl;
    std::cout << std::setw(6) << "N"
              << std::setw(10) << "K_target"
              << std::setw(26) << "Target (T)"
              << std::setw(16) << "Runtime (ms)"
              << std::setw(16) << "Runtime (detik)"
              << std::setw(14) << "Verifikasi"
              << std::setw(12) << "K_witness"
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------------" << std::endl;

    for (int nidx : ns) {
        std::string fn = "density1_n" + std::to_string(nidx) + "_instance.txt";
        int inst_n = 0;
        u128 target = 0;
        std::vector<u128> A;

        if (!load_instance(fn, inst_n, target, A)) {
            std::cerr << "[ERROR]: Gagal membuka berkas: " << fn << std::endl;
            continue;
        }

        int reps = (nidx <= 20) ? 50 : (nidx <= 25 ? 10 : 1);
        double total_ms = 0.0;
        BenchResult last_res;

        for (int r = 0; r < reps; ++r) {
            BenchResult res = solve_mitm(A, target);
            total_ms += res.runtime_ms;
            last_res = res;
        }
        double avg_ms = total_ms / reps;

        u128 check_sum = 0;
        for (int i = 0; i < nidx; ++i) {
            if ((last_res.witness_mask >> i) & 1) {
                check_sum += A[i];
            }
        }
        bool verified = (check_sum == target);

        std::cout << std::setw(6) << nidx
                  << std::setw(10) << (nidx / 2)
                  << std::setw(26) << u128_to_dec(target)
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_ms
                  << std::setw(16) << std::fixed << std::setprecision(6) << (avg_ms / 1000.0)
                  << std::setw(14) << (verified ? "100% VALID" : "ERROR")
                  << std::setw(12) << last_res.witness_k
                  << std::endl;
    }

    std::cout << "========================================================================================" << std::endl;
    return 0;
}
