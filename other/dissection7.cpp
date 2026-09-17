// superhybrid_stage2_nested8_EXPERIMENTAL.cpp
//
// STAGE 2 -- EXPERIMENTAL / NOT VALIDATED AS THOROUGHLY AS STAGE 1.
//
// ============================================================
// IMPORTANT HONESTY NOTE -- READ BEFORE TRUSTING THIS FILE
// ============================================================
// This is NOT an implementation of the paper's proven r=7 Dissect3(7,1)
// algorithm (Dinur-Dunkelman-Keller-Shamir, CRYPTO 2012). That algorithm's
// memory saving comes from BIT-SLICING the running-sum counter and doing
// partial-state guessing with carry propagation across slices -- a
// genuinely different and more delicate construction than plain element
// partitioning. Implementing that correctly needs careful small-scale
// testing (n=14, n=21) before it can be trusted, which has not been done
// here.
//
// What THIS file actually does: it takes the Stage 1 idea (avoid
// materializing full cross products; stream sums via heap-based
// ascending/descending generators) and applies it ONE LEVEL DEEPER.
// Instead of building each of the 4 outer quarters by brute-force
// 2^(n/4) mask enumeration (which is the dominant memory cost we
// measured in Stage 1), each quarter is itself split into two eighths
// and built via the SAME AscendingPairGen machinery, bucketed by
// cardinality. This is a valid, commutative-sum-only optimization
// (subset sum allows arbitrary regrouping since + is associative and
// commutative), but it is a heuristic extension, not a complexity
// result with a proof behind it the way Stage 1's memory bound is.
//
// Treat this file as a starting point to experiment with, not as a
// drop-in "faster solver". Test on small N (16, 24, 32) and compare
// against Stage 1's output before trusting it on N=96.
//
// Usage:
//   superhybrid_stage2.exe <file_or_instance_string> [target] [threads] [timelimit_s]
//
// Build:
//   g++ -O3 -march=native -mtune=native -std=c++17 -pthread superhybrid_stage2_nested8_EXPERIMENTAL.cpp -o superhybrid_stage2.exe

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

static std::string u128str(u128 x) {
    if (x == 0) return "0";
    std::string s;
    while (x) { s.push_back(char('0' + x % 10)); x /= 10; }
    std::reverse(s.begin(), s.end());
    return s;
}

static bool parse_u128(const std::string& s, u128& out) {
    if (s.empty()) return false;
    u128 x = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        x = x * 10 + u128(c - '0');
    }
    out = x;
    return true;
}

static std::vector<u128> parse_numbers(const std::string& text) {
    std::vector<u128> v;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            u128 x;
            if (parse_u128(cur, x)) v.push_back(x);
            cur.clear();
        }
    };
    for (char c : text) {
        if (c >= '0' && c <= '9') cur.push_back(c);
        else flush();
    }
    flush();
    return v;
}

struct Instance {
    std::vector<u128> a;
    u128 target = 0;
    u32 k = 0;

    bool load(const std::string& src, const std::string& target_arg = "") {
        std::string text = src;
        std::ifstream f(src);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        auto nums = parse_numbers(text);
        if (nums.empty()) return false;

        if (!target_arg.empty()) {
            if (!parse_u128(target_arg, target)) return false;
            a = std::move(nums);
        } else {
            if (nums.size() < 2) return false;
            target = nums.back();
            nums.pop_back();
            a = std::move(nums);
        }
        if (a.empty()) return false;
        k = static_cast<u32>(a.size() / 2);
        return true;
    }
};

struct Item {
    u128 sum;
    u32 mask;   // mask relative to the LOCAL group it was enumerated over
    u8 card;
};

struct Candidate {
    bool found = false;
    std::vector<u32> bits; // original global indices
    u128 sum = 0;
};

static inline u32 popcnt(u32 x) {
    return static_cast<u32>(__builtin_popcount(x));
}

// ------------------------------------------------------------
// GA guidance (unchanged in spirit from Stage 1 / original)
// ------------------------------------------------------------

struct GAResult {
    std::vector<double> target_frequency;
};

class GAAdvisor {
    const Instance& I;
    std::mt19937_64 rng;

    struct Individual {
        std::vector<u8> x;
        u128 sum = 0;
        u32 card = 0;
        u128 score = 0;
    };

    static u128 absdiff(u128 a, u128 b) { return a >= b ? a - b : b - a; }

    void evaluate(Individual& p) {
        p.sum = 0; p.card = 0;
        for (size_t i = 0; i < I.a.size(); ++i)
            if (p.x[i]) { p.sum += I.a[i]; ++p.card; }
        u128 d = absdiff(p.sum, I.target);
        u128 kd = p.card > I.k ? p.card - I.k : I.k - p.card;
        p.score = d + kd * (u128(1) << 48);
    }

public:
    explicit GAAdvisor(const Instance& inst, u64 seed) : I(inst), rng(seed) {}

    GAResult run(double seconds) {
        const size_t N = I.a.size();
        const size_t POP = 96;
        std::vector<Individual> pop(POP), next(POP);

        for (auto& p : pop) {
            p.x.assign(N, 0);
            std::vector<size_t> idx(N);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (size_t j = 0; j < I.k && j < N; ++j) p.x[idx[j]] = 1;
            evaluate(p);
        }

        Individual best = *std::min_element(pop.begin(), pop.end(),
            [](const Individual& a, const Individual& b) { return a.score < b.score; });

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<u64> hit(N, 0);
        u64 generations = 0;

        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
            ++generations;
            std::sort(pop.begin(), pop.end(),
                [](const Individual& a, const Individual& b) { return a.score < b.score; });
            if (pop.front().score < best.score) best = pop.front();

            next[0] = pop[0]; next[1] = pop[1]; next[2] = pop[2]; next[3] = pop[3];

            for (size_t p = 4; p < POP; ++p) {
                size_t a = rng() % POP, b = rng() % POP;
                const Individual& pa = pop[a].score < pop[b].score ? pop[a] : pop[b];
                a = rng() % POP; b = rng() % POP;
                const Individual& pb = pop[a].score < pop[b].score ? pop[a] : pop[b];

                Individual child;
                child.x.resize(N);
                for (size_t i = 0; i < N; ++i) child.x[i] = (rng() & 1) ? pa.x[i] : pb.x[i];

                double pm = ((generations % 250) == 0) ? 0.10 : 0.025;
                for (size_t i = 0; i < N; ++i)
                    if (double(rng() % 1000000) / 1000000.0 < pm) child.x[i] ^= 1;

                evaluate(child);
                next[p] = std::move(child);
            }
            pop.swap(next);
            for (size_t p = 0; p < std::min<size_t>(12, POP); ++p)
                for (size_t i = 0; i < N; ++i)
                    if (pop[p].x[i]) ++hit[i];
        }

        GAResult r;
        r.target_frequency.resize(N);
        double denom = double(std::max<u64>(1, generations * 12));
        for (size_t i = 0; i < N; ++i) {
            double importance = double(hit[i]) / denom;
            r.target_frequency[i] = 0.65 * importance + 0.35 * double(best.x[i]);
        }
        return r;
    }
};

// ------------------------------------------------------------
// Guided 4-way partition into quarters, then each quarter split
// again into two eighths (unchanged balancing logic, applied twice).
// ------------------------------------------------------------

struct Partition {
    std::vector<size_t> q[4];      // 4 quarters, global indices
    std::vector<size_t> eighth[4][2]; // each quarter split into 2 eighths
};

static void balance_split(const Instance& I,
                          const std::vector<size_t>& src,
                          const GAResult& G,
                          std::mt19937_64& rng,
                          int parts,
                          std::vector<size_t>* out) {
    const size_t sz = src.size();
    const size_t psize = sz / parts;

    std::vector<size_t> idx = src;
    std::shuffle(idx.begin(), idx.end(), rng);

    std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        double ax = G.target_frequency[x] + (double(rng() % 1000) / 1000000.0);
        double ay = G.target_frequency[y] + (double(rng() % 1000) / 1000000.0);
        return ax > ay;
    });

    std::vector<u128> sums(parts, 0);

    for (size_t x : idx) {
        int best = -1;
        for (int g = 0; g < parts; ++g) {
            if (out[g].size() >= psize) continue;
            if (best < 0 || sums[g] < sums[best]) best = g;
        }
        if (best < 0) best = int(rng() % parts);
        out[best].push_back(x);
        sums[best] += I.a[x];
    }

    for (int g = 0; g < parts; ++g) {
        while (out[g].size() > psize) {
            size_t x = out[g].back();
            out[g].pop_back();
            int dst = -1;
            for (int h = 0; h < parts; ++h)
                if (out[h].size() < psize) { dst = h; break; }
            if (dst < 0) break;
            out[dst].push_back(x);
        }
    }
}

static Partition make_partition_8(const Instance& I,
                                  const GAResult& G,
                                  std::mt19937_64& rng) {
    const size_t N = I.a.size();
    std::vector<size_t> all(N);
    std::iota(all.begin(), all.end(), 0);

    Partition P;
    std::vector<size_t> quarters[4];
    balance_split(I, all, G, rng, 4, quarters);

    for (int g = 0; g < 4; ++g) {
        P.q[g] = quarters[g];
        std::vector<size_t> eighths[2];
        balance_split(I, quarters[g], G, rng, 2, eighths);
        P.eighth[g][0] = eighths[0];
        P.eighth[g][1] = eighths[1];
    }

    return P;
}

// ------------------------------------------------------------
// Cardinality-bucketed enumeration (for the innermost 1/8 groups,
// which are small enough -- n/8 elements -- to brute-force).
// ------------------------------------------------------------

struct CardBuckets {
    std::vector<std::vector<Item>> buckets;
};

static CardBuckets enumerate_bucketed(const Instance& I,
                                      const std::vector<size_t>& g,
                                      int min_k, int max_k) {
    const int n = static_cast<int>(g.size());
    CardBuckets result;
    if (n >= 31) return result;

    result.buckets.resize(n + 1);
    const u32 total = 1u << n;

    for (u32 mask = 0; mask < total; ++mask) {
        u32 pc = popcnt(mask);
        if (static_cast<int>(pc) < min_k || static_cast<int>(pc) > max_k) continue;

        u128 s = 0;
        u32 m = mask;
        while (m) {
            u32 b = static_cast<u32>(__builtin_ctz(m));
            s += I.a[g[b]];
            m &= m - 1;
        }
        result.buckets[pc].push_back({s, mask, static_cast<u8>(pc)});
    }

    for (auto& b : result.buckets)
        std::sort(b.begin(), b.end(), [](const Item& x, const Item& y) { return x.sum < y.sum; });

    return result;
}

// ------------------------------------------------------------
// Heap-based ascending/descending pair generators (same as Stage 1)
// ------------------------------------------------------------

class AscendingPairGen {
    const std::vector<Item>& X;
    const std::vector<Item>& Y;
    struct Node { u128 sum; u32 i, j; bool operator>(const Node& o) const { return sum > o.sum; } };
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;
public:
    AscendingPairGen(const std::vector<Item>& x, const std::vector<Item>& y) : X(x), Y(y) {
        if (!Y.empty())
            for (u32 i = 0; i < X.size(); ++i) heap.push({X[i].sum + Y[0].sum, i, 0});
    }
    bool next(u128& sum, u32& mx, u32& my) {
        if (heap.empty()) return false;
        Node n = heap.top(); heap.pop();
        sum = n.sum; mx = X[n.i].mask; my = Y[n.j].mask;
        if (n.j + 1 < Y.size()) heap.push({X[n.i].sum + Y[n.j + 1].sum, n.i, n.j + 1});
        return true;
    }
};

class DescendingPairGen {
    const std::vector<Item>& X;
    const std::vector<Item>& Y;
    struct Node { u128 sum; u32 i, j; bool operator<(const Node& o) const { return sum < o.sum; } };
    std::priority_queue<Node> heap;
public:
    DescendingPairGen(const std::vector<Item>& x, const std::vector<Item>& y) : X(x), Y(y) {
        if (!Y.empty()) {
            u32 lastJ = static_cast<u32>(Y.size()) - 1;
            for (u32 i = 0; i < X.size(); ++i) heap.push({X[i].sum + Y[lastJ].sum, i, lastJ});
        }
    }
    bool next(u128& sum, u32& mx, u32& my) {
        if (heap.empty()) return false;
        Node n = heap.top(); heap.pop();
        sum = n.sum; mx = X[n.i].mask; my = Y[n.j].mask;
        if (n.j >= 1) heap.push({X[n.i].sum + Y[n.j - 1].sum, n.i, n.j - 1});
        return true;
    }
};

// ------------------------------------------------------------
// Build a cardinality-bucketed representation of an ENTIRE quarter
// by combining its two eighths via the ascending generator, instead
// of brute-force enumerating the whole quarter directly. This is
// the "one level deeper" nesting relative to Stage 1.
//
// Masks returned are relative to the CONCATENATION [eighth0 | eighth1]
// of the quarter's own local index space (0..qsize-1), matching the
// convention verify_masks() expects.
// ------------------------------------------------------------

static CardBuckets build_quarter_via_inner_ss(const Instance& I,
                                              const std::vector<size_t>& eighth0,
                                              const std::vector<size_t>& eighth1,
                                              int quarter_min_k,
                                              int quarter_max_k) {
    const int e0n = static_cast<int>(eighth0.size());
    const int e1n = static_cast<int>(eighth1.size());

    // Enumerate each eighth fully (small: n/8 elements), bucketed by card.
    CardBuckets E0 = enumerate_bucketed(I, eighth0, 0, e0n);
    CardBuckets E1 = enumerate_bucketed(I, eighth1, 0, e1n);

    CardBuckets Q;
    Q.buckets.resize(e0n + e1n + 1);

    for (int c0 = 0; c0 <= e0n && c0 < static_cast<int>(E0.buckets.size()); ++c0) {
        const auto& b0 = E0.buckets[c0];
        if (b0.empty()) continue;
        for (int c1 = 0; c1 <= e1n && c1 < static_cast<int>(E1.buckets.size()); ++c1) {
            int totalc = c0 + c1;
            if (totalc < quarter_min_k || totalc > quarter_max_k) continue;

            const auto& b1 = E1.buckets[c1];
            if (b1.empty()) continue;

            // Stream all combinations ascending; materialize into the
            // quarter-level bucket (still O(|b0|*|b1|) *within this one
            // cardinality pair*, but each pair is much smaller than the
            // full quarter, and we never build a global |quarter|^2
            // cross product like the ORIGINAL file's hash join did).
            AscendingPairGen gen(b0, b1);
            u128 sum; u32 m0, m1;
            while (gen.next(sum, m0, m1)) {
                // Remap eighth-local mask bits into quarter-local bit
                // positions: eighth0 occupies bits [0, e0n), eighth1
                // occupies bits [e0n, e0n+e1n).
                u32 combined = m0 | (m1 << e0n);
                Q.buckets[totalc].push_back({sum, combined, static_cast<u8>(totalc)});
            }
        }
    }

    for (auto& b : Q.buckets)
        std::sort(b.begin(), b.end(), [](const Item& x, const Item& y) { return x.sum < y.sum; });

    return Q;
}

// ------------------------------------------------------------
// Exact candidate reconstruction. Quarter masks are relative to
// [eighth0 | eighth1] concatenation as built above.
// ------------------------------------------------------------

static bool verify_masks(const Instance& I,
                         const Partition& P,
                         u32 m0, u32 m1, u32 m2, u32 m3,
                         Candidate& C) {
    u128 sum = 0;
    u32 card = 0;
    C.bits.clear();
    C.bits.reserve(I.a.size());

    for (int g = 0; g < 4; ++g) {
        u32 mask = (g == 0 ? m0 : g == 1 ? m1 : g == 2 ? m2 : m3);

        const auto& e0 = P.eighth[g][0];
        const auto& e1 = P.eighth[g][1];
        const int e0n = static_cast<int>(e0.size());

        for (size_t j = 0; j < e0.size(); ++j) {
            if ((mask >> j) & 1u) {
                size_t original = e0[j];
                C.bits.push_back(static_cast<u32>(original));
                sum += I.a[original];
                ++card;
            }
        }
        for (size_t j = 0; j < e1.size(); ++j) {
            if ((mask >> (e0n + j)) & 1u) {
                size_t original = e1[j];
                C.bits.push_back(static_cast<u32>(original));
                sum += I.a[original];
                ++card;
            }
        }
    }

    if (sum != I.target || card != I.k) return false;
    C.sum = sum;
    C.found = true;
    return true;
}

// ------------------------------------------------------------
// Worker
// ------------------------------------------------------------

class SuperHybridWorker8 {
    const Instance& I;
    double limit;
    std::atomic<bool>& stop;
    std::mt19937_64 rng;

public:
    SuperHybridWorker8(const Instance& inst, u64 s, double seconds, std::atomic<bool>& st)
        : I(inst), limit(seconds), stop(st), rng(s) {}

    bool run(Candidate& result, const GAResult& guide) {
        const auto begin_time = std::chrono::steady_clock::now();
        auto expired = [&]() {
            if (stop.load(std::memory_order_relaxed)) return true;
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_time).count() >= limit;
        };

        Partition P = make_partition_8(I, guide, rng);

        const int q = static_cast<int>(P.q[0].size());
        const int K = static_cast<int>(I.k);
        const int center = K / 4;
        const int lo = std::max(0, center - 4);
        const int hi = std::min(q, center + 4);

        CardBuckets A = build_quarter_via_inner_ss(I, P.eighth[0][0], P.eighth[0][1], lo, hi);
        if (expired()) return false;
        CardBuckets B = build_quarter_via_inner_ss(I, P.eighth[1][0], P.eighth[1][1], lo, hi);
        if (expired()) return false;
        CardBuckets C = build_quarter_via_inner_ss(I, P.eighth[2][0], P.eighth[2][1], lo, hi);
        if (expired()) return false;
        CardBuckets D = build_quarter_via_inner_ss(I, P.eighth[3][0], P.eighth[3][1], lo, hi);
        if (expired()) return false;

        u64 iter_counter = 0;

        for (int ca = lo; ca <= hi && ca < static_cast<int>(A.buckets.size()); ++ca) {
        for (int cb = lo; cb <= hi && cb < static_cast<int>(B.buckets.size()); ++cb) {
            int rem_ab = ca + cb;
            if (rem_ab > K) continue;
            const auto& Abkt = A.buckets[ca];
            const auto& Bbkt = B.buckets[cb];
            if (Abkt.empty() || Bbkt.empty()) continue;

        for (int cc = lo; cc <= hi && cc < static_cast<int>(C.buckets.size()); ++cc) {
            int cd = K - rem_ab - cc;
            if (cd < 0 || cd < lo || cd > hi || cd >= static_cast<int>(D.buckets.size())) continue;
            const auto& Cbkt = C.buckets[cc];
            const auto& Dbkt = D.buckets[cd];
            if (Cbkt.empty() || Dbkt.empty()) continue;

            AscendingPairGen abGen(Abkt, Bbkt);
            DescendingPairGen cdGen(Cbkt, Dbkt);

            u128 abSum, cdSum;
            u32 abMaskA, abMaskB, cdMaskC, cdMaskD;
            bool hasAB = abGen.next(abSum, abMaskA, abMaskB);
            bool hasCD = cdGen.next(cdSum, cdMaskC, cdMaskD);

            while (hasAB && hasCD) {
                if ((++iter_counter & 0xFFFFu) == 0 && expired()) return false;

                u128 total = abSum + cdSum;
                if (total == I.target) {
                    if (verify_masks(I, P, abMaskA, abMaskB, cdMaskC, cdMaskD, result)) {
                        stop.store(true, std::memory_order_relaxed);
                        return true;
                    }
                    hasAB = abGen.next(abSum, abMaskA, abMaskB);
                } else if (total < I.target) {
                    hasAB = abGen.next(abSum, abMaskA, abMaskB);
                } else {
                    hasCD = cdGen.next(cdSum, cdMaskC, cdMaskD);
                }
            }
        }
        }
        }

        return false;
    }
};

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: superhybrid_stage2.exe <file_or_instance_string> [target] [threads] [timelimit_s]\n";
        return 2;
    }

    std::string source = argv[1];
    std::string target_arg = argc >= 3 ? argv[2] : "";
    int threads = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 4;
    double timelimit = argc >= 5 ? std::max(1.0, std::atof(argv[4])) : 60.0;

    Instance I;
    if (!I.load(source, target_arg)) {
        std::cerr << "ERROR: invalid instance/target.\n";
        return 2;
    }

    const size_t N = I.a.size();

    if (N % 8 != 0) {
        std::cerr << "ERROR: N must be divisible by 8 for the nested 4x2-way partition.\n";
        return 2;
    }
    if (N / 8 >= 31) {
        std::cerr << "ERROR: eighth size exceeds 30-bit mask capacity.\n";
        return 2;
    }

    std::cout
        << "=============================================\n"
        << " SUPERHYBRID SUBSET SUM -- STAGE 2 (EXPERIMENTAL, nested 8-way)\n"
        << " NOTE: not the paper's proven r=7 bound. See header comment.\n"
        << "=============================================\n"
        << "N        : " << N << "\n"
        << "k        : " << I.k << "\n"
        << "Target   : " << u128str(I.target) << "\n"
        << "Threads  : " << threads << "\n"
        << "Time     : " << timelimit << " s\n"
        << "=============================================\n";

    std::atomic<bool> stop(false);
    std::vector<std::thread> workers;
    std::vector<Candidate> answers(threads);
    auto global_start = std::chrono::steady_clock::now();

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid]() {
            u64 seed = 0x9e3779b97f4a7c15ULL ^ (u64(tid + 1) * 0xbf58476d1ce4e5b9ULL);
            double ga_time = std::min(3.0, timelimit * 0.08);

            GAAdvisor advisor(I, seed);
            GAResult guide = advisor.run(ga_time);

            if (stop.load(std::memory_order_relaxed)) return;

            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - global_start).count();
            double remain = std::max(0.1, timelimit - elapsed);

            SuperHybridWorker8 solver(I, seed ^ 0x94d049bb133111ebULL, remain, stop);
            solver.run(answers[tid], guide);
        });
    }

    for (auto& t : workers) t.join();

    Candidate final_answer;
    bool found = false;
    for (const Candidate& c : answers) {
        if (c.found) { final_answer = c; found = true; break; }
    }

    if (!found) {
        std::cout << "\nRESULT: NOT SOLVED\nNo exact k-subset was found within the time limit.\n";
        return 1;
    }

    u128 verify_sum = 0;
    u32 verify_k = 0;
    std::vector<u8> used(N, 0);

    for (u32 idx : final_answer.bits) {
        if (idx >= N || used[idx]) {
            std::cerr << "ERROR: invalid witness.\n";
            return 3;
        }
        used[idx] = 1;
        verify_sum += I.a[idx];
        ++verify_k;
    }

    if (verify_sum != I.target || verify_k != I.k) {
        std::cerr << "ERROR: witness verification failed.\n";
        return 3;
    }

    std::cout << "\nRESULT: SOLVED\nSum    : " << u128str(verify_sum)
               << "\nTarget : " << u128str(I.target)
               << "\nk      : " << verify_k << "\nIndices: ";

    std::sort(final_answer.bits.begin(), final_answer.bits.end());
    for (size_t i = 0; i < final_answer.bits.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << final_answer.bits[i];
    }
    std::cout << "\n";

    return 0;
}