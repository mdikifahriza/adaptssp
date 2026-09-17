// superhybrid_stage1_ss.cpp
//
// STAGE 1: SuperHybrid solver, fixed to use TRUE Schroeppel-Shamir merging.
//
// What changed vs. the original superhybrid.cpp:
//   1. Added missing <array> include (the original used std::array without
//      including it -- undefined behavior on strict compilers).
//   2. The old SuperHybridWorker built a FULL unordered_multimap of every
//      (C item x D item) pair before searching -- that is O(|C|*|D|) memory,
//      which can approach O(2^{n/2}) and is exactly the blow-up Schroeppel-
//      Shamir is supposed to avoid.
//   3. Replaced it with the classical trick: bucket each quarter's items by
//      EXACT cardinality, then for every valid (ca,cb,cc,cd) cardinality
//      quadruple summing to k, generate A+B sums in ASCENDING order and
//      C+D sums in DESCENDING order using a min/max-heap "merge of sorted
//      rows" generator (the textbook way to stream the sums of two sorted
//      arrays without materializing the full cross product). A classic
//      two-pointer sweep then finds an exact match.
//      Memory footprint of the merge step becomes O(|bucket_A[ca]| +
//      |bucket_C[cc]|) instead of O(|C|*|D|). Time complexity is unchanged
//      (still ~2^{n/2} in the worst case) -- this stage buys MEMORY, not
//      asymptotic time. That is the actual definition of Schroeppel-Shamir.
//
// This file keeps the GA guidance / partition machinery from the original
// file untouched, since that part was already structurally fine (its value
// is heuristic partition quality, not correctness).
//
// Usage:
//   superhybrid_stage1.exe <file_or_instance_string> [target] [threads] [timelimit_s]
//
// Build:
//   g++ -O3 -march=native -mtune=native -std=c++17 -pthread superhybrid_stage1_ss.cpp -o superhybrid_stage1.exe

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
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i64 = int64_t;

static constexpr u32 M1 = (1u << 20) - 1u; // 2^20-1
static constexpr u32 M2 = 65521u;

static std::string u128str(u128 x) {
    if (x == 0) return "0";
    std::string s;
    while (x) {
        s.push_back(char('0' + x % 10));
        x /= 10;
    }
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
        if (c >= '0' && c <= '9')
            cur.push_back(c);
        else
            flush();
    }
    flush();
    return v;
}

struct Instance {
    std::vector<u128> a;
    u128 target = 0;
    u32 k = 0;

    bool load(const std::string& src,
              const std::string& target_arg = "") {
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
            if (!parse_u128(target_arg, target))
                return false;

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
    u32 mask;
    u8 card;
};

struct Candidate {
    bool found = false;
    std::vector<u32> bits;
    u128 sum = 0;
};

static inline u32 popcnt(u32 x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<u32>(__builtin_popcount(x));
#else
    u32 c = 0;
    while (x) {
        x &= x - 1;
        ++c;
    }
    return c;
#endif
}

// ------------------------------------------------------------
// GA guidance (unchanged from original)
// ------------------------------------------------------------

struct GAResult {
    std::vector<double> importance;
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

    static u128 absdiff(u128 a, u128 b) {
        return a >= b ? a - b : b - a;
    }

    void evaluate(Individual& p) {
        p.sum = 0;
        p.card = 0;

        for (size_t i = 0; i < I.a.size(); ++i) {
            if (p.x[i]) {
                p.sum += I.a[i];
                ++p.card;
            }
        }

        u128 d = absdiff(p.sum, I.target);
        u128 kd = p.card > I.k ? p.card - I.k : I.k - p.card;

        p.score = d + kd * u128(1) * (u128(1) << 48);
    }

public:
    explicit GAAdvisor(const Instance& inst, u64 seed)
        : I(inst), rng(seed) {}

    GAResult run(double seconds) {
        const size_t N = I.a.size();
        const size_t POP = 96;

        std::vector<Individual> pop(POP), next(POP);

        for (auto& p : pop) {
            p.x.resize(N);
            std::fill(p.x.begin(), p.x.end(), 0);

            std::vector<size_t> idx(N);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);

            for (size_t j = 0; j < I.k && j < N; ++j)
                p.x[idx[j]] = 1;

            evaluate(p);
        }

        auto best_it = [&]() {
            return std::min_element(
                pop.begin(), pop.end(),
                [](const Individual& a, const Individual& b) {
                    return a.score < b.score;
                });
        };

        Individual best = *best_it();

        const auto t0 = std::chrono::steady_clock::now();

        std::vector<u64> hit(N, 0);
        u64 generations = 0;

        while (true) {
            double elapsed =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();

            if (elapsed >= seconds)
                break;

            ++generations;

            std::sort(pop.begin(), pop.end(),
                      [](const Individual& a, const Individual& b) {
                          return a.score < b.score;
                      });

            if (pop.front().score < best.score)
                best = pop.front();

            next[0] = pop[0];
            next[1] = pop[1];
            next[2] = pop[2];
            next[3] = pop[3];

            for (size_t p = 4; p < POP; ++p) {
                size_t a = rng() % POP;
                size_t b = rng() % POP;
                const Individual& pa =
                    pop[a].score < pop[b].score ? pop[a] : pop[b];

                a = rng() % POP;
                b = rng() % POP;
                const Individual& pb =
                    pop[a].score < pop[b].score ? pop[a] : pop[b];

                Individual child;
                child.x.resize(N);

                for (size_t i = 0; i < N; ++i)
                    child.x[i] =
                        (rng() & 1) ? pa.x[i] : pb.x[i];

                double pm = 0.025;
                if ((generations % 250) == 0)
                    pm = 0.10;

                for (size_t i = 0; i < N; ++i) {
                    double r = double(rng() % 1000000) / 1000000.0;
                    if (r < pm)
                        child.x[i] ^= 1;
                }

                evaluate(child);
                next[p] = std::move(child);
            }

            pop.swap(next);

            for (size_t p = 0; p < std::min<size_t>(12, POP); ++p) {
                for (size_t i = 0; i < N; ++i)
                    if (pop[p].x[i])
                        ++hit[i];
            }
        }

        GAResult r;
        r.importance.resize(N);
        r.target_frequency.resize(N);

        double denom = double(std::max<u64>(1, generations * 12));

        for (size_t i = 0; i < N; ++i) {
            r.importance[i] = double(hit[i]) / denom;
            r.target_frequency[i] =
                0.65 * r.importance[i] +
                0.35 * double(best.x[i]);
        }

        return r;
    }
};

// ------------------------------------------------------------
// Guided 4-way partition (unchanged from original)
// ------------------------------------------------------------

struct Partition {
    std::vector<size_t> q[4];
};

static Partition make_partition(const Instance& I,
                                const GAResult& G,
                                std::mt19937_64& rng) {
    const size_t N = I.a.size();
    const size_t qsize = N / 4;

    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
              [&](size_t x, size_t y) {
                  return G.target_frequency[x] >
                         G.target_frequency[y];
              });

    Partition P;
    std::array<u128, 4> sums{0, 0, 0, 0};

    std::shuffle(idx.begin(), idx.end(), rng);

    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t x, size_t y) {
                         double ax =
                             G.target_frequency[x] +
                             (double(rng() % 1000) / 1000000.0);
                         double ay =
                             G.target_frequency[y] +
                             (double(rng() % 1000) / 1000000.0);
                         return ax > ay;
                     });

    for (size_t x : idx) {
        int best = -1;

        for (int g = 0; g < 4; ++g) {
            if (P.q[g].size() >= qsize)
                continue;

            if (best < 0 || sums[g] < sums[best])
                best = g;
        }

        if (best < 0)
            best = int(rng() % 4);

        P.q[best].push_back(x);
        sums[best] += I.a[x];
    }

    for (int g = 0; g < 4; ++g) {
        while (P.q[g].size() > qsize) {
            size_t x = P.q[g].back();
            P.q[g].pop_back();

            int dst = -1;
            for (int h = 0; h < 4; ++h) {
                if (P.q[h].size() < qsize) {
                    dst = h;
                    break;
                }
            }

            if (dst < 0)
                break;

            P.q[dst].push_back(x);
        }
    }

    return P;
}

// ------------------------------------------------------------
// Quarter subset generation, bucketed by EXACT cardinality and
// sorted by sum ascending within each bucket. This replaces the
// original flat enumerate_quarter output with a structure that
// the Schroeppel-Shamir generators below can consume directly.
// ------------------------------------------------------------

struct CardBuckets {
    // buckets[c] = items with popcount==c, sorted ascending by sum.
    std::vector<std::vector<Item>> buckets;
    int max_card = 0;
};

static CardBuckets enumerate_quarter_bucketed(const Instance& I,
                                              const std::vector<size_t>& q,
                                              int min_k,
                                              int max_k) {
    const int n = static_cast<int>(q.size());
    CardBuckets result;

    if (n >= 31) return result; // masks below use u32.

    result.max_card = n;
    result.buckets.resize(n + 1);

    const u32 total = 1u << n;

    for (u32 mask = 0; mask < total; ++mask) {
        u32 pc = popcnt(mask);
        if (static_cast<int>(pc) < min_k ||
            static_cast<int>(pc) > max_k)
            continue;

        u128 s = 0;
        u32 m = mask;

        while (m) {
            u32 b = static_cast<u32>(__builtin_ctz(m));
            s += I.a[q[b]];
            m &= m - 1;
        }

        result.buckets[pc].push_back({s, mask, static_cast<u8>(pc)});
    }

    for (auto& b : result.buckets) {
        std::sort(b.begin(), b.end(),
                  [](const Item& x, const Item& y) { return x.sum < y.sum; });
    }

    return result;
}

// ------------------------------------------------------------
// Ascending generator: streams sums of X[i] + Y[j] over a fixed
// pair of buckets (X ascending, Y ascending) in ascending overall
// order, using a min-heap of size |X| (classic "merge |X| sorted
// rows" technique). Never materializes the |X|*|Y| cross product.
// ------------------------------------------------------------

class AscendingPairGen {
    const std::vector<Item>& X;
    const std::vector<Item>& Y;

    struct Node {
        u128 sum;
        u32 i, j;
        bool operator>(const Node& o) const { return sum > o.sum; }
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;

public:
    AscendingPairGen(const std::vector<Item>& x, const std::vector<Item>& y)
        : X(x), Y(y) {
        if (!Y.empty()) {
            for (u32 i = 0; i < X.size(); ++i)
                heap.push({X[i].sum + Y[0].sum, i, 0});
        }
    }

    bool next(u128& sum, u32& maskX, u32& maskY) {
        if (heap.empty()) return false;

        Node n = heap.top();
        heap.pop();

        sum = n.sum;
        maskX = X[n.i].mask;
        maskY = Y[n.j].mask;

        if (n.j + 1 < Y.size())
            heap.push({X[n.i].sum + Y[n.j + 1].sum, n.i, n.j + 1});

        return true;
    }
};

// ------------------------------------------------------------
// Descending generator: same idea, but streams sums in descending
// order (starts from the largest Y element for every row).
// ------------------------------------------------------------

class DescendingPairGen {
    const std::vector<Item>& X;
    const std::vector<Item>& Y;

    struct Node {
        u128 sum;
        u32 i, j;
        bool operator<(const Node& o) const { return sum < o.sum; }
    };

    std::priority_queue<Node> heap; // default: max-heap

public:
    DescendingPairGen(const std::vector<Item>& x, const std::vector<Item>& y)
        : X(x), Y(y) {
        if (!Y.empty()) {
            u32 lastJ = static_cast<u32>(Y.size()) - 1;
            for (u32 i = 0; i < X.size(); ++i)
                heap.push({X[i].sum + Y[lastJ].sum, i, lastJ});
        }
    }

    bool next(u128& sum, u32& maskX, u32& maskY) {
        if (heap.empty()) return false;

        Node n = heap.top();
        heap.pop();

        sum = n.sum;
        maskX = X[n.i].mask;
        maskY = Y[n.j].mask;

        if (n.j >= 1)
            heap.push({X[n.i].sum + Y[n.j - 1].sum, n.i, n.j - 1});

        return true;
    }
};

// ------------------------------------------------------------
// Exact candidate reconstruction (unchanged semantics)
// ------------------------------------------------------------

static bool verify_masks(const Instance& I,
                         const Partition& P,
                         u32 m0, u32 m1,
                         u32 m2, u32 m3,
                         Candidate& C) {
    u128 sum = 0;
    u32 card = 0;

    C.bits.clear();
    C.bits.reserve(I.a.size());

    for (int g = 0; g < 4; ++g) {
        u32 mask = (g == 0 ? m0 :
                    g == 1 ? m1 :
                    g == 2 ? m2 : m3);

        for (size_t j = 0; j < P.q[g].size(); ++j) {
            if ((mask >> j) & 1u) {
                size_t original = P.q[g][j];
                C.bits.push_back(static_cast<u32>(original));
                sum += I.a[original];
                ++card;
            }
        }
    }

    if (sum != I.target || card != I.k)
        return false;

    C.sum = sum;
    C.found = true;
    return true;
}

// ------------------------------------------------------------
// SuperHybrid worker (Stage 1: true Schroeppel-Shamir merge)
// ------------------------------------------------------------

class SuperHybridWorker {
    const Instance& I;
    u64 seed;
    double limit;
    std::atomic<bool>& stop;

    std::mt19937_64 rng;

public:
    SuperHybridWorker(const Instance& inst,
                      u64 s,
                      double seconds,
                      std::atomic<bool>& st)
        : I(inst),
          seed(s),
          limit(seconds),
          stop(st),
          rng(s) {}

    bool run(Candidate& result,
             const GAResult& guide) {
        const auto begin_time =
            std::chrono::steady_clock::now();

        auto expired = [&]() {
            if (stop.load(std::memory_order_relaxed))
                return true;

            double e =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    begin_time).count();

            return e >= limit;
        };

        Partition P = make_partition(I, guide, rng);

        const int q = static_cast<int>(P.q[0].size());
        const int K = static_cast<int>(I.k);

        const int center = K / 4;
        const int lo = std::max(0, center - 4);
        const int hi = std::min(q, center + 4);

        CardBuckets A = enumerate_quarter_bucketed(I, P.q[0], lo, hi);
        if (expired()) return false;

        CardBuckets B = enumerate_quarter_bucketed(I, P.q[1], lo, hi);
        if (expired()) return false;

        CardBuckets C = enumerate_quarter_bucketed(I, P.q[2], lo, hi);
        if (expired()) return false;

        CardBuckets D = enumerate_quarter_bucketed(I, P.q[3], lo, hi);
        if (expired()) return false;

        u64 iter_counter = 0;

        // Iterate every cardinality quadruple (ca,cb,cc,cd) with
        // ca+cb+cc+cd == K. Window is small (<=9 per quarter typically),
        // so this outer loop is cheap (<= ~9^4 combinations).
        for (int ca = lo; ca <= hi && ca < static_cast<int>(A.buckets.size()); ++ca) {
        for (int cb = lo; cb <= hi && cb < static_cast<int>(B.buckets.size()); ++cb) {
            int rem_ab = ca + cb;
            if (rem_ab > K) continue;

            const auto& Abkt = A.buckets[ca];
            const auto& Bbkt = B.buckets[cb];
            if (Abkt.empty() || Bbkt.empty()) continue;

        for (int cc = lo; cc <= hi && cc < static_cast<int>(C.buckets.size()); ++cc) {
            int cd = K - rem_ab - cc;
            if (cd < 0 || cd < lo || cd > hi ||
                cd >= static_cast<int>(D.buckets.size()))
                continue;

            const auto& Cbkt = C.buckets[cc];
            const auto& Dbkt = D.buckets[cd];
            if (Cbkt.empty() || Dbkt.empty()) continue;

            // Ascending stream of A+B sums, descending stream of C+D sums.
            // Memory here is O(|Abkt| + |Cbkt|), not O(|Abkt|*|Bbkt|).
            AscendingPairGen abGen(Abkt, Bbkt);
            DescendingPairGen cdGen(Cbkt, Dbkt);

            u128 abSum, cdSum;
            u32 abMaskA, abMaskB, cdMaskC, cdMaskD;

            bool hasAB = abGen.next(abSum, abMaskA, abMaskB);
            bool hasCD = cdGen.next(cdSum, cdMaskC, cdMaskD);

            while (hasAB && hasCD) {
                if ((++iter_counter & 0xFFFFu) == 0 && expired())
                    return false;

                u128 total = abSum + cdSum;

                if (total == I.target) {
                    if (verify_masks(I, P,
                                      abMaskA, abMaskB,
                                      cdMaskC, cdMaskD,
                                      result)) {
                        stop.store(true, std::memory_order_relaxed);
                        return true;
                    }
                    // Exact sum+cardinality matched by construction, so a
                    // verify_masks failure here should not happen; if it
                    // ever does (defensive), just move on.
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
        std::cerr
            << "Usage: superhybrid_stage1.exe "
            << "<file_or_instance_string> "
            << "[target] [threads] [timelimit_s]\n";
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

    if (N % 4 != 0) {
        std::cerr
            << "ERROR: N must be divisible by 4 for the "
               "four-way SuperHybrid partition.\n";
        return 2;
    }

    if (N / 4 >= 31) {
        std::cerr
            << "ERROR: quarter size exceeds 30-bit mask capacity.\n";
        return 2;
    }

    std::cout
        << "=============================================\n"
        << " SUPERHYBRID SUBSET SUM -- STAGE 1 (true S-S)\n"
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
        workers.emplace_back(
            [&, tid]() {
                u64 seed =
                    0x9e3779b97f4a7c15ULL ^
                    (u64(tid + 1) * 0xbf58476d1ce4e5b9ULL);

                double ga_time = std::min(3.0, timelimit * 0.08);

                GAAdvisor advisor(I, seed);
                GAResult guide = advisor.run(ga_time);

                if (stop.load(std::memory_order_relaxed))
                    return;

                double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        global_start).count();

                double remain = std::max(0.1, timelimit - elapsed);

                SuperHybridWorker solver(
                    I,
                    seed ^ 0x94d049bb133111ebULL,
                    remain,
                    stop);

                solver.run(answers[tid], guide);
            });
    }

    for (auto& t : workers)
        t.join();

    Candidate final_answer;
    bool found = false;

    for (const Candidate& c : answers) {
        if (c.found) {
            final_answer = c;
            found = true;
            break;
        }
    }

    if (!found) {
        std::cout
            << "\nRESULT: NOT SOLVED\n"
            << "No exact k-subset was found within the time limit.\n";
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

    std::cout
        << "\nRESULT: SOLVED\n"
        << "Sum    : " << u128str(verify_sum) << "\n"
        << "Target : " << u128str(I.target) << "\n"
        << "k      : " << verify_k << "\n"
        << "Indices: ";

    std::sort(final_answer.bits.begin(), final_answer.bits.end());

    for (size_t i = 0; i < final_answer.bits.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << final_answer.bits[i];
    }

    std::cout << "\n";

    return 0;
}