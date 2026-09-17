// superhybrid.cpp
// Single SuperHybrid solver:
// GA guidance -> guided 4-way partition -> modular meet-in-the-middle
// -> exact u128 verification -> exact cardinality verification.
//
// Input is NEVER hard-coded.
// Usage:
//   superhybrid.exe <file_or_instance_string> [target] [threads] [timelimit_s]
//
// Example:
//   superhybrid.exe instance_n96.txt 2885139962834542255881671728668 4 60
//
// Build:
//   g++ -O3 -march=native -mtune=native -std=c++17 -pthread superhybrid.cpp -o superhybrid.exe

#include <algorithm>
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

            // File/string may contain:
            //   a1 a2 ... an target
            // If no target argument is supplied, last number is target.
            target = nums.back();
            nums.pop_back();
            a = std::move(nums);
        }

        if (a.empty()) return false;

        // Benchmark is k=48 for N=96.
        // For reduced instances, preserve the proportional construction:
        // k = N/2.
        k = static_cast<u32>(a.size() / 2);

        return true;
    }
};

struct Item {
    u128 sum;
    u32 mask;
    u8 card;
};

struct QInfo {
    std::vector<u128> val;
    u32 n = 0;
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

static inline u32 mod1(u128 x) {
    // M1 = 2^20 - 1.
    // Repeated folding is cheap and avoids an expensive general modulo.
    u64 lo = static_cast<u64>(x);
    u64 hi = static_cast<u64>(x >> 64);

    u64 r = (lo & M1) + (lo >> 20);
    r += (hi & M1) + (hi >> 20);

    r = (r & M1) + (r >> 20);
    r = (r & M1) + (r >> 20);

    return static_cast<u32>(r & M1);
}

static inline u32 mod2(u128 x) {
    return static_cast<u32>(x % M2);
}

static inline u32 mod_combined(u128 x) {
    return mod1(x) ^ (mod2(x) << 20);
}

static u128 sum_mask(const std::vector<u128>& a,
                     size_t begin,
                     size_t len,
                     u32 mask) {
    u128 s = 0;
    while (mask) {
        u32 b = static_cast<u32>(__builtin_ctz(mask));
        s += a[begin + b];
        mask &= mask - 1;
    }
    return s;
}

// ------------------------------------------------------------
// GA guidance
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

        // Main objective: distance from target.
        // Secondary objective: exact cardinality.
        u128 d = absdiff(p.sum, I.target);
        u128 kd = p.card > I.k ? p.card - I.k : I.k - p.card;

        // Make cardinality meaningful but never stronger than
        // the sum objective by a huge amount.
        p.score = d + kd * u128(1) * (u128(1) << 48);
    }

public:
    explicit GAAdvisor(const Instance& inst, u64 seed)
        : I(inst), rng(seed) {}

    GAResult run(double seconds) {
        const size_t N = I.a.size();

        // Population deliberately modest. GA is a navigator, not the
        // exponential search engine.
        const size_t POP = 96;

        std::vector<Individual> pop(POP), next(POP);

        for (auto& p : pop) {
            p.x.resize(N);

            // Start near exact k.
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

            // Elite.
            next[0] = pop[0];
            next[1] = pop[1];
            next[2] = pop[2];
            next[3] = pop[3];

            for (size_t p = 4; p < POP; ++p) {
                // Tournament selection.
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

                // Uniform crossover.
                for (size_t i = 0; i < N; ++i)
                    child.x[i] =
                        (rng() & 1) ? pa.x[i] : pb.x[i];

                // Mutation.
                double pm = 0.025;

                // Occasional stronger mutation prevents stagnation.
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

            // Frequency is deliberately accumulated from good individuals.
            for (size_t p = 0; p < std::min<size_t>(12, POP); ++p) {
                for (size_t i = 0; i < N; ++i)
                    if (pop[p].x[i])
                        ++hit[i];
            }
        }

        GAResult r;
        r.importance.resize(N);
        r.target_frequency.resize(N);

        // Importance = frequency among elite solutions.
        double denom = double(std::max<u64>(1, generations * 12));

        for (size_t i = 0; i < N; ++i) {
            r.importance[i] = double(hit[i]) / denom;

            // Blend with the best individual's actual membership.
            r.target_frequency[i] =
                0.65 * r.importance[i] +
                0.35 * double(best.x[i]);
        }

        return r;
    }
};

// ------------------------------------------------------------
// Guided partition
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

    // Strong GA ordering, with controlled noise.
    std::sort(idx.begin(), idx.end(),
              [&](size_t x, size_t y) {
                  return G.target_frequency[x] >
                         G.target_frequency[y];
              });

    // Four-way greedy balancing.
    Partition P;

    std::array<u128, 4> sums{0, 0, 0, 0};

    // Small random perturbation so independent workers don't all
    // create exactly the same partition.
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

    // Ensure exact quarter size.
    // This also works for the intended N=96 benchmark.
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
// Quarter subset generation
//
// The critical optimization is to generate only cardinalities
// compatible with k. Each quarter is 24 elements for N=96.
// ------------------------------------------------------------

static void enumerate_quarter(const Instance& I,
                               const std::vector<size_t>& q,
                               int min_k,
                               int max_k,
                               std::vector<Item>& out) {
    const int n = static_cast<int>(q.size());

    if (n >= 31) return; // masks below use u32.

    const u32 total = 1u << n;

    out.clear();

    // Reserve only when useful.
    size_t reserve_est = 0;

    for (int k = min_k; k <= max_k; ++k) {
        if (k < 0 || k > n) continue;

        // Approximate binomial using iterative multiplication.
        u64 c = 1;
        for (int j = 1; j <= k; ++j) {
            c = c * u64(n - j + 1) / u64(j);
            if (c > 20'000'000) break;
        }

        if (c < 20'000'000)
            reserve_est += static_cast<size_t>(c);
    }

    if (reserve_est)
        out.reserve(reserve_est);

    for (u32 mask = 0; mask < total; ++mask) {
        u32 pc = popcnt(mask);
        if (pc < static_cast<u32>(min_k) ||
            pc > static_cast<u32>(max_k))
            continue;

        u128 s = 0;
        u32 m = mask;

        while (m) {
            u32 b = static_cast<u32>(__builtin_ctz(m));
            s += I.a[q[b]];
            m &= m - 1;
        }

        out.push_back({
            s,
            mask,
            static_cast<u8>(pc)
        });
    }
}

// ------------------------------------------------------------
// Exact candidate reconstruction
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
// SuperHybrid worker
//
// Uses two pair layers:
//
//   Q0 + Q1
//   Q2 + Q3
//
// The second side is indexed by two modular residues.
// This is deliberately exact at the final stage.
//
// Instead of sorting by modular value and using a heap
// (invalid because modular order wraps), use hash buckets.
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

        // For exact k=N/2, each quarter's possible cardinalities
        // are 0..qsize, but pair layers only need feasible ranges.
        const int q = static_cast<int>(P.q[0].size());
        const int K = static_cast<int>(I.k);

        std::vector<Item> A, B, C, D;

        // For a four-way balanced partition, the useful cardinalities
        // are centered around K/4. We retain a broad ±4 window.
        //
        // For N=96, K=48 => 8..16.
        const int center = K / 4;
        const int lo = std::max(0, center - 4);
        const int hi = std::min(q, center + 4);

        enumerate_quarter(I, P.q[0], lo, hi, A);
        if (expired()) return false;

        enumerate_quarter(I, P.q[1], lo, hi, B);
        if (expired()) return false;

        enumerate_quarter(I, P.q[2], lo, hi, C);
        if (expired()) return false;

        enumerate_quarter(I, P.q[3], lo, hi, D);
        if (expired()) return false;

        // Map right-side pair:
        //
        // R = C + D
        //
        // We need:
        //   A+B+R = target
        //
        // Store exact pair sums grouped by (M1,M2).
        //
        // To control memory, use the smaller pair side as the
        // indexed side whenever possible.
        struct Key {
            u32 r1;
            u32 r2;

            bool operator==(const Key& o) const {
                return r1 == o.r1 && r2 == o.r2;
            }
        };

        struct KeyHash {
            size_t operator()(const Key& k) const {
                u64 x = (u64(k.r1) << 32) | k.r2;
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebULL;
                x ^= x >> 31;
                return static_cast<size_t>(x);
            }
        };

        struct Pair {
            u32 c;
            u32 d;
            u8 card;
            u128 sum;
        };

        // Build right pair index.
        std::unordered_multimap<Key, Pair, KeyHash> index;

        // Avoid impossible cardinalities immediately.
        const size_t estimated =
            std::min<size_t>(C.size() * 8, 12'000'000);

        index.reserve(estimated);

        for (const Item& c : C) {
            if (expired()) return false;

            for (const Item& d : D) {
                u32 pc = static_cast<u32>(c.card) +
                         static_cast<u32>(d.card);

                if (pc > static_cast<u32>(K))
                    continue;

                u128 s = c.sum + d.sum;

                Key key{mod1(s), mod2(s)};

                index.emplace(
                    key,
                    Pair{
                        c.mask,
                        d.mask,
                        static_cast<u8>(pc),
                        s
                    });
            }
        }

        if (expired()) return false;

        // Search left pair.
        for (const Item& a : A) {
            if (expired()) return false;

            for (const Item& b : B) {
                u32 left_card =
                    static_cast<u32>(a.card) +
                    static_cast<u32>(b.card);

                if (left_card > I.k)
                    continue;

                u128 left = a.sum + b.sum;
                u128 need = I.target - left;

                // Since all elements are non-negative, left <= target
                // is required.
                if (left > I.target)
                    continue;

                Key key{mod1(need), mod2(need)};

                auto range = index.equal_range(key);

                for (auto it = range.first;
                     it != range.second;
                     ++it) {
                    const Pair& p = it->second;

                    if (left_card + p.card != I.k)
                        continue;

                    // Modular collision passed. Now exact u128.
                    if (left + p.sum != I.target)
                        continue;

                    if (verify_masks(
                            I, P,
                            a.mask, b.mask,
                            p.c, p.d,
                            result)) {
                        stop.store(true, std::memory_order_relaxed);
                        return true;
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
            << "Usage: superhybrid.exe "
            << "<file_or_instance_string> "
            << "[target] [threads] [timelimit_s]\n";
        return 2;
    }

    std::string source = argv[1];

    std::string target_arg =
        argc >= 3 ? argv[2] : "";

    int threads =
        argc >= 4 ? std::max(1, std::atoi(argv[3])) : 4;

    double timelimit =
        argc >= 5 ? std::max(1.0, std::atof(argv[4])) : 60.0;

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
        << " SUPERHYBRID SUBSET SUM\n"
        << "=============================================\n"
        << "N        : " << N << "\n"
        << "k        : " << I.k << "\n"
        << "Target   : " << u128str(I.target) << "\n"
        << "Threads  : " << threads << "\n"
        << "Time     : " << timelimit << " s\n"
        << "M1       : " << M1 << "\n"
        << "M2       : " << M2 << "\n"
        << "=============================================\n";

    // --------------------------------------------------------
    // GA guidance.
    //
    // One GA guidance run per worker with independent seeds.
    // No adaptive router: every worker executes the same
    // SuperHybrid pipeline.
    // --------------------------------------------------------

    std::atomic<bool> stop(false);

    std::vector<std::thread> workers;
    std::vector<Candidate> answers(threads);

    auto global_start =
        std::chrono::steady_clock::now();

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back(
            [&, tid]() {
                u64 seed =
                    0x9e3779b97f4a7c15ULL ^
                    (u64(tid + 1) * 0xbf58476d1ce4e5b9ULL);

                // GA consumes a small fraction of the total budget.
                double ga_time =
                    std::min(3.0, timelimit * 0.08);

                GAAdvisor advisor(I, seed);
                GAResult guide = advisor.run(ga_time);

                if (stop.load(std::memory_order_relaxed))
                    return;

                double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        global_start).count();

                double remain =
                    std::max(0.1, timelimit - elapsed);

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

    // --------------------------------------------------------
    // Independent final verification.
    // --------------------------------------------------------

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

    if (verify_sum != I.target ||
        verify_k != I.k) {
        std::cerr << "ERROR: witness verification failed.\n";
        return 3;
    }

    std::cout
        << "\nRESULT: SOLVED\n"
        << "Sum    : " << u128str(verify_sum) << "\n"
        << "Target : " << u128str(I.target) << "\n"
        << "k      : " << verify_k << "\n"
        << "Indices: ";

    std::sort(final_answer.bits.begin(),
              final_answer.bits.end());

    for (size_t i = 0; i < final_answer.bits.size(); ++i) {
        if (i) std::cout << ' ';
        std::cout << final_answer.bits[i];
    }

    std::cout << "\n";

    return 0;
}