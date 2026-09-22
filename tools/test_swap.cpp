// Tes diferensial zero_sum_swap.h vs oracle brute force.
// Oracle: enumerasi SEMUA subset (n<=18), ambil yang jumlahnya == target, lalu BFS pada graf
// "T bertetangga dengan S bila |S\T|<=m dan |T\S|<=m" mulai dari solusi awal. Himpunan solusi dan
// ukuran tiap lapis BFS harus sama persis dengan hasil explorer (max_solutions=0, tanpa cap).
#include "zero_sum_swap.h"   // butuh -I<folder sumber>, mis. g++ -I. tools/test_swap.cpp
#include <cstdio>
#include <map>
#include <random>
#include <unordered_map>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::vector<size_t> mask_to_idx(uint32_t m, size_t n) {
    std::vector<size_t> v; for (size_t i = 0; i < n; ++i) if (m >> i & 1) v.push_back(i); return v;
}

struct Oracle { std::vector<size_t> layer_sizes; std::set<std::vector<size_t>> all; };

static Oracle oracle(const std::vector<u128> &vals, u128 target, uint32_t init, int m) {
    const size_t n = vals.size();
    std::vector<uint32_t> sols;
    for (uint32_t mask = 0; mask < (1u << n); ++mask) {
        u128 s = 0; for (size_t i = 0; i < n; ++i) if (mask >> i & 1) s += vals[i];
        if (s == target) sols.push_back(mask);
    }
    std::unordered_map<uint32_t, int> dist; dist[init] = 0;
    std::vector<uint32_t> cur{init};
    Oracle O; O.layer_sizes.push_back(1);
    while (!cur.empty()) {
        std::vector<uint32_t> nxt;
        for (uint32_t S : cur) for (uint32_t T : sols) {
            if (dist.count(T)) continue;
            if (__builtin_popcount(S & ~T) <= m && __builtin_popcount(T & ~S) <= m) { dist[T] = 1; nxt.push_back(T); }
        }
        // dist diisi setelah lapis selesai supaya lapis yang sama tidak saling memanjangkan
        if (!nxt.empty()) O.layer_sizes.push_back(nxt.size());
        cur = nxt;
    }
    for (auto &kv : dist) O.all.insert(mask_to_idx(kv.first, n));
    return O;
}

int main() {
    std::mt19937_64 rng(12345);
    int cases = 0, nontrivial = 0; size_t maxsol = 0;
    for (int it = 0; it < 1500; ++it) {
        const size_t n = 4 + rng() % 13; // 4..16
        const int mode = static_cast<int>(rng() % 4);
        std::vector<u128> vals(n);
        for (auto &v : vals) {
            if (mode == 0) v = rng() % 6;                       // banyak duplikat + nol
            else if (mode == 1) v = 1 + rng() % 12;              // banyak solusi
            else if (mode == 2) { static const u128 pool[5] = {((u128)1 << 120) + 7, ((u128)1 << 119) + 3, ((u128)1 << 118), ((u128)1 << 120) + 7 + ((u128)1 << 119) + 3, ((u128)1 << 117) + 11}; v = pool[rng() % 5]; } // >2^64, total > 2^64
            else v = (u128)(rng() % 40) << 70;                   // besar, kelipatan
        }
        // pilih solusi awal acak, target = jumlahnya
        uint32_t init = static_cast<uint32_t>(rng() % (1u << n));
        u128 target = 0; for (size_t i = 0; i < n; ++i) if (init >> i & 1) target += vals[i];
        const int m = 1 + static_cast<int>(rng() % 6);
        SwapExploreParams P; P.max_swap_size = m; P.max_solutions = 0; P.max_subsets_per_side = 50000000; P.verbose = false;
        auto R = explore_zero_sum_swaps(vals, target, mask_to_idx(init, n), P);
        CHECK(R.error.empty(), "error: %s", R.error.c_str());
        CHECK(!R.capped && R.stop_reason == "closure", "harus closure, dapat '%s'", R.stop_reason.c_str());
        CHECK(R.verify_failures == 0, "verify_failures=%zu", R.verify_failures);
        Oracle O = oracle(vals, target, init, m);
        std::set<std::vector<size_t>> got; for (auto &w : R.all_solutions) { CHECK(got.insert(w.indices).second, "duplikat"); CHECK(w.sum == target, "sum!=target"); }
        CHECK(got == O.all, "himpunan beda: got=%zu oracle=%zu (n=%zu m=%d mode=%d)", got.size(), O.all.size(), n, m, mode);
        // tier: explorer mencatat satu tier terakhir yang 0 baru; bandingkan semua tier bukan-nol
        std::vector<size_t> tsz{1}; for (auto &t : R.tiers) if (t.new_solutions > 0) tsz.push_back(t.new_solutions);
        CHECK(tsz == O.layer_sizes, "ukuran tier beda (n=%zu m=%d mode=%d)", n, m, mode);
        CHECK(R.all_solutions[0].indices == mask_to_idx(init, n), "[0] bukan solusi awal");
        ++cases; if (got.size() > 1) ++nontrivial; maxsol = std::max(maxsol, got.size());
    }
    std::printf("diferensial: %d kasus, %d dgn >1 solusi, solusi terbanyak=%zu, fails=%d\n", cases, nontrivial, maxsol, fails);

    // ---- error handling ----
    std::vector<u128> v{5, 3, 2, 7};
    { SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps(v, 8, {0, 1}, P); CHECK(R.error.empty(), "valid ditolak"); }
    { SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps(v, 9, {0, 1}, P); CHECK(!R.error.empty(), "sum salah tidak ditolak"); CHECK(R.all_solutions.empty(), "seharusnya kosong saat error"); }
    { SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps(v, 10, {0, 0, 1}, P); CHECK(!R.error.empty(), "indeks ganda tidak ditolak"); }
    { SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps(v, 8, {0, 9}, P); CHECK(!R.error.empty(), "indeks di luar rentang tidak ditolak"); }
    { SwapExploreParams P; P.verbose = false; P.max_swap_size = 0; auto R = explore_zero_sum_swaps(v, 8, {0, 1}, P); CHECK(!R.error.empty(), "m=0 tidak ditolak"); }
    { SwapExploreParams P; P.verbose = false; P.max_swap_size = kMaxSwapSize + 1; auto R = explore_zero_sum_swaps(v, 8, {0, 1}, P); CHECK(!R.error.empty(), "m>max tidak ditolak"); }
    { std::vector<u128> big{~(u128)0, 1}; SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps(big, 1, {1}, P); CHECK(!R.error.empty(), "overflow total tidak ditolak"); }
    { SwapExploreParams P; P.verbose = false; auto R = explore_zero_sum_swaps({}, 0, {}, P); CHECK(R.error.empty() && R.all_solutions.size() == 1 && !R.capped, "n=0 target=0"); }
    // ---- cap ----
    { std::vector<u128> ones(14, 1); SwapExploreParams P; P.verbose = false; P.max_solutions = 10; P.max_swap_size = 3;
      auto R = explore_zero_sum_swaps(ones, 5, {0, 1, 2, 3, 4}, P);
      CHECK(R.capped && R.stop_reason == "max_solutions tercapai" && R.all_solutions.size() == 10, "cap solusi: capped=%d size=%zu '%s'", R.capped, R.all_solutions.size(), R.stop_reason.c_str()); }
    { std::vector<u128> ones(14, 1); SwapExploreParams P; P.verbose = false; P.max_solutions = 1;
      auto R = explore_zero_sum_swaps(ones, 5, {0, 1, 2, 3, 4}, P); CHECK(R.capped && R.all_solutions.size() == 1 && R.tiers_run == 0, "cap=1 langsung"); }
    { std::vector<u128> ones(14, 1); SwapExploreParams P; P.verbose = false; P.max_solutions = 0; P.max_tiers = 1; P.max_swap_size = 1;
      auto R = explore_zero_sum_swaps(ones, 5, {0, 1, 2, 3, 4}, P); CHECK(R.capped && R.stop_reason == "max_tiers tercapai", "max_tiers: '%s'", R.stop_reason.c_str()); }
    { std::vector<u128> ones(14, 1); SwapExploreParams P; P.verbose = false; P.max_solutions = 0; P.max_subsets_per_side = 10; P.max_swap_size = 3;
      auto R = explore_zero_sum_swaps(ones, 5, {0, 1, 2, 3, 4}, P); CHECK(R.capped && R.skipped_nodes > 0 && R.all_solutions.size() == 1, "skip node: skipped=%zu size=%zu", R.skipped_nodes, R.all_solutions.size()); }
    std::printf("total fails=%d\n", fails);
    return fails ? 1 : 0;
}
