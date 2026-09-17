#pragma once
#include "solver_header.h"

// ══════════════════════════════════════════════════════════════════════════
// TieredZeroSumSwapExplorer
//
// Ide: ZeroSumSwapExtractor asli cuma sekali jalan dari 1 solusi awal.
// Di sini kita jadikan BFS bertingkat (tiered/wavefront):
//   Tier 0 = solusi awal
//   Tier 1 = semua solusi baru hasil swap dari Tier 0
//   Tier 2 = semua solusi baru hasil swap dari (setiap solusi) Tier 1
//   ...
//   berhenti saat satu tier tidak menghasilkan solusi baru sama sekali
//     (closure tercapai di bawah relasi "swap <=max_swap_size elemen")
//
// PENTING (batasan, harus jujur disampaikan):
// - Ini BUKAN pembuktian "semua solusi subset-sum ada N buah". Ini hanya
//   menjelajahi solusi yang TERHUBUNG dari solusi awal lewat rantai swap
//   berukuran kecil (default <=4 in + <=4 out per langkah, sama seperti
//   ZeroSumSwapExtractor asli). Solusi lain yang butuh swap lebih besar
//   (>4 elemen sekaligus di satu sisi) tidak akan ketemu.
// - Kompleksitas per node: O(C(|S_in|,<=m) + C(|S_out|,<=m) log(...)) —
//   BISA meledak kalau n besar & m besar. Karena itu ada max_solutions
//   (dari SolverBudget) dan max_tiers sebagai pengaman keras.
// ══════════════════════════════════════════════════════════════════════════
class TieredZeroSumSwapExplorer {
public:
    struct TierInfo {
        int tier;
        size_t new_solutions_found;
        size_t cumulative_total;
    };

    // Jalankan BFS bertingkat. Mengisi stats.all_solutions secara in-place.
    // max_swap_size: ukuran maksimum subset yang ditukar per sisi (default 4, sama seperti extractor asli)
    // max_tiers: pengaman keras jumlah generasi (default besar, praktis tak terbatas)
    static std::vector<TierInfo> explore(const Instance& inst,
                                          ExecutionStats& stats,
                                          const SolverBudget& budget,
                                          int max_swap_size = 4,
                                          int max_tiers = 100000,
                                          bool verbose = true) {
        std::vector<TierInfo> tier_log;
        if (!stats.has_solution || stats.sample_solution.original_indices.empty()) {
            if (verbose) std::cout << "[TieredZeroSumSwap] tidak ada solusi awal, skip.\n";
            return tier_log;
        }

        std::set<std::vector<int>> seen;
        if (stats.all_solutions.empty()) stats.all_solutions.push_back(stats.sample_solution);
        std::deque<SolutionWitness> frontier;
        for (auto& w : stats.all_solutions) {
            std::vector<int> key = w.original_indices;
            std::sort(key.begin(), key.end());
            if (seen.insert(key).second) frontier.push_back(w);
        }

        if (verbose) {
            std::cout << "[TieredZeroSumSwap] mulai dari " << frontier.size()
                      << " solusi awal (Tier 0), max_swap_size=" << max_swap_size << "\n";
        }

        int tier = 0;
        bool capped = false;
        while (!frontier.empty() && tier < max_tiers) {
            if (stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all) {
                capped = true;
                break;
            }
            size_t before = stats.all_solutions.size();
            std::deque<SolutionWitness> next_frontier;

            for (auto& w : frontier) {
                if (stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all) {
                    capped = true;
                    break;
                }
                auto found = swap_from_one(inst, w, seen, max_swap_size, budget, stats);
                for (auto& nw : found) next_frontier.push_back(nw);
            }

            size_t after = stats.all_solutions.size();
            tier++;
            tier_log.push_back({tier, after - before, after});
            if (verbose) {
                std::cout << "  Tier " << tier << ": solusi baru = " << (after - before)
                          << "  (total kumulatif = " << after << ")\n";
            }

            if (after == before) break; // closure: tidak ada solusi baru, berhenti
            frontier = std::move(next_frontier);
            if (capped) break;
        }

        if (verbose) {
            if (capped) {
                std::cout << "[TieredZeroSumSwap] berhenti karena max_solutions budget tercapai ("
                          << budget.max_solutions << "). Belum tentu closure penuh.\n";
            } else if (tier >= max_tiers) {
                std::cout << "[TieredZeroSumSwap] berhenti karena max_tiers tercapai (" << max_tiers << ").\n";
            } else {
                std::cout << "[TieredZeroSumSwap] CLOSURE tercapai: tier terakhir tidak menghasilkan solusi baru.\n";
            }
            std::cout << "[TieredZeroSumSwap] total solusi unik ditemukan = " << stats.all_solutions.size() << "\n";
        }

        stats.solution_count = std::max(stats.solution_count, (u128)stats.all_solutions.size());
        return tier_log;
    }

private:
    // Sama seperti ZeroSumSwapExtractor::extract, tapi bekerja dari SATU witness
    // tertentu (bukan selalu stats.sample_solution), dan MENGEMBALIKAN solusi baru
    // yang ditemukan (untuk dijadikan frontier tier berikutnya) alih-alih cuma
    // menambah ke stats.all_solutions.
    static std::vector<SolutionWitness> swap_from_one(const Instance& inst,
                                                        const SolutionWitness& base,
                                                        std::set<std::vector<int>>& seen,
                                                        int max_swap_size,
                                                        const SolverBudget& budget,
                                                        ExecutionStats& stats) {
        std::vector<SolutionWitness> newly_found;

        std::vector<Element> S_in, S_out;
        std::unordered_set<int> in_set(base.original_indices.begin(), base.original_indices.end());
        for (const auto& e : inst.A) (in_set.count(e.orig_idx) ? S_in : S_out).push_back(e);

        struct Subset { u64 sum; u64 mask; };
        std::vector<Subset> in_subsets;
        int n_in = (int)S_in.size();
        std::function<void(int, u64, u64, int)> gen_in = [&](int idx, u64 csum, u64 cmask, int cnt) {
            if (cnt > 0) in_subsets.push_back({csum, cmask});
            if (cnt == max_swap_size) return;
            for (int j = idx; j < n_in; ++j) gen_in(j + 1, csum + S_in[j].val, cmask | (1ULL << j), cnt + 1);
        };
        gen_in(0, 0, 0, 0);
        std::sort(in_subsets.begin(), in_subsets.end(),
                  [](const Subset& a, const Subset& b) { return a.sum < b.sum; });

        int n_out = (int)S_out.size();
        std::function<void(int, u64, u64, int)> gen_out = [&](int idx, u64 csum, u64 cmask, int cnt) {
            if (stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all) return;
            if (cnt > 0) {
                auto lo = std::lower_bound(in_subsets.begin(), in_subsets.end(), Subset{csum, 0},
                                            [](const Subset& a, const Subset& b) { return a.sum < b.sum; });
                for (auto it = lo; it != in_subsets.end() && it->sum == csum; ++it) {
                    SolutionWitness wit;
                    for (int k = 0; k < n_in; ++k)
                        if (!((it->mask >> k) & 1)) { wit.original_indices.push_back(S_in[k].orig_idx); wit.values.push_back(S_in[k].val); }
                    for (int k = 0; k < n_out; ++k)
                        if ((cmask >> k) & 1) { wit.original_indices.push_back(S_out[k].orig_idx); wit.values.push_back(S_out[k].val); }
                    wit.sort_indices();

                    std::vector<int> key = wit.original_indices; // sudah sorted oleh sort_indices()
                    if (!seen.insert(key).second) continue; // sudah pernah ditemukan di tier manapun

                    u128 s = 0; for (u64 v : wit.values) s += v;
                    wit.sum = s;
                    stats.all_solutions.push_back(wit);
                    newly_found.push_back(wit);

                    if (stats.all_solutions.size() >= budget.max_solutions && !budget.exhaustive_find_all) return;
                }
            }
            if (cnt == max_swap_size) return;
            for (int j = idx; j < n_out; ++j) gen_out(j + 1, csum + S_out[j].val, cmask | (1ULL << j), cnt + 1);
        };
        gen_out(0, 0, 0, 0);

        return newly_found;
    }
};
