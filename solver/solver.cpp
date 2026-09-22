// ============================================================================
// Subset-Sum Solver n=96 - versi window-based (perbaikan atas Gosper exact-w)
// ============================================================================
// Alur (lihat rencana_perbaikan.md untuk detail matematis tiap fase):
//   Fase 0: compute_cardinality_bounds()  + residue_cascade_sieve()
//   Fase 1: iterasi kandidat total-bobot m, dari tengah [k_min,k_max] ke tepi
//   Fase 2: per m, generate kandidat 4-blok via DFS+window (bukan exact-w)
//   Fase 3: join modular 24-bit + trial acak (dipertahankan dari versi awal)
//
// Kejujuran laporan: kalau tidak ketemu dalam budget waktu, program
// melaporkan itu apa adanya (bukan pura-pura UNSAT), plus statistik yang
// sudah dicoba (m mana saja, ukuran list, dsb) supaya bisa dilanjutkan.
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "common.hpp"
#include "bounds.hpp"
#include "residue_sieve.hpp"
#include "weight_window.hpp"
#include "block_gen.hpp"
#include "join_core.hpp"

using Clock = std::chrono::high_resolution_clock;

struct RunConfig {
    double delta_std = 3.0;         // lebar window bobot per blok (dalam std dev)
    uint64_t trials_per_m = 40;      // trial R1 acak per kandidat m
    double time_budget_sec = 90.0;   // budget waktu total (wall clock)
    double memory_budget_mb = 1500;  // budget memori kasar untuk list per blok
};

int main(int argc, char** argv) {
    RunConfig cfg;
    if (argc > 1) cfg.time_budget_sec = std::atof(argv[1]);
    if (argc > 2) cfg.delta_std = std::atof(argv[2]);
    if (argc > 3) cfg.trials_per_m = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) cfg.memory_budget_mb = std::atof(argv[4]);

    std::cout << "========================================================\n";
    std::cout << "  SUBSET-SUM SOLVER N=96 - WINDOW-BASED (v2)\n";
    std::cout << "========================================================\n";
    std::cout << "Config: time_budget=" << cfg.time_budget_sec << "s, delta_std="
               << cfg.delta_std << ", trials_per_m=" << cfg.trials_per_m << "\n\n";

    std::string target_str = "2885139962834542255881671728668";
    std::vector<std::string> arr_str = {
        "64300852301747213155390518686", "54284071953024346822603392087", "75972072006955199644957375388",
        "61429940148315267567327568609", "73447852662580382709217655496", "57138472814643822457035728886",
        "50277836707594685260584418954", "51966778669230426263872003781", "53490051955568236002214393381",
        "70637041326140132469873756286", "72436475493310700745923195362", "68749831331379668726267786875",
        "57493108649170101187692178652", "42760627126952266723314088760", "44155495572681791380402298601",
        "50557155706096273314307736331", "67771387061995455118517407283", "44581037295885641544853820618",
        "66283150608800342930677255886", "62332122001331169409828188390", "71762004913917867011814667437",
        "74089452958401677769736764725", "68607477734383781386014534812", "61267905273898190023363659151",
        "75252690818474212210670859575", "40375880707315106315608550071", "52101527181847894779504350171",
        "61204423988863088447554182294", "55730790569637857594796117253", "50121554566572490550479513136",
        "43849757770526769460839711155", "51932916437146171990401806693", "44951954648915374208837737673",
        "49776282147967337018033118955", "69123706783352025532082568593", "41100552458679260452945606976",
        "77079836838475308932436747269", "74500458557096698339112390856", "70179047641705511344281489769",
        "61420996775495503780679704278", "51610687726256673105677579619", "70933247406654390990849585589",
        "40054475759081389303766883379", "42754457689742711089426425028", "43461396565999151222636900800",
        "70908949843512308147783689832", "72976174710205074432477777159", "70541898689231146536196269226",
        "68051569751209585060573425427", "75571234877950598583477407783", "69842187069251064217907036099",
        "51180135740413322745518315782", "72262334307017230676621084626", "53270213957313114417084865443",
        "65287494637406632430811473367", "74709097272909873825057380587", "70839387940423076091700931649",
        "41262367615528476710096442130", "42982724580152043769673704222", "54679228711960147972445867122",
        "77706431526679443867606823230", "43857375798250060701126473433", "45510923760002202105891244023",
        "61738922539271314120758163852", "50520987337438964248490145888", "57654902939154508616413806291",
        "68247602235921381800707844894", "51967128167601352347006053548", "78535741087417493034160349231",
        "56494490380214253149696452092", "75527976009283763117756090790", "70859912785524943796232319579",
        "41972293160046301948708897475", "50697386729690436189073400325", "45970081308022002261498368625",
        "69887052127878303114989244983", "70514009899104370734451609272", "70410373470740023926676827190",
        "70731117183519223783617075568", "72916295235268575729018504824", "53413648494596476286886250539",
        "60466522226364868219294441613", "56762970883461286382539975156", "70095522488810605402083170232",
        "60142784872297152954606705439", "62013095473094127930855669231", "78477204386652359512885151306",
        "62471603407529555157529599745", "43286304567582015692418630452", "55078462722065238848108747677",
        "55923800463075106968571878471", "75250466254473771558876739839", "77238742130883993532880088493",
        "58553965811586557587440961606", "65762984111912352450708579864", "45470226602864138443092881109"
    };

    int128 T = parse_int128(target_str);
    std::vector<int128> a;
    for (const auto& s : arr_str) a.push_back(parse_int128(s));
    int n = (int)a.size();

    auto t_start = Clock::now();
    auto elapsed_sec = [&]() {
        return std::chrono::duration<double>(Clock::now() - t_start).count();
    };

    // ---------------- FASE 0.1: bounds ----------------
    std::vector<int128> a_desc = a;
    std::sort(a_desc.begin(), a_desc.end(), std::greater<int128>());
    CardinalityBounds cb = compute_cardinality_bounds(a_desc, T);

    std::cout << "[FASE 0.1] Cardinality bounds (terbukti, bukan estimasi):\n";
    if (!cb.exact_hit_possible) {
        std::cout << "  Target di luar rentang total sum yang mungkin -> UNSAT terbukti.\n";
        return 0;
    }
    std::cout << "  k_min = " << cb.k_min << ", k_max = " << cb.k_max
               << "  (" << cb.feasible_k_count << " nilai m mungkin)\n\n";

    if (cb.feasible_k_count <= 0) {
        std::cout << "  Tidak ada k feasible -> UNSAT terbukti.\n";
        return 0;
    }

    // ---------------- FASE 0.2: residue sieve ----------------
    std::vector<int> primes = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
    SieveResult sieve = residue_cascade_sieve(a, T, cb.k_min, cb.k_max, primes);
    std::cout << "[FASE 0.2] ResidueCascadeSieve (prima " ;
    for (size_t i = 0; i < primes.size(); ++i) std::cout << primes[i] << (i + 1 < primes.size() ? "," : "");
    std::cout << "):\n";
    std::cout << "  Elemen tereliminasi (provably tak terpakai): " << sieve.eliminated_count
               << " / " << n << "\n";
    if (sieve.proved_unsat) {
        std::cout << "  Sisa elemen < k_min -> UNSAT terbukti.\n";
        return 0;
    }

    // Bangun daftar elemen yang bertahan + pemetaan ke index asli.
    std::vector<int128> a_reduced;
    std::vector<int> orig_index;
    for (int i = 0; i < n; ++i) {
        if (sieve.keep[i]) { a_reduced.push_back(a[i]); orig_index.push_back(i); }
    }
    int nr = (int)a_reduced.size();
    std::cout << "  Ukuran instance setelah sieve: n' = " << nr << "\n\n";

    // Split jadi 4 blok seimbang (ukuran boleh tidak sama persis kalau nr%4 != 0).
    int base = nr / 4, rem = nr % 4;
    std::vector<int> block_sizes(4, base);
    for (int i = 0; i < rem; ++i) block_sizes[i]++;
    std::vector<std::vector<int128>> blocks(4);
    std::vector<std::vector<int>> block_orig_idx(4); // orig index per elemen dlm blok
    {
        int pos = 0;
        for (int b = 0; b < 4; ++b) {
            for (int i = 0; i < block_sizes[b]; ++i) {
                blocks[b].push_back(a_reduced[pos]);
                block_orig_idx[b].push_back(orig_index[pos]);
                pos++;
            }
        }
    }
    std::cout << "[FASE 2 setup] Ukuran blok: " << block_sizes[0] << "," << block_sizes[1]
               << "," << block_sizes[2] << "," << block_sizes[3] << "\n\n";

    // ---------------- FASE 1: iterasi m dari tengah ke tepi ----------------
    std::vector<int> m_order;
    {
        int lo = cb.k_min, hi = cb.k_max;
        int mid = (lo + hi) / 2;
        m_order.push_back(mid);
        int d = 1;
        while ((int)m_order.size() < cb.feasible_k_count) {
            if (mid - d >= lo) m_order.push_back(mid - d);
            if ((int)m_order.size() >= cb.feasible_k_count) break;
            if (mid + d <= hi) m_order.push_back(mid + d);
            d++;
        }
    }

    bool found = false;
    uint128 sol_mask_reduced = 0; // mask atas index REDUCED (posisi di a_reduced/blocks)
    int found_m = -1;
    uint64_t rng_seed = 1337;

    for (int m : m_order) {
        if (elapsed_sec() > cfg.time_budget_sec) {
            std::cout << "[!] Budget waktu habis sebelum mencoba semua m. Berhenti di m=" << m << " (belum dicoba).\n";
            break;
        }
        std::cout << "[FASE 1] Mencoba m = " << m << "  (t=" << std::fixed << std::setprecision(1)
                   << elapsed_sec() << "s)\n";

        // FASE 1: window per blok
        std::vector<WeightWindow> win(4);
        double total_est = 0.0;
        bool window_ok = true;
        for (int b = 0; b < 4; ++b) {
            win[b] = window_for_block(m, nr, block_sizes[b], cfg.delta_std);
            double est = 0.0;
            for (int w = win[b].w_lo; w <= win[b].w_hi; ++w) est += n_choose_k_d(block_sizes[b], w);
            total_est += est;
            std::cout << "    blok " << b << ": window=[" << win[b].w_lo << "," << win[b].w_hi
                       << "]  ~" << (uint64_t)est << " kandidat\n";
        }
        // Estimasi hemat-memori: blok 0,2 -> ModElement (48B), blok 1,3 -> Element (32B).
        // (2 blok * 48B + 2 blok * 32B) / 4 blok = rata2 40B/kandidat, plus overhead
        // sementara saat merge (L12/L34) diperhitungkan longgar via faktor 1.5x.
        double est_mb = total_est * 40.0 * 1.5 / 1e6;
        std::cout << "    Estimasi memori kasar: ~" << std::setprecision(1) << est_mb << " MB\n";
        if (est_mb > cfg.memory_budget_mb) {
            std::cout << "    [skip] Melebihi budget memori (" << cfg.memory_budget_mb
                       << " MB) - perkecil delta_std atau naikkan budget.\n";
            continue;
        }

        // FASE 2: generate kandidat per blok
        std::vector<std::vector<BlockCandidate>> block_cands(4);
        for (int b = 0; b < 4; ++b) {
            gen_block_candidates(blocks[b], win[b].w_lo, win[b].w_hi, block_cands[b]);
        }
        std::cout << "    Ukuran list aktual: " << block_cands[0].size() << "," << block_cands[1].size()
                   << "," << block_cands[2].size() << "," << block_cands[3].size() << "\n";

        // Konversi ke representasi HEMAT MEMORI:
        //  - blok 0 (L1) dan blok 2 (L3) -> langsung ke ModElement terurut,
        //    TANPA menyimpan salinan Element mentah (hemat ~40% memori utk
        //    kedua blok ini, karena tidak ada duplikasi Element+ModElement).
        //  - blok 1 (L2) dan blok 3 (L4) -> Element biasa (dipakai apa
        //    adanya saat iterasi di join_core).
        int block_start[4];
        block_start[0] = 0;
        for (int b = 1; b < 4; ++b) block_start[b] = block_start[b - 1] + block_sizes[b - 1];

        // Pilih lebar modulus adaptif (bukan fixed 24-bit): mendekati
        // log2(ukuran list terbesar) supaya ukuran hasil merge (L1+L2 atau
        // L3+L4) proporsional terhadap ukuran list asli (prinsip HGJ:
        // |merge| ~ |L1|*|L2| / 2^mod_bits).
        size_t max_list_sz = std::max({block_cands[0].size(), block_cands[1].size(),
                                        block_cands[2].size(), block_cands[3].size()});
        uint64_t mod_bits = 20;
        while ((1ULL << mod_bits) < max_list_sz && mod_bits < 40) mod_bits++;

        auto to_mod_sorted = [&](int b) {
            std::vector<ModElement> out(block_cands[b].size());
            uint64_t MOD_MASK = (1ULL << mod_bits) - 1;
            for (size_t i = 0; i < block_cands[b].size(); ++i) {
                int128 s = block_cands[b][i].sum;
                uint128 gmask = (uint128)block_cands[b][i].local_mask << block_start[b];
                out[i] = {(uint32_t)((uint64_t)(s & (int128)MOD_MASK)), s, gmask};
            }
            std::sort(out.begin(), out.end());
            return out;
        };
        auto to_element = [&](int b) {
            std::vector<Element> out(block_cands[b].size());
            for (size_t i = 0; i < block_cands[b].size(); ++i) {
                uint128 gmask = (uint128)block_cands[b][i].local_mask << block_start[b];
                out[i] = {block_cands[b][i].sum, gmask};
            }
            return out;
        };

        std::vector<ModElement> L1_mod = to_mod_sorted(0);
        std::vector<Element> L2 = to_element(1);
        std::vector<ModElement> L3_mod = to_mod_sorted(2);
        std::vector<Element> L4 = to_element(3);
        // Bebaskan block_cands sekarang karena sudah dikonversi.
        for (auto& v : block_cands) { std::vector<BlockCandidate>().swap(v); }

        std::cout << "    mod_bits=" << mod_bits << "\n";

        // FASE 3: join
        JoinResult jr = join_core_search(L1_mod, L2, L3_mod, L4, T, cfg.trials_per_m, rng_seed++, mod_bits);
        if (jr.found) {
            found = true;
            sol_mask_reduced = jr.mask;
            found_m = m;
            std::cout << "  [!] SOLUSI DITEMUKAN pada m=" << m << ", trial=" << jr.trial_found_at << "!\n\n";
            break;
        } else {
            std::cout << "    Tidak ditemukan pada " << cfg.trials_per_m << " trial untuk m=" << m << ".\n\n";
        }
    }

    auto t_end = Clock::now();
    std::chrono::duration<double> diff = t_end - t_start;

    std::cout << "========================================================\n";
    if (found) {
        // Petakan mask reduced -> index asli 0..95
        std::vector<int> solution_indices;
        for (int i = 0; i < nr; ++i) {
            if ((sol_mask_reduced >> i) & 1) solution_indices.push_back(orig_index[i]);
        }
        std::sort(solution_indices.begin(), solution_indices.end());

        std::cout << " [SUCCESS] Solusi ditemukan (m=" << found_m << ") dalam "
                   << diff.count() << " detik!\n";
        std::cout << " Indeks Subset Solusi (0-based, terhadap array asli 96 elemen): [";
        for (size_t i = 0; i < solution_indices.size(); ++i)
            std::cout << solution_indices[i] << (i + 1 < solution_indices.size() ? ", " : "");
        std::cout << "]\n";
        std::cout << " |subset| = " << solution_indices.size() << "\n";

        int128 check_sum = 0;
        for (int idx : solution_indices) check_sum += a[idx];
        std::cout << " Total Penjumlahan : " << int128_to_string(check_sum) << "\n";
        std::cout << " Target Sebenarnya : " << int128_to_string(T) << "\n";
        std::cout << " Match             : " << (check_sum == T ? "YA (terverifikasi)" : "TIDAK - BUG!") << "\n";
    } else {
        std::cout << " [BELUM KETEMU] Tidak ditemukan solusi dalam budget waktu "
                   << cfg.time_budget_sec << "s (" << diff.count() << " detik berlalu).\n";
        std::cout << " Ini BUKAN bukti UNSAT -- ruang m/window/trial belum habis dieksplorasi.\n";
        std::cout << " Saran: naikkan time_budget, delta_std, atau trials_per_m,\n";
        std::cout << " atau jalankan paralel per-m di mesin multi-core.\n";
    }
    std::cout << "========================================================\n";

    return 0;
}
