#include "zweydingerCore.hpp"
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::string file_path = "hgj_96bit_instance.txt";
    std::string tgt_str   = "2885139962834542255881671728668";
    unsigned num_threads  = 4;
    double time_limit_s   = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) num_threads = (unsigned)std::atoi(argv[++i]);
        } else if (arg == "-l" || arg == "--time") {
            if (i + 1 < argc) time_limit_s = std::atof(argv[++i]);
        } else if (arg == "--target") {
            if (i + 1 < argc) tgt_str = argv[++i];
        } else if (arg == "--file") {
            if (i + 1 < argc) file_path = argv[++i];
        } else if (i == 1) {
            file_path = arg;
        } else if (i == 2) {
            if (arg != "-" && arg != "default" && arg != "auto") tgt_str = arg;
        } else if (i == 3) {
            num_threads = (unsigned)std::atoi(arg.c_str());
        } else if (i == 4) {
            time_limit_s = std::atof(arg.c_str());
        }
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "     ZWEYDINGER-ESSER 96-BIT SUBSET SUM SOLVER (Base-List Reuse Engine)         " << std::endl;
    std::cout << "     Inspired by FloydZ/decoding & cryptanalysislib (Eurocrypt 2022/1329)      " << std::endl;
    std::cout << "     Accelerated with SSE4.1, AES-NI PRNG, POPCNT & FlatBucketHashMap           " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Berkas Masukan    : " << file_path << std::endl;
    std::cout << "Target Nilai (T)  : " << (tgt_str.empty() ? "(Otomatis dari berkas)" : tgt_str) << std::endl;

    FullInstance inst;
    if (!FullInstance::load(file_path, tgt_str, inst)) {
        std::cerr << "[ERROR]: Gagal membaca berkas instans: " << file_path << std::endl;
        return 1;
    }

    std::cout << "Jumlah Elemen (N) : " << inst.n << " (Target K = " << inst.k << ")" << std::endl;
    std::cout << "Feasible K Window : [" << inst.k_min << " .. " << inst.k_max << "] (Preprosesing K-Bounds)" << std::endl;
    std::cout << "Densitas Instans  : " << std::fixed << std::setprecision(4) << inst.density << std::endl;
    std::cout << "Alokasi Worker    : " << (num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) << " worker threads" << std::endl;
    std::cout << "Batas Waktu       : " << (time_limit_s <= 0.0 ? "Unlimited" : (std::to_string((long long)time_limit_s) + " detik")) << std::endl;
    std::cout << "Fitur Utama       : Base-List Reuse (Zero Regeneration per Trial) + FlatBucketHashMap" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Memulai pencarian solusi... Silakan tunggu..." << std::endl;

    ZweydingerReport report = ZweydingerSspSolver::solve(inst, num_threads, time_limit_s, true);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                 LAPORAN EKSEKUSI SOLVER ZWEYDINGER-ESSER-96                   " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Status Eksplorasi : " << (report.solved ? "SOLVED (SATISFIABLE)" : "NOT FOUND / TIMEOUT") << std::endl;
    std::cout << "Total Runtime     : " << std::fixed << std::setprecision(2) << (report.runtime_ms / 1000.0) << " detik (" 
              << std::setprecision(2) << report.runtime_ms << " ms)" << std::endl;
    std::cout << "Trial Selesai     : " << report.trials_completed << " dekomposisi perantara" << std::endl;
    std::cout << "Epoch Permutasi   : " << report.epochs_completed << " permutasi baru (Full Refresh MitM)" << std::endl;
    double trials_per_sec = (report.runtime_ms > 0) ? (report.trials_completed / (report.runtime_ms / 1000.0)) : 0;
    std::cout << "Throughput Speed  : " << std::fixed << std::setprecision(1) << trials_per_sec << " trial/detik" << std::endl;
    std::cout << "Kandidat Level 1  : " << report.candidates_l1 << " kandidat terpasangkan" << std::endl;
    std::cout << "Kandidat Level 2  : " << report.candidates_l2 << " kandidat terpasangkan" << std::endl;
    std::cout << "Peak RAM Memory   : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB" << std::endl;
    std::cout << "Verifikasi Hasil  : " << report.verification_status << std::endl;

    if (report.solved) {
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "SAKSI EKSAK DITEMUKAN (" << report.witness_indices.size() << " Elemen):" << std::endl;
        std::cout << "Indeks Elemen : [";
        for (size_t i = 0; i < report.witness_indices.size(); ++i) {
            std::cout << report.witness_indices[i] << (i + 1 < report.witness_indices.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;

        std::cout << "Verifikasi Jumlahan : " << u128_to_dec(report.witness_sum) << std::endl;
        std::cout << "Kecocokan Target    : " << (report.witness_sum == inst.target ? "100% COCOK PRESISI" : "GAGAL") << std::endl;
    }
    std::cout << "================================================================================\n" << std::endl;

    return report.solved ? 0 : 1;
}
