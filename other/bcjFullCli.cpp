#include "bcjFullCore.hpp"
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::string file_path = "hgj_96bit_instance.txt";
    std::string tgt_str   = "2885139962834542255881671728668";
    unsigned num_threads  = 0;
    double time_limit_s   = 0.0;

    if (argc >= 2) file_path = argv[1];
    if (argc >= 3) tgt_str = argv[2];
    if (argc >= 4) num_threads = (unsigned)std::atoi(argv[3]);
    if (argc >= 5) time_limit_s = std::atof(argv[4]);

    std::cout << "================================================================================" << std::endl;
    std::cout << "   BCJ/BBSS-96 FULL HIERARCHICAL SOLVER (Asymmetric +2 Representation)         " << std::endl;
    std::cout << "   Accelerated with SIMD SSE4.1/4.2, AES-NI & Out-of-Core SSD Engine           " << std::endl;
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
    std::cout << "Fitur Aktif       : 3-Level Depth Tree + BBSS +2 Retention + SIMD POPCNT/CRC32/AES-NI" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Memulai pencarian solusi... Silakan tunggu..." << std::endl;

    FullExecutionReport report = BCJFullSolver::solve(inst, num_threads, time_limit_s, true);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                     LAPORAN EKSEKUSI SOLVER BCJ/BBSS-96                        " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Status Eksplorasi : " << (report.solved ? "SOLVED (SATISFIABLE)" : "NOT FOUND / TIMEOUT") << std::endl;
    std::cout << "Total Runtime     : " << std::fixed << std::setprecision(2) << (report.runtime_ms / 1000.0) << " detik (" 
              << std::setprecision(2) << report.runtime_ms << " ms)" << std::endl;
    std::cout << "Trial Selesai     : " << report.trials_completed << " dekomposisi acak" << std::endl;
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
