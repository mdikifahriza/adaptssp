#include "bcjCore.hpp"
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
    std::cout << "       BCJ-96 (Becker-Coron-Joux Optimized Solver: gamma=0, Stream Hash)        " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Berkas Masukan    : " << file_path << std::endl;
    std::cout << "Target Nilai (T)  : " << tgt_str << std::endl;

    BCJInstance inst;
    if (!BCJInstance::load_from_txt_or_string(file_path, tgt_str, inst)) {
        std::cerr << "[ERROR]: Gagal membaca berkas instans." << std::endl;
        return 1;
    }

    std::cout << "Jumlah Elemen (N) : " << inst.elements.size() << " (k = " << inst.elements.size() / 2 << ")" << std::endl;
    std::cout << "Densitas Instans  : " << std::fixed << std::setprecision(4) << inst.density << std::endl;
    std::cout << "Worker Threads    : " << (num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Memulai pencarian solusi BCJ... Silakan tunggu..." << std::endl;

    BCJExecutionReport report = BCJSolver::solve(inst, num_threads, time_limit_s, true);

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                         LAPORAN EKSEKUSI SOLVER BCJ                            " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Status Eksplorasi : " << (report.solved ? "SOLVED (SATISFIABLE)" : "NOT FOUND / TIMEOUT") << std::endl;
    std::cout << "Total Runtime     : " << std::fixed << std::setprecision(2) << (report.runtime_ms / 1000.0) << " detik" << std::endl;
    std::cout << "Partisi Diuji     : " << report.partitions_evaluated << " partisi acak" << std::endl;
    std::cout << "Total Query Modulo: " << report.queries_evaluated << " queries" << std::endl;
    std::cout << "Peak RAM Memory   : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB" << std::endl;

    if (report.solved) {
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "SAKSI EKSAK DITEMUKAN (" << report.witness_indices.size() << " Elemen):" << std::endl;
        std::cout << "Indeks Elemen : [";
        for (size_t i = 0; i < report.witness_indices.size(); ++i) {
            std::cout << report.witness_indices[i] << (i + 1 < report.witness_indices.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;

        std::cout << "Verifikasi Jumlahan : " << u128_to_string(report.witness_sum) << std::endl;
        std::cout << "Kecocokan Target    : " << (report.witness_sum == inst.target ? "100% COCOK PRESISI" : "GAGAL") << std::endl;
    }
    std::cout << "================================================================================\n" << std::endl;

    return report.solved ? 0 : 1;
}