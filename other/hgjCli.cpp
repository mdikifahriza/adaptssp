#include "hgjCore.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void print_help(const char* prog) {
    std::cout << "========================================================================================================\n";
    std::cout << "          HGJ-96 (Howgrave-Graham & Joux + Tail-Table Pruning + Multi-Core + Early Abort)               \n";
    std::cout << "========================================================================================================\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << prog << " [input.txt] [target] [threads] [time_limit_s]\n\n";
    std::cout << "ARGUMENTS:\n";
    std::cout << "  [input.txt]      : Berkas teks instans 96-bit (Default: hgj_96bit_instance.txt).\n";
    std::cout << "                     Komentar diawali '#', '//', atau ';' diabaikan otomatis.\n";
    std::cout << "  [target]         : Nilai target T (Default: 2885139962834542255881671728668).\n";
    std::cout << "  [threads]        : Jumlah worker threads (Default: 0 = otomatis deteksi core CPU).\n";
    std::cout << "  [time_limit_s]   : Batas waktu dalam detik (Default: 0 = unlimited sampai ketemu).\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  1. Jalankan benchmark HGJ 96-bit dengan seluruh core CPU:\n";
    std::cout << "     " << prog << " hgj_96bit_instance.txt 2885139962834542255881671728668\n\n";
    std::cout << "  2. Jalankan dengan 8 threads dan batas waktu 1800 detik (30 menit):\n";
    std::cout << "     " << prog << " hgj_96bit_instance.txt 2885139962834542255881671728668 8 1800\n";
    std::cout << "========================================================================================================\n";
}

int main(int argc, char* argv[]) {
    std::string file_path = "hgj_96bit_instance.txt";
    std::string tgt_str   = "2885139962834542255881671728668";
    unsigned num_threads  = 0;
    double time_limit_s   = 0.0;

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "--help" || arg1 == "-h" || arg1 == "/?") {
            print_help(argv[0]);
            return 0;
        }
        file_path = arg1;
    }
    if (argc >= 3) {
        tgt_str = argv[2];
    }
    if (argc >= 4) {
        num_threads = (unsigned)std::atoi(argv[3]);
    }
    if (argc >= 5) {
        time_limit_s = std::atof(argv[4]);
    }

    bool output_json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            output_json = true;
        }
    }

    if (!output_json) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << "           HGJ-96 STANDALONE SOLVER - MEMULAI INISIALISASI INSTANS              " << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "Sumber Berkas      : " << file_path << std::endl;
    }

    HGJInstance inst;
    if (!HGJInstance::load_from_txt_or_string(file_path, tgt_str, inst)) {
        if (output_json) {
            std::cout << "{\"error\": \"Gagal membaca berkas atau tidak ada elemen valid\"}\n";
        } else {
            std::cerr << "[ERROR]: Gagal membaca berkas atau tidak ada elemen angka valid ditemukan.\n";
        }
        return 1;
    }

    if (!output_json) {
        std::cout << "Jumlah Elemen (N)  : " << inst.elements.size() << " elemen (k = " << (inst.elements.size() / 2) << ")" << std::endl;
        std::cout << "Target Value (T)   : " << inst.target << std::endl;
        std::cout << "Total Sum (Sigma)  : " << inst.total_sum << std::endl;
        std::cout << "Densitas Masalah   : " << std::fixed << std::setprecision(4) << inst.density << std::endl;
        std::cout << "Alokasi Threads    : " << (num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) << " worker threads" << std::endl;
        std::cout << "Batas Waktu        : " << (time_limit_s <= 0.0 ? "Unlimited / Infinite (Sampai Ketemu)" : (std::to_string((long long)time_limit_s) + " detik")) << std::endl;
        std::cout << "Fitur Aktif        : Two-Pointer Sweep + Buffer Preallocation + Interval Pruning + Early Abort" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "Sedang menjalankan pencarian solusi paralel... Harap tunggu..." << std::endl;
    }

    HGJExecutionReport report = HGJSolver::solve(inst, num_threads, time_limit_s, !output_json);

    if (output_json) {
        std::cout << "{\n"
                  << "  \"solver\": \"HGJ\",\n"
                  << "  \"elements_count\": " << inst.elements.size() << ",\n"
                  << "  \"target\": \"" << inst.target << "\",\n"
                  << "  \"density\": " << inst.density << ",\n"
                  << "  \"status\": \"" << (report.solved ? "SOLVED (SATISFIABLE)" : "TIMEOUT / NOT FOUND") << "\",\n"
                  << "  \"solved\": " << (report.solved ? "true" : "false") << ",\n"
                  << "  \"runtime_ms\": " << std::fixed << std::setprecision(4) << report.runtime_ms << ",\n"
                  << "  \"peak_ram_mb\": " << std::fixed << std::setprecision(2) << report.peak_ram_mb << ",\n"
                  << "  \"partitions_evaluated\": " << report.partitions_evaluated << ",\n"
                  << "  \"modular_queries_evaluated\": " << report.modular_queries_evaluated << ",\n"
                  << "  \"verified\": " << (report.verified ? "true" : "false") << ",\n"
                  << "  \"witness_count\": " << report.witness.indices.size() << "\n"
                  << "}\n";
        return report.solved ? 0 : 1;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "                 HGJ-96 STANDALONE SOLVER - LAPORAN EKSEKUSI                    " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Status             : " << (report.solved ? "SOLVED (SATISFIABLE)" : "TIMEOUT / NOT FOUND") << std::endl;
    std::cout << "Total Runtime      : " << std::fixed << std::setprecision(2) << (report.runtime_ms / 1000.0) << " detik (" 
              << std::setprecision(4) << report.runtime_ms << " ms)" << std::endl;
    std::cout << "Partisi Diuji      : " << report.partitions_evaluated << " randomized balanced splits" << std::endl;
    std::cout << "Modular Queries    : " << report.modular_queries_evaluated << " queries modulo M" << std::endl;
    std::cout << "Kombinasi Dibangun : " << report.base_combinations_generated << " entries (C(24,12) lists)" << std::endl;
    std::cout << "Parity Prunes      : " << report.parity_prunes << " prunes (1-cycle bitwise filter)" << std::endl;
    std::cout << "Interval Prunes    : " << report.interval_prunes << " prunes (Tail-table bound filter)" << std::endl;
    std::cout << "High-Bit Prunes    : " << report.highbit_prunes << " prunes (32-bit SIMD filter)" << std::endl;
    std::cout << "Peak RAM Memory    : " << std::fixed << std::setprecision(2) << report.peak_ram_mb << " MB" << std::endl;
    std::cout << "L7 Verifier        : " << (report.verified ? "[100% INDEPENDENTLY VERIFIED VALID]" : "[VERIFICATION FAILED]") << std::endl;
    std::cout << "Pesan Verifikator  : " << report.verification_msg << std::endl;

    if (report.solved) {
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "SOLUSI SAKSI EKSAK DITEMUKAN (" << report.witness.indices.size() << " Elemen):" << std::endl;
        std::cout << "Indeks Elemen : [";
        for (size_t i = 0; i < report.witness.indices.size(); ++i) {
            std::cout << report.witness.indices[i];
            if (i + 1 < report.witness.indices.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Nilai-Nilai Elemen : [";
        for (size_t i = 0; i < report.witness.values.size(); ++i) {
            std::cout << report.witness.values[i];
            if (i + 1 < report.witness.values.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Verifikasi Jumlahan   : " << report.witness.sum << " (Target = " << inst.target << ")" << std::endl;
        std::cout << "Selisih Eksak         : " << (report.witness.sum == inst.target ? "0 (PAS 100%)" : "TIDAK COCOK") << std::endl;
    }
    std::cout << "================================================================================\n" << std::endl;

    return report.solved ? 0 : 1;
}
