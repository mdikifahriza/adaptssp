#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#include <memory>
#include <filesystem>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "bcjSimd.hpp"

namespace fs = std::filesystem;

// Entry biner yang sangat padat untuk menghemat I/O dan memori
#pragma pack(push, 1)
struct DiskEntry {
    u64  key;       // Kunci sortir (residu / hash modulo)
    u128 sum;       // Nilai parsial sum 96-bit
    u128 pos_mask;  // Mask koordinat +1 dan +2
    u128 neg_mask;  // Mask koordinat -1
};
#pragma pack(pop)

// Perbandingan untuk Min-Heap External Sort
struct DiskEntryComparator {
    inline bool operator()(const DiskEntry& a, const DiskEntry& b) const {
        return a.key > b.key; // Min-heap (terkecil di top)
    }
};

// Buffered Reader untuk membaca file run biner secara sekuensial cepat
class BufferedDiskReader {
public:
    std::string filepath;
    FILE* fp = nullptr;
    std::vector<DiskEntry> buffer;
    size_t buf_idx = 0;
    size_t buf_len = 0;
    static const size_t CHUNK_SIZE = 16384; // 16K entri = ~768 KB buffer per run
    bool eof_reached = false;

    BufferedDiskReader(const std::string& path) : filepath(path) {
        fp = fopen(path.c_str(), "rb");
        if (fp) {
            buffer.resize(CHUNK_SIZE);
            refill();
        } else {
            eof_reached = true;
        }
    }

    ~BufferedDiskReader() {
        close();
    }

    void close() {
        if (fp) {
            fclose(fp);
            fp = nullptr;
        }
    }

    void refill() {
        if (!fp) { eof_reached = true; return; }
        buf_len = fread(buffer.data(), sizeof(DiskEntry), CHUNK_SIZE, fp);
        buf_idx = 0;
        if (buf_len == 0) {
            eof_reached = true;
            close();
        }
    }

    inline bool peek(DiskEntry& item) {
        if (buf_idx >= buf_len) refill();
        if (eof_reached || buf_len == 0) return false;
        item = buffer[buf_idx];
        return true;
    }

    inline bool pop(DiskEntry& item) {
        if (buf_idx >= buf_len) refill();
        if (eof_reached || buf_len == 0) return false;
        item = buffer[buf_idx++];
        return true;
    }
};

// Pengelola File Temporer dan External Sort
class DiskSpillManager {
public:
    std::string temp_dir;
    std::vector<std::string> temp_files;

    DiskSpillManager() {
        // Gunakan folder lokal scratch di folder kerja untuk performa NVMe maksimal
        temp_dir = "./scratch_bcj_spill";
        if (!fs::exists(temp_dir)) {
            fs::create_directories(temp_dir);
        }
    }

    ~DiskSpillManager() {
        cleanup();
    }

    void cleanup() {
        for (const auto& f : temp_files) {
            std::remove(f.c_str());
        }
        temp_files.clear();
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
    }

    std::string create_temp_path(const std::string& prefix) {
        static u64 counter = 0;
        std::string path = temp_dir + "/" + prefix + "_" + std::to_string(++counter) + ".bin";
        temp_files.push_back(path);
        return path;
    }

    // Tulis satu chunk memori ke disk
    void write_chunk_to_file(const std::vector<DiskEntry>& chunk, const std::string& path) {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) return;
        if (!chunk.empty()) {
            fwrite(chunk.data(), sizeof(DiskEntry), chunk.size(), fp);
        }
        fclose(fp);
    }

    // External K-Way Merge Sort: Menggabungkan beberapa run file biner menjadi 1 file terurut
    std::string external_kway_merge(const std::vector<std::string>& run_paths, const std::string& out_prefix) {
        if (run_paths.empty()) return "";
        if (run_paths.size() == 1) return run_paths[0];

        std::string merged_path = create_temp_path(out_prefix);
        FILE* out_fp = fopen(merged_path.c_str(), "wb");
        if (!out_fp) return "";

        // Min-heap berisi pair (DiskEntry, reader_id)
        struct HeapItem {
            DiskEntry entry;
            size_t reader_id;
            bool operator>(const HeapItem& other) const {
                return entry.key > other.entry.key;
            }
        };

        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> min_heap;
        std::vector<std::unique_ptr<BufferedDiskReader>> readers;

        for (size_t i = 0; i < run_paths.size(); ++i) {
            auto reader = std::make_unique<BufferedDiskReader>(run_paths[i]);
            DiskEntry item;
            if (reader->pop(item)) {
                min_heap.push({item, i});
            }
            readers.push_back(std::move(reader));
        }

        std::vector<DiskEntry> out_buf;
        const size_t OUT_CHUNK = 16384;
        out_buf.reserve(OUT_CHUNK);

        while (!min_heap.empty()) {
            HeapItem top = min_heap.top();
            min_heap.pop();

            out_buf.push_back(top.entry);
            if (out_buf.size() >= OUT_CHUNK) {
                fwrite(out_buf.data(), sizeof(DiskEntry), out_buf.size(), out_fp);
                out_buf.clear();
            }

            // Ambil entri berikutnya dari reader yang sama
            DiskEntry next_entry;
            if (readers[top.reader_id]->pop(next_entry)) {
                min_heap.push({next_entry, top.reader_id});
            }
        }

        if (!out_buf.empty()) {
            fwrite(out_buf.data(), sizeof(DiskEntry), out_buf.size(), out_fp);
            out_buf.clear();
        }
        fclose(out_fp);

        // Hapus file-file pecahan asal untuk hemat ruang SSD
        for (const auto& p : run_paths) {
            std::remove(p.c_str());
        }

        return merged_path;
    }
};
