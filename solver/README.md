# Subset-Sum Solver n=96 (window-based, v2)

## Status kerja (jujur, per saat file ini di-download)

Sudah selesai & teruji:
- `common.hpp` — helper int128 parse/print, n-choose-k.
- `bounds.hpp` — Fase 0.1, `compute_cardinality_bounds()` (k_min/k_max terbukti).
  **Hasil di instance benchmark kita: k_min=41, k_max=55** (lebar, sesuai
  dugaan di rencana karena semua elemen magnitude-nya mirip / density≈1).
- `residue_sieve.hpp` — Fase 0.2, `residue_cascade_sieve()` modulo prima
  kecil {3,5,7,11,13,17,19,23,29,31}. **Hasil: 0 elemen tereliminasi** untuk
  instance ini (sesuai peringatan di rencana: instance "acak murni" tidak
  banyak terbantu sieve modular — bukan bug, memang begitu ekspektasinya).
- `weight_window.hpp` — Fase 1, `window_for_block()` berbasis sebaran
  hypergeometric.
- `block_gen.hpp` — Fase 2, DFS+pruning per blok (sudah dioptimasi: versi
  awal pakai `std::function` yang lambat, sudah diganti fungsi biasa).
- `join_core.hpp` — Fase 3, join modular adaptif (mod_bits menyesuaikan
  ukuran list, bukan fixed 24-bit) + hemat memori (blok 0/2 langsung jadi
  `ModElement` tanpa duplikasi, blok 1/3 tetap `Element`).
- `solver.cpp` — orkestrasi penuh, sudah **berhasil dikompilasi** dan
  **sudah diverifikasi jalan** (Fase 0 tervalidasi, satu percobaan m=48
  dengan window sempit berhasil generate ~11.6 juta kandidat/blok tanpa OOM).

**Yang BELUM selesai saat file ini diserahkan:**
Saya belum menuntaskan satu run penuh (banyak nilai m, trial cukup) untuk
benar-benar MENEMUKAN solusi 96-elemen tsb. Temuan penting dari eksperimen
sejauh ini (silakan verifikasi ulang):

1. Untuk instance ini (elemen ~2^96, density≈1), sebaran bobot per-blok
   (24 elemen/blok) punya standar deviasi cukup besar (~2.1) relatif
   terhadap ukuran bloknya, sehingga **window sempit pun (delta_std kecil)
   tetap menghasilkan ~9-12 juta kandidat/blok** — teknik window TIDAK
   memangkas ukuran sebanyak yang diharapkan rencana awal untuk instance
   acak seperti ini (ini sudah diprediksi sebagai risiko di dokumen rencana,
   bagian "Ringkasan Risiko").
2. Mesin ini terbatas 1 core / 3.9GB RAM (sama seperti yang disebut di
   rencana awal). Dengan ~11.6 juta kandidat/blok, budget memori realistis
   ada di sekitar 2.5-3GB — pas-pasan, jadi `memory_budget_mb` perlu diatur
   hati-hati (lihat parameter di bawah).
3. Waktu per percobaan m (generate + beberapa trial join) berkisar puluhan
   detik di mesin 1-core ini sebelum optimasi DFS; setelah optimasi
   (mengganti `std::function` dengan fungsi biasa) seharusnya jauh lebih
   cepat, tapi saya belum sempat mengukur ulang & menjalankan skenario
   penuh (banyak m x banyak trial) sampai selesai karena keterbatasan waktu
   sesi ini.

## Cara pakai

```bash
g++ -O3 -std=c++17 -march=native solver.cpp -o solver
./solver <time_budget_detik> <delta_std> <trials_per_m> <memory_budget_mb>
# contoh:
./solver 300 1.0 30 3000
```

- `time_budget_detik`: batas waktu total (wall clock). Solver berhenti jujur
  melaporkan "belum ketemu" (bukan mengklaim UNSAT) kalau habis.
- `delta_std`: lebar window bobot per blok dalam satuan std-dev
  hypergeometric. Makin besar = makin lengkap tapi makin besar memori/waktu.
  Untuk instance ini, `0.5-1.0` sudah menghasilkan jutaan kandidat/blok;
  `>1.5` mendekati enumerasi penuh 2^24/blok (~4GB+, kemungkinan OOM di
  mesin 3.9GB).
- `trials_per_m`: jumlah percobaan filter modular acak (R1) per nilai m.
- `memory_budget_mb`: kalau estimasi kasar suatu m melebihi ini, m tsb
  dilewati (dicoba m lain) — supaya tidak OOM di tengah jalan.

## Saran lanjutan (belum sempat saya coba semua)

1. **Naikkan `delta_std` bertahap** (mis. 0.6 → 0.8 → 1.0) sambil pantau
   `free -h`, cari titik terbesar yang masih aman di mesin Anda.
2. **Jalankan lebih lama** (`time_budget` besar, mis. 600-1800 detik) dan
   biarkan solver mencoba banyak nilai m dari tengah [k_min,k_max] ke tepi —
   ini caranya sudah diimplementasikan (lihat `m_order` di `solver.cpp`).
3. Kalau tersedia mesin dengan **RAM lebih besar / multi-core**, bagian
   "FASE 4 — Paralelisasi" di rencana awal (jalankan beberapa m sekaligus
   di core berbeda, early-exit begitu satu ketemu) akan sangat membantu —
   kerangka fungsi (`join_core_search` per-m independen) sudah dirancang
   supaya mudah dipararelkan (tinggal bungkus loop `for (int m : m_order)`
   dengan thread pool).
4. Kalau butuh **completeness murni tanpa asumsi** (menjamin ketemu bila
   ada solusi, bukan cuma probabilistik), satu-satunya cara aman adalah
   `delta_std` besar mendekati penuh — yang berarti kembali ke masalah
   memori 2^24/blok seperti versi awal. Alternatif sungguhan untuk itu
   adalah algoritma Schroeppel-Shamir (streaming heap, O(2^{n/2}) waktu
   tapi O(2^{n/4}) memori TANPA kehilangan completeness) yang disebut di
   akhir dokumen rencana sebagai opsi cadangan — belum diimplementasikan
   di sini.

## Struktur file

```
common.hpp          # int128 parse/print, n-choose-k
bounds.hpp           # Fase 0.1: compute_cardinality_bounds()
residue_sieve.hpp     # Fase 0.2: residue_cascade_sieve()
weight_window.hpp      # Fase 1: window_for_block()
block_gen.hpp            # Fase 2: gen_block_candidates() (DFS+pruning, tanpa std::function)
join_core.hpp              # Fase 3: join_core_search() modular adaptif, hemat memori
solver.cpp                  # main: orkestrasi Fase 0-3, parsing instance, laporan hasil
solver_binary                # hasil kompilasi -O3 -march=native (dibuat di sandbox ini;
                              # kompilasi ulang di mesin Anda kalau arsitektur CPU beda)
```
