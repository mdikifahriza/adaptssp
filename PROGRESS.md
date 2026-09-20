# PROGRESS — markshare_main (hybrid CPU/GPU)

Rencana induk: `rencana.md` (Revisi 3). Nomor langkah di bawah mengacu ke §11 rencana.
Terakhir diperbarui: 19 Sep 2026 (sesi 4: Langkah 4, alat tes, 4b, 4c, housekeeping — SEMUA DITULIS, BELUM ADA YANG DIKOMPILASI/DIJALANKAN) — **Langkah 1 + 2 selesai dan TERVERIFIKASI di sandbox CPU (diferensial SS vs brute force, sanitizer). Bug ke-3 (`filter_by_cardinality`) diperbaiki. Langkah 3 (CMake) LULUS di Colab T4: build nvcc sm_75 sukses, TER jalur GPU menemukan solusi valid, compute-sanitizer 0 error, diferensial SS 0 mismatch. Berikutnya: Langkah 4.**

## Status

- Langkah aktif: **verifikasi di Colab** untuk semua yang ditulis di sesi 4 (Langkah 4, 4b, 4c, alat tes). Cukup satu perintah: `bash tools/colab_tests.sh` (lihat "Cara verifikasi sesi 4"). Setelah itu: baca `work/profil_*.txt` -> putuskan K6 dan apakah Langkah 5/6 layak.
- Terakhir selesai: Langkah 3 (CMake baru, diuji di Colab T4). Langkah 1 + 2 sekarang juga terverifikasi pada build GPU di Colab (diferensial SS 0 mismatch).
- Sedang dikerjakan: —
- Terputus di tengah? Tidak.
- Utang yang masih tersisa: (a) langkah 9 Tes Colab (bandingan waktu GPU vs binary CPU-only pada mesin yang sama) belum menghasilkan output — ulangi dengan `time` bawaan bash, `/usr/bin/time` kemungkinan tidak ada di Colab; (b) baseline v0 asli tidak pernah disimpan — diganti dengan tes diferensial vs brute force (lebih kuat untuk SS), tapi TER hanya dicek lewat planted instance + verifikasi `.sol`, bukan diff byte-per-byte; (c) audit titik cek SS (galloping / return-on-verify-fail / duplikat) belum.

## Cara lanjut di chat baru

Upload: `rencana.md`, `PROGRESS.md`, dan semua file sumber terbaru (lihat "Daftar file"). Lalu tempel:

```
Baca PROGRESS.md dan rencana.md. Lanjutkan langkah berikutnya sesuai PROGRESS.md.
Jangan ulangi analisis yang sudah ada. Di akhir, kirim file hasil dan PROGRESS.md yang diperbarui.
```

Aturan tiap sesi: satu langkah (atau satu sub-langkah), edit terarah bukan tulis ulang file, verifikasi yang bisa dilakukan di sandbox (g++ saja, tanpa nvcc/GPU), lalu kirim file yang berubah + PROGRESS.md.

## Keputusan (default dari rencana §10; centang bila sudah dikonfirmasi)

- [x] K1 — input matriks m×n dibuang; `-f` wajib 1D; `markshare.hpp` dihapus **(diterapkan di kode sesi ini — belum dikonfirmasi eksplisit oleh user, masih pakai default rencana)**
- [ ] K2 — di SS GPU, CPU hanya orkestrator + verifier; hybrid penuh hanya di TER *(belum relevan — belum masuk Langkah 6)*
- [ ] K3 — default explorer `--extsol`: `max_swap_size=4`, ukuran in/out boleh beda, dedup `std::set` *(belum relevan — Langkah 4)*
- [ ] K4 — kalibrasi split TER adaptif proporsional per merge (tanpa benchmark offline) *(belum relevan — Langkah 5)*
- [ ] K5 — sort tetap `std::sort` CPU sampai profil 4b menyatakan lain *(belum relevan — Langkah 4b/8)*
- [ ] K6 — nasib `eps*`/`w1`/`w2` diputuskan setelah 4b (hapus vs implementasi paper-faithful) *(belum relevan — Langkah 4b)*

## Checklist

### Langkah 0 — Setup
- [x] `git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse` **(dilakukan di sandbox sesi ini untuk keperluan compile-test; TIDAK persisten — ulangi clone ini juga di Colab/lingkungan kerja nyata, dan commit `external/argparse` atau tambahkan sebagai submodule supaya tidak hilang)**
- [x] Build CPU-only versi asli di sandbox (tanpa `WITH_GPU`) berhasil — **catatan: ini build atas kode HASIL Langkah 1+2, bukan v0 asli** (v0 asli tidak pernah di-build terpisah, lihat catatan di bawah)
- [ ] Simpan output baseline (TER dan SS, instance kecil, `OMP_NUM_THREADS=1`) untuk tes regresi — **belum, karena v0 asli tidak dibangun sebagai titik pembanding terpisah sebelum Langkah 1 diterapkan**
- [ ] `git init` + commit awal "v0 asli" di Colab — **dilakukan user di Colab (di luar sandbox ini), tidak bisa dikonfirmasi dari sini**

### Langkah 1 — Correctness (§4.1, §4.3)
- [x] Bug 1: `TerEntry::psum` `int64_t` → `uint64_t` (`ter_solver.cuh`) + perbarui komentar header
- [x] `generate_half_base_pool`: buang semua cast `(int64_t)`
- [x] `merge_root_and_solve`: `target_u64`, `uv_sum`, `req_w`, lambda `lower_bound` jadi unsigned
- [x] Verifikasi Bug 1 (sesi 3): UBSan `signed-integer-overflow` pada versi v0 (tanpa `-fwrapv`), instance n=32/63-bit, melaporkan overflow nyata (contoh: `-1590236663257275846 + -9135427836569013149 cannot be represented in type 'long int'`); pada versi dengan `psum` unsigned: 0 laporan. Output TER identik dengan v0 pada p24/p28/big32 (dibandingkan di sandbox terpisah dengan build v0 asli). Pada kode sesi ini (Langkah 1+2+fix filter) TER di ASan+UBSan menemukan solusi dan `.sol` terverifikasi OK pada p24, p28, big32.
- [x] Bug 2: `can_fit_size_t()` cek total semua nilai dalam `u128` ≤ `UINT64_MAX`
- [x] Verifikasi Bug 2: `tools/difftest.py <bin> 80 11 --big` (n=16, nilai 2^61..2^62.3, target = jumlah 3 elemen, total > 2^64). Jalur lama (`can_fit_size_t` cek per nilai): **17 dari 80 tidak ditemukan** (sesi pengujian terpisah, basis kode v0). Kode sesi ini: **0 dari 80**.
- [x] **Bug 3 (BARU, tidak ada di rencana §4): heap-buffer-overflow di `filter_by_cardinality` (`main.cpp`)** — kompaksi branchless selalu menulis `fw[out]`/`fm[out]` untuk tiap elemen, termasuk elemen tak valid setelah entri valid terakhir, padahal buffer hanya `n_valid` → tulis 1 elemen di luar batas saat `out == n_valid`. Ditemukan lewat AddressSanitizer. **Fix:** alokasi `n_valid + 1`, lalu `resize(n_valid)` setelah kompaksi, return dengan `std::move`. **Bukti:** tanpa fix (kontrol = kode sesi ini dengan fix dicabut), `difftest.py bin/ctrl 300 1` → **19 dari 300 gagal, semuanya CRASH, bukan jawaban salah**: 17× SIGABRT (rc=-6), 2× SIGSEGV (rc=-11). Dengan fix: 0 gagal. Fungsi juga diuji terisolasi vs brute force (n=1..64, semua rentang kardinalitas ≤ 6) dengan ASan+UBSan: versi lama SEGV, versi baru `mismatch=0`. Fix ada di `main.cpp` (v2).
- [x] Regresi lain pada kode sesi ini (Langkah 1+2+fix Bug 3, CPU-only, sandbox): `difftest.py` SS vs brute-force MITM → 300 instance (seed 1): 0 mismatch; 1200 instance (seed 2): 0 mismatch; `--big` 80 instance: 0 mismatch; ASan+UBSan 200 instance (seed 3): 0 mismatch. Batas: n ≤ 22, hanya solver SS (bukan TER), brute force = MITM Python.

Perubahan tambahan yang dilakukan sekalian di sini (bukan di checklist asli, tapi bagian dari fix psum yang sama):
- Dibersihkan cast `(uint64_t)` redundan pada `psum` di `merge_level2`/`merge_level1` (aman, tidak wajib, per catatan rencana §4.1).
- Pesan CLI TER/SS dibedakan sesuai §4.4 (TER: "bukan bukti infeasible"; SS: "infeasible (exhaustive)" hanya kalau `k_radius < 0`, selainnya "tidak ditemukan pada rentang k yang dicoba").

### Langkah 1b — Audit exactness (§4.4)
- [ ] Brute-force checker (bitmask, n ≤ 24) — **belum, di luar scope sesi ini (user minta Langkah 1–2 saja)**
- [x] Tes diferensial SS vs brute force (`tools/difftest.py`, MITM Python, n ≤ 22, ~1700 instance + 80 instance `--big`) — 0 mismatch pada kode sesi ini. Catatan: script menghitung apa pun tanpa teks "Found feasible solution!" sebagai MISS, termasuk crash; sekarang rc proses ikut dicetak supaya crash terlihat.
- [ ] Audit titik cek (galloping, return-on-verify-fail, duplikat) — belum
- [x] Pesan output CLI dibedakan (TER tidak ketemu / SS exhaustive / SS `k_radius ≥ 0`) — **sudah, dikerjakan sekalian di Langkah 1/2 (lihat main.cpp akhir)**
- [ ] TER planted n=32–48 success rate — belum

### Langkah 2 — Dead code, CLI, AVX2 (§5.3, §6, §7)
- [x] Lepas `GpuData` dan `run_on_gpu` dari `print_info_line` (+ pemanggilnya di `shroeppel_shamir_1d`)
- [x] Hapus dead code sesuai daftar §6 di `main.cpp`: `unpack_mask`/`unpack_mask_into`, `print_subset_and_compute_sum`, `extract_subset`, `concat_vectors`, `print_four_list_solution`, `append_solution_to_file`, `write_four_list_solution_to_file`, `custom_hash_cpu`, `struct HashIdx`, `flatten_and_encode_tuples_cpu_kv`, `compute_scores_cpu`, `combine_scores_cpu`, `verify_solution`, `max_encodable_dimension`, `find_matching_pairs_cpu`, `evaluate_cpu`, `evaluate_gpu`, `struct PipelineBuffer`, `struct EvalResult`, `evaluate_gpu_or_cpu`, `print_and_verify_solution`, `compute_peak_k`, `checkpoint_path_for_instance`, `load_checkpoint_completed_k`, `append_checkpoint_completed_k`, `clear_checkpoint`, `struct KSetupResult`, `compute_k_setup`, `shroeppel_shamir_dim_reduced`. `HeapNode`/`extract_pairs_from_heap` dipastikan **tetap ada** (dipakai `shroeppel_shamir_1d`, jalur hidup).
- [x] Hapus `cuda_kernels.cu`, `cuda_kernels.cuh`, `markshare.hpp` — **file-file ini TIDAK disertakan lagi di output/proyek**; `#include "markshare.hpp"` dan `#include "cuda_kernels.cuh"` (di bawah `#ifdef WITH_GPU`) sudah dibuang dari `main.cpp`. **Aksi wajib untuk user:** hapus ketiga file fisiknya (`cuda_kernels.cu`, `cuda_kernels.cuh`, `markshare.hpp`) dari working tree/Colab — Claude tidak bisa menghapus file di luar sandbox percakapan ini.
- [x] Hapus cabang `else` di `main()`, fallback `MarkShareFeas(path)`; `-f` wajib (`argparse` `.required()`) + error jelas bila format bukan 1D
- [x] Hapus flag `-m -n -k -i -s --reduce`
- [x] Pertahankan `--max_pairs`, `--mem_budget_gb`, `max_pairs_per_chunk`, `detect_system_ram_bytes`, `compute_default_max_pairs`
- [x] Hapus `--gpu` dan `TerParams::use_gpu` (level 1 TER sekarang selalu mencoba jalur GPU di bawah `#ifdef WITH_GPU`, tanpa toggle runtime; kalau dikompilasi tanpa `WITH_GPU` otomatis jatuh ke CPU seperti sebelumnya)
- [x] SSE → AVX2 di `generate_subsets<uint64_t>`; include → `<immintrin.h>` (`_mm256_set1_epi64x/_mm256_loadu_si256/_mm256_storeu_si256/_mm256_add_epi64`, unroll 4×u64 bukan 2×)
- [x] Tinjau `<execution>` / `PSORT` / `USE_PARALLEL_SORT` — **dihapus total**: satu-satunya pemakai `PSORT` adalah `find_matching_pairs_cpu` yang sudah dead code dan sudah dihapus; `<execution>` dan `<future>` juga dihapus (tidak ada pemakai lain).
- [x] Build CPU-only **compile + link** lulus (g++ 13, `-std=c++17 -O2 -march=broadwell -fopenmp`, tanpa `WITH_GPU`, tanpa `-mavx2` eksplisit — `-march=broadwell` sudah mengimplikasikannya). `--help`, jalur tanpa `-f` (exit 1, pesan `-f: required.`), dan jalur file bukan-1D (exit 1, pesan error baru) **sudah dites manual dan berperilaku sesuai desain**.
- [x] Verifikasi Langkah 2 (sesi 3): kode hasil dead-code removal + AVX2 + fix Bug 3 di-build CPU-only (`g++ -O2 -march=broadwell -fopenmp`, dan varian `-fsanitize=address,undefined`) → lulus tes diferensial SS (0 mismatch, lihat Langkah 1) dan TER+SS end-to-end pada planted p24, p28, big32 (`.sol` diverifikasi dengan `tools/verify_sol.py`-setara). **Bukan** diff byte-per-byte terhadap baseline v0 (baseline itu memang tidak pernah disimpan).

### Langkah 3 — CMake (§10)
- [x] `project(markshare_main LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit REQUIRED)`, `find_package(OpenMP REQUIRED)` — **ditulis, belum di-build dengan nvcc**
- [x] `CUDA_ARCHITECTURES 75`; flag CXX `-march=broadwell -mtune=broadwell -funroll-loops`; nvcc host flags lewat `-Xcompiler=`; `-lineinfo` untuk profiling
- [x] Opsi `USE_CUDA` dan `ENABLE_NATIVE_OPT` **tidak ada** di CMake baru; tidak ada opsi build tanpa GPU (GPU wajib). `src/cuda_kernels.cu` tidak ada di daftar sumber.
- [x] **Keputusan (menyimpang dari kata-kata §10):** makro `WITH_GPU` **tetap didefinisikan, tanpa syarat**, lewat `target_compile_definitions(... PRIVATE WITH_GPU)`. Alasan: `ter_solver.cpp`/`ter_kernel.*` masih memakai guard `#ifdef WITH_GPU`; kalau definisinya dihapus begitu saja, jalur GPU tidak pernah ikut terkompilasi. Guard bisa dibuang dari kode belakangan kalau mau.
- [x] `CUDA_SEPARABLE_COMPILATION OFF` — cukup: hanya satu `.cu`, tanpa `__device__`/`__constant__`/`__managed__` lintas-file (grep di `ter_kernel.cu`/`.cuh`). LTO **tidak dinyalakan** (belum bisa diuji tanpa nvcc); coba di Colab kalau mau.
- [x] Auto-deteksi tata letak: pakai `src/` bila `src/main.cpp` ada, kalau tidak pakai direktori root (upload sesi ini flat; `rencana.md` menyebut `src/cuda_kernels.cu`, jadi repo asli mungkin memakai `src/`).
- [x] Uji sebagian di sandbox: varian CMake tanpa CUDA (CXX-only) → configure OK, `main.cpp`+`ter_solver.cpp` compile dengan `-march=broadwell -mtune=broadwell -funroll-loops -fopenmp -O3`, `-DWITH_GPU`, include argparse benar; satu-satunya error link adalah `undefined reference to run_level1_merge_gpu` (simbol yang disediakan `ter_kernel.cu`) — sesuai harapan.
- [x] Build di Colab (T4) lulus: configure OK, `ter_solver.cpp`, `main.cpp`, `ter_kernel.cu` (CUDA object) ter-compile, link sukses, `build/markshare_main` 438464 byte. Tidak ada error nvcc (warning tidak dilaporkan). CLI: `--help` OK, tanpa `-f` → `-f: required.` exit 1, file bukan 1D → pesan error format + exit 1.
- [x] Jalur GPU TER jalan dan hasilnya benar (Colab T4, 2 thread CPU): p24, p28, big32 → baris `Execution mode : Hybrid CPU (OpenMP) + GPU (Tesla T4 / CUDA)`, `SOLUTION FOUND` di run 1, `tools/verify_sol.py` OK. Waktu (GPU, run tunggal): p24 2.909 s, p28 0.835 s, big32 0.849 s. `compute-sanitizer --tool memcheck` pada p24: `ERROR SUMMARY: 0 errors` (run 276.8 s di bawah sanitizer, wajar).
- [x] Bukti GPU terpakai bersifat TIDAK LANGSUNG: polling `nvidia-smi` selama run berulang menunjukkan `memory.used` naik dari 0 MiB ke 105–109 MiB dan utilization 73–99 %. Belum ada bukti per-kernel dari `nsys`.
- [x] Tes diferensial SS pada build Colab: 300 instance (seed 1) mismatch=0; `--big` 80 instance mismatch=0.
- [ ] (opsional) konfirmasi per-kernel: `nsys profile -t cuda --stats=true ...` harus menampilkan `level1_merge_kernel`

### Langkah 4 — `zero_sum_swap.h` native + `--extsol` (§4.2, §9)
- [x] `zero_sum_swap.h` native ada (BFS bertingkat, `u128` eksak, cek total <= 2^128-1, validasi solusi awal, verifikasi ulang `sum == target` sebelum diterima, dedup `std::set`, pengaman `max_solutions`/`max_tiers`/`max_subsets_per_side`, status `capped` vs `closure`). Dibaca penuh sesi 4; header sendiri compile bersih dengan `-Wall -Wextra -Wconversion` + ASan/UBSan (satu-satunya warning: `__int128` di `-Wpedantic`, sama dengan `main.cpp`) dan 1 kasus mini jalan benar ({1,2,3,4}, target 5, awal {0,3} -> Tier 1 = 1 solusi baru, Tier 2 = 0).
- [x] `main.cpp` (v3): `#include "zero_sum_swap.h"`; flag baru `--extsol`, `--extsol_swap_size` (default 4, 1..6, divalidasi), `--extsol_max_solutions` (default 1000, 0 = tanpa batas); fungsi `run_extsol(...)`.
- [x] Solusi awal diambil dari TER (`res.solution_indices`) dan dari SS: `print_and_write_1d_ss_solution` dan `shroeppel_shamir_1d` mendapat parameter opsional `std::vector<size_t> *solution_out = nullptr` (indeks asli, karena `list1..list4` adalah potongan berurutan dari `values`, tanpa permutasi).
- [x] Keluaran: `<instance>.extsol`, satu bitstring per baris, baris 1 = solusi awal (sama dengan `.sol`). Pesan CLI membedakan `closure` (bukan bukti tak ada solusi lain di luar komponen terhubung) dan `capped`.
- [x] Compile-check `main.cpp` sesudah edit: `g++ -std=c++17 -O2 -march=broadwell -fopenmp -c main.cpp` tidak mencetak error (catatan: kode keluar yang tercetak berasal dari `head`, bukan g++, jadi anggap ini indikasi saja). **Link penuh + run belum dicoba.**
- [ ] **Belum diuji (semua):** `tools/test_swap.cpp` (tes diferensial explorer vs oracle BFS brute force, n <= 16, nilai duplikat dan > 2^64, kasus error dan cap) sudah ditulis tapi **belum pernah dijalankan** — sesi berhenti sebelum run karena permintaan user. Perintah: `g++ -std=c++17 -O2 -g -fsanitize=address,undefined -o tools/test_swap tools/test_swap.cpp && ./tools/test_swap` (jangan pakai `time` di `sh`; pakai `bash -c`).
- [ ] Tes end-to-end `--solver ss --extsol` dan `--solver ter --extsol` pada instance dengan banyak solusi (nilai kecil, n ~ 20-24): semua baris `.extsol` harus berjumlah target, unik, baris 1 == isi `.sol`.
- [ ] Build CMake/Colab dengan `main.cpp` v3 (belum).
- [ ] K3 (default explorer) menunggu konfirmasi user.

### Sesi 4 (lanjutan) — alat tes, 4b, 4c, housekeeping  **[ditulis, BELUM dikompilasi, BELUM dijalankan — permintaan user]**
Satu-satunya pemeriksaan yang dilakukan: `bash -n` pada skrip shell dan `ast.parse` pada skrip Python (sintaks saja). **Tidak ada satu pun file C++/CUDA yang dikompilasi setelah edit ini** (`main.cpp` dan `zero_sum_swap.h` sempat dikompilasi sebelum edit 4b/4c; `ter_solver.cpp`, `ter_kernel.cu` dan header barunya belum sama sekali). Perkirakan beberapa putaran perbaikan kompilasi di Colab, terutama di `ter_kernel.cu`.

**Alat (tools/)**
- [x] `genplant.py` (planted; `--k`, `--dups` untuk banyak solusi), `verify_sol.py` (`.sol` dan `--extsol`), `bruteforce.py` (MITM eksak n <= 30; `--component` = oracle BFS komponen swap untuk mengecek `.extsol`), `difftest.py` (SS vs brute force; `--big` untuk total > 2^64), `test_swap.cpp` (butuh `-I<folder sumber>` sekarang), `colab_tests.sh`, `setup_git.sh`.
- [x] `colab_tests.sh`: build GPU (CMake) + CPU-only (g++), test_swap, TER e2e, ekuivalensi 4c (CPU 1 thread dan GPU), `--extsol` e2e vs oracle, difftest, success rate + profil TER n=32/40/48, waktu GPU vs CPU. `QUICK=1` versi cepat, `FULL=1` menambah compute-sanitizer dan nsys. Ringkasan PASS/FAIL/SKIP di akhir; log di `colab_tests.log`.
- [x] `.gitignore`; `setup_git.sh` (git init, argparse di-vendor dengan membuang `.git`-nya, `REMOVE_OLD=1` menghapus `cuda_kernels.*` dan `markshare.hpp`); opsi `-DENABLE_LTO=ON` di CMake (default OFF, hanya unit C++, belum diuji).

**Langkah 4b — instrumentasi (risiko rendah)**
- [x] `ter_solve` mengukur per fase: gen_pool, sort_pool_right (sekali), L2, sortC, L1, root; ringkasan berawalan `[TER-PROFIL]` (rata-rata ms per run + persen, ukuran L2/L1, berapa list kena cap, run dengan L1 kosong, Mpasang/s untuk L1, panggilan GPU vs CPU, fallback GPU->CPU). Peringatan bila pool L3 lebih kecil dari `target_L3` (uji hipotesis n=96 di §4.4).
- [x] GPU: `cudaEvent` untuk h2d/kernel/d2h, jam dinding untuk alloc/free, dan waktu panggilan pertama (isolasi inisialisasi konteks CUDA — menguji dugaan "p24 2.9 s karena init"). `gpu_merge_stats_get()/reset()` di `ter_kernel.cuh`.
- [x] `--ter_stats --runs N`: tidak berhenti di solusi pertama, jalankan tepat N run, cetak `success_rate`. Ditolak bila tanpa `--runs` atau dengan `--autorestart` (`TerParams::continue_after_found`, `TerResult::successful_runs`).
- [x] Fallback GPU->CPU tidak lagi diam-diam: peringatan di stderr + penyebab (`[GPU] ...`); kernel launch/sync/memcpy sekarang dicek errornya (dulu diabaikan -> hasil kosong diam-diam), guard grid > 2^31-1 blok dan overflow |A|*|B|, buffer device RAII (tidak bocor di jalur error).
- Format baris `[Run k]` bertambah ` | ms: L2=.. sortC=.. L1=.. root=..` sebelum ` --> ...`; awal barisnya tetap (`L2 avg=.., L1=(..)`) supaya tetap bisa di-grep.
- [ ] Keputusan K6 (`eps*`/`w1`/`w2`) dan tuning `l1..r2`: **menunggu data dari `work/profil_*.txt`** — bagian ini memang keputusan user.

**Langkah 4c — sidecar key + bucket lookup (risiko sedang)**
- [x] `--ter_bucket` (default mati = jalur lama persis). `Level1Sidecar` (`a_ps`, `b_ps`, `c_keys`, `start`, `shift`) dibangun tiap `merge_level1`; `bits = floor(log2 |C|)` (rata-rata 1–2 kunci per bucket; n=64: tabel ~64 KB, kunci ~216 KB). Validasi: bila `c_keys` tidak terurut, sidecar ditolak dan jalur lama dipakai (dengan peringatan).
- [x] CPU: lookup memakai kunci uint64 saja; `TerEntry` disentuh hanya saat kunci cocok. GPU: kernel baru `level1_merge_kernel_bucket`; kernel lama tidak diubah.
- **Temuan (hasil membaca kode, BELUM dibuktikan lewat tes):** quick-reject `(u.pos & v.pos) & (u.neg | v.neg)` di `merge_level1` CPU selalu bernilai 0 bila `pos` dan `neg` tiap entri tidak pernah tumpang tindih di satu koordinat (invarian `TerEntry`) — jadi dia tidak pernah menolak apa pun. Jalur bucket membuangnya (hasil tetap sama karena `check_and_add_ternary` memvalidasi penuh). **Ini mengoreksi rencana §5.1 poin 8**: quick-reject TIDAK perlu ditambahkan ke kernel GPU. Bila tes 4c menemukan hasil beda, periksa asumsi invarian ini dulu.
- Kriteria lulus: CPU 1 thread -> baris `L2 avg=.., L1=(..)` tiap run identik antara binary search dan bucket; GPU -> identik bila tidak ada list kena cap (bila kena cap, urutan `atomicAdd` tak deterministik, bandingkan manual). Skrip sudah mengecek keduanya.
- [ ] Belum dikerjakan: bucket untuk `merge_level2` dan root (rencana hanya menyebut L1); tiling cache (langkah 7, tunggu profil).

**Cara verifikasi sesi 4 (di Colab, dari akar proyek)**
```bash
# taruh semua file sumber + CMakeLists.txt + tools/ dalam satu folder (folder src/ juga dikenali)
bash tools/colab_tests.sh            # atau QUICK=1 / FULL=1
# kompilasi C++/CUDA yang gagal akan muncul di bagian "1. Build"; tempel pesan errornya ke chat baru
```
Setelah lulus: `bash tools/setup_git.sh` (dan `REMOVE_OLD=1` bila sudah yakin menghapus `cuda_kernels.*`/`markshare.hpp`).

*(Langkah 5 dst. tidak berubah dari draft sebelumnya — lihat rencana.md §11 untuk isi lengkap.)*

## Daftar file (versi)

| File | Versi | Catatan |
|---|---|---|
| `main.cpp` | **v4 (sesi 4)** | v2 + `--extsol*` (v3) + `--ter_bucket`, `--ter_stats`. **Belum dikompilasi/diuji.** |
| `ter_solver.cpp` | **v2 (sesi 4)** — `merge_level1` + `ter_solve` ditulis ulang (profil, sidecar/bucket, mode statistik); belum dikompilasi. v1: | Langkah 1 (Bug 1) + Langkah 2 (`use_gpu` dibuang dari `merge_level1`) diterapkan. Dikirim di sesi ini. |
| `ter_solver.cuh` | **v2 (sesi 4)** — `Level1Sidecar`, `TerParams::{use_bucket_lookup,continue_after_found}`, `TerResult::successful_runs`. v1: | `TerEntry::psum` → `uint64_t`; `TerParams::use_gpu` dibuang. Dikirim di sesi ini. |
| `ter_kernel.cu`, `ter_kernel.cuh` | **v1 (sesi 4)** — ditulis ulang: kernel bucket, `GpuMergeStats`, cek error, RAII, guard grid; kernel lama tidak diubah. **Belum pernah dikompilasi dengan nvcc.** v0 (asli): | Tidak disentuh sesi ini — `psum` ikut jadi `uint64_t` otomatis lewat `TerEntry`, perilaku tidak berubah. Belum diverifikasi compile (butuh nvcc, tidak ada di sandbox). |
| `cuda_kernels.cu`, `cuda_kernels.cuh` | — | **DIHAPUS** (Langkah 2). User perlu hapus file fisiknya dari repo sendiri. |
| `markshare.hpp` | — | **DIHAPUS** (Langkah 2, K1). User perlu hapus file fisiknya dari repo sendiri. |
| `pairs_tuple.hpp`, `profiler.hpp` | v0 (asli) | tetap, tidak diubah |
| `CMakeLists.txt` | **v2 (sesi 4)** — + opsi `ENABLE_LTO` (OFF). v1 (sesi 3): | Ditulis ulang dari nol (v0 asli tidak ada di upload): CUDA wajib, sm_75, `WITH_GPU` selalu aktif. Belum di-build dengan nvcc. |
| `tools/difftest.py`, `tools/genplant.py`, `tools/verify_sol.py`, `tools/bruteforce.py`, `tools/colab_tests.sh`, `tools/setup_git.sh`, `.gitignore` | **ditulis ulang (sesi 4)** | versi sesi 3 tidak ada di upload; ini rekonstruksi, belum dijalankan. Argumen `genplant.py` sama (`OUT N BITS SEED`). |
| `zero_sum_swap.h` | v1 (native, tidak diubah sesi 4) | Langkah 4; header compile bersih, belum ada tes diferensial yang dijalankan |
| `tools/test_swap.cpp` | baru (sesi 4), belum dijalankan | tes diferensial explorer vs oracle BFS |
| `external/argparse/` | — | di-clone di sandbox untuk compile-test sesi ini (tidak persisten); clone ulang di Colab |

## Perintah tes

**Compile-test yang sudah dijalankan sesi ini (sandbox, g++ 13, tanpa nvcc/GPU):**
```bash
git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse
g++ -std=c++17 -O2 -march=broadwell -fopenmp -Iexternal/argparse/include/argparse -c main.cpp -o main.o
g++ -std=c++17 -O2 -march=broadwell -fopenmp -c ter_solver.cpp -o ter_solver.o
g++ -std=c++17 -O2 -march=broadwell -fopenmp -Iexternal/argparse/include/argparse main.cpp ter_solver.cpp -o markshare_main
./markshare_main --help          # tampilkan flag baru (tanpa -m/-n/-k/-i/-s/--reduce/--gpu)
./markshare_main                 # harus exit 1, "-f: required."
./markshare_main -f /tmp/bad.txt # harus exit 1, error format bukan 1D
```
Semua di atas **lulus**. Yang BELUM dijalankan (utang untuk sesi berikutnya, diminta user untuk berhenti sebelum sampai sini):
```bash
# contoh instance planted kecil untuk sanity-check TER dan SS end-to-end
./markshare_main -f <instance_1d.prb> --solver ter --runs 5
./markshare_main -f <instance_1d.prb> --solver ss

# instance dengan total nilai > 2^64 (planted), untuk buktikan fix Bug 2:
# can_fit_size_t() harus menolak jalur uint64_t dan pakai u128
./markshare_main -f <instance_total_over_2p64.prb> --solver ss
```

## Tes Colab (Langkah 3)

Semua di runtime GPU T4. Ringkasan (perintah lengkap ada di balasan sesi 3):
```bash
nvidia-smi; nvcc --version; cmake --version; lscpu | grep -o -wE "avx2|bmi2" | sort -u
# taruh sumber + CMakeLists.txt + tools/ di satu folder; hapus cuda_kernels.* dan markshare.hpp
git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/markshare_main --help; ./build/markshare_main   # exit 1, "-f: required."
python3 tools/genplant.py p24.prb 24 20 7; python3 tools/genplant.py p28.prb 28 24 3; python3 tools/genplant.py big32.prb 32 63 5
./build/markshare_main -f p28.prb --solver ter --runs 3      # harus mencetak "Hybrid CPU (OpenMP) + GPU"
python3 tools/verify_sol.py p28.prb
python3 tools/difftest.py build/markshare_main 300 1          # SS, 0 mismatch
python3 tools/difftest.py build/markshare_main 80 11 --big    # SS, 0 mismatch
compute-sanitizer --tool memcheck ./build/markshare_main -f p24.prb --solver ter --runs 1
nsys profile -t cuda --stats=true ./build/markshare_main -f p28.prb --solver ter --runs 1   # kernel harus muncul
```

## Masalah terbuka / catatan

- **Jalur GPU TER punya fallback CPU diam-diam**: `merge_level1` memanggil `run_level1_merge_gpu(...)`; kalau mengembalikan `false` (mis. `cudaMalloc` gagal) kode lanjut ke jalur CPU tanpa pesan. Jadi "solusi ditemukan" TIDAK membuktikan GPU terpakai — buktikan dengan `nsys`/polling `nvidia-smi` (lihat Tes Colab). Pertimbangkan cetak peringatan saat fallback (Langkah 5).
- **Waktu TER jalur GPU pada instance kecil tampak lambat**: p28 0.835 s di Colab, sedangkan di sandbox CPU-only (1 thread, bahkan dengan ASan+UBSan) p28 0.104 s dan p24 0.382 s. Mesin berbeda, jadi ini BUKAN kesimpulan; dugaan (belum diuji): inisialisasi konteks CUDA pada panggilan pertama (p24, yang dijalankan pertama, 2.9 s) plus `cudaMalloc`/`cudaMemcpy` per panggilan `run_level1_merge_gpu`, sedangkan ukuran L1 hanya 16384 entri. Perlu profil per fase (Langkah 4b) dan perbandingan pada mesin yang sama sebelum menyimpulkan apa pun.
- `Release` = `-O3 -DNDEBUG`, jadi semua `assert` (mis. di `extract_pairs_from_heap`) mati di build CMake. Tes sandbox sesi 3 dijalankan tanpa `-DNDEBUG` (assert aktif) dan lulus.
- Bug 3 muncul sebagai dua jenis crash berbeda (17× SIGABRT, 2× SIGSEGV pada 300 instance) — kemungkinan tergantung tata letak heap (dugaan, belum diselidiki). Karena itu satu kali lulus pada kode tanpa fix tidak membuktikan apa-apa.

- **Verifikasi regresi belum tuntas** (lihat di atas): perubahan psum signed→unsigned di root TER dianalisis benar secara manual (equality + lower_bound comparator konsisten sebelum/sesudah), tapi belum dibuktikan dengan run instance nyata dibandingkan baseline.
- **Bug 2 belum dibuktikan dengan instance nyata** (total > 2^64) — perbaikan `can_fit_size_t()` sendiri sudah benar secara statis (grand total dihitung di u128, incremental, dengan early-exit begitu melewati `UINT64_MAX`), tapi belum ada jalan `python`/`prb` generator untuk instance seperti itu di sesi ini.
- File fisik `cuda_kernels.cu`, `cuda_kernels.cuh`, `markshare.hpp` **masih ada di upload lama** — Claude tidak menghapus file di luar file yang diserahkan sebagai output sesi ini; user perlu menghapusnya sendiri dari repo kerja (Colab/git) sesuai keputusan K1/§6.
- `ter_kernel.cu`/`ter_kernel.cuh` tidak disentuh: `psum` sekarang `uint64_t` lewat `TerEntry`, jadi kode GPU otomatis ikut benar (cast `(uint64_t)` di kernel jadi redundant no-op, tidak mengubah perilaku), tapi belum pernah dikompilasi dengan nvcc di sesi manapun — tetap harus diverifikasi di Colab.
- `eps*` di `ter_default_params()` masih belum dipakai apa pun — nasibnya tetap diputuskan di Langkah 4b (tidak disentuh sesi ini).
- Sandbox Claude: g++ 13, 1 core, tanpa nvcc/GPU/cmake. Semua yang menyentuh `.cu` dan benchmark T4 tetap harus diuji di Colab.

## Log sesi

| Tanggal | Langkah | Hasil | File yang berubah |
|---|---|---|---|
| 19 Sep 2026 | 1 (Bug 1 + Bug 2) | Kode selesai; compile OK; **run/diff regresi belum dilakukan** (dihentikan oleh user) | `ter_solver.cuh`, `ter_solver.cpp`, `main.cpp` |
| 19 Sep 2026 | 2 (dead code, CLI, AVX2) | Kode selesai; compile+link OK; jalur error CLI dites manual; **diff output vs baseline belum dilakukan** | `main.cpp`, `ter_solver.cpp`, `ter_solver.cuh` |
| 19 Sep 2026 (sesi 3) | Fix Bug 3 + verifikasi Langkah 1/2 | `filter_by_cardinality` diperbaiki; kontrol tanpa fix 19/300 crash, dengan fix 0; diferensial SS 0 mismatch (~1700 + 80 `--big`), ASan/UBSan bersih | `main.cpp` |
| 19 Sep 2026 (sesi 4, lanjutan) | alat tes + 4b + 4c + housekeeping | Semua ditulis; hanya cek sintaks skrip; **C++/CUDA belum dikompilasi**; verifikasi lewat `tools/colab_tests.sh` | `main.cpp`, `ter_solver.cpp/.cuh`, `ter_kernel.cu/.cuh`, `CMakeLists.txt`, `tools/*`, `.gitignore`, `PROGRESS.md` |
| 19 Sep 2026 (sesi 4) | 4 (`--extsol`) | Kode `main.cpp` v3 ditulis; header dicek compile; **tidak ada tes yang dijalankan** (atas permintaan user) | `main.cpp`, `PROGRESS.md`, `tools/test_swap.cpp` |
| 19 Sep 2026 (sesi 3) | 3 (CMake) | `CMakeLists.txt` baru ditulis (CPU+GPU wajib, sm_75); uji parsial CXX-only OK di sandbox; **lulus build + tes GPU di Colab T4** (TER GPU benar, memcheck 0 error, diferensial SS 0 mismatch); perbandingan waktu vs CPU-only belum tuntas | `CMakeLists.txt`, `tools/*` |
