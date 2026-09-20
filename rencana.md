# Rencana: Fix Correctness + Hybrid CPU/GPU + Optimasi markshare_main

> **Revisi 3** — memasukkan hasil review eksternal (celah exactness & kecepatan), setelah tiap poin dicek ke kode.
> Diterima: §4.4 baru (batas exactness), audit SS + tes diferensial, profiling per fase, kalibrasi split konkret,
> sidecar key + bucket lookup (§5.1 poin 10), keputusan K4–K6, langkah 1b/4b/4c di §11, dan `PROGRESS.md`.
> Diturunkan: klaim "realloc GPU paling signifikan" (overhead ≈ ms vs kernel ≈ ratusan ms → hygiene, bukan prioritas;
> koreksi juga atas klaim saya sendiri di Revisi 2) dan "sort wajib pindah ke thrust" (list kecil → ukur dulu).
> Sudah terjawab sejak Revisi 2: `find_matching_pairs_cpu` dan overlap nama fungsi (§6).
>
> **Revisi 2** — disinkronkan dengan kode asli (hasil baca ulang + grep pemakaian fungsi). Perubahan utama:
> (1) §4.1 diperluas + catatan `-fwrapv`; (2) §4.3 baru: overflow jalur SS `uint64_t`;
> (3) §5.1 catatan buffer persisten & SoA; (4) **§5.2 diganti** desain enumerasi per rentang skor;
> (5) §5.3 & §6 daftar flag/dead code dikoreksi; (6) §7 BMI2 dikoreksi; (7) §10–11 blocker & urutan diperbarui;
> (8) §12 workflow baru (hemat token, tahan putus di tengah).

## 1. Tujuan

1. Menghilangkan bug correctness (UB psum, missing verification di zero-sum-swap) **sebelum** optimasi apa pun.
2. Menghapus CLI flag `--gpu` — CPU dan GPU **selalu dipakai bersamaan** untuk kedua solver (`ter` dan `ss`), porsi kerja dibagi otomatis agar total waktu eksekusi paling cepat, bukan dipilih manual oleh user.
3. Merampingkan kode: hapus dead code pipeline `MarkShareFeas` lama, ganti SSE dengan AVX2 pada jalur yang memang masih dipakai.
4. Optimasi CPU (Xeon Broadwell) dan GPU (Tesla T4) sesuai spek asli yang sudah dikonfirmasi.
5. **Kejujuran hasil:** output CLI tidak boleh mengklaim "infeasible" tanpa dasar — TER bersifat heuristik/probabilistik, SS exhaustive hanya bila `--k_radius -1` dan lolos audit (§4.4).

---

## 2. Spesifikasi hardware target (dikonfirmasi dari output asli)

**CPU** — `lscpu` Colab:
- Intel Xeon @ 2.20GHz, family 6, model 79 (Broadwell)
- 2 CPU logis, 1 core fisik, 2 thread/core (HyperThreading) — **bukan 2 core fisik**
- Flags relevan: `sse sse2 ssse3 sse4_1 sse4_2 fma popcnt aes avx avx2 bmi1 bmi2 rdseed adx` — **tidak ada AVX-512**
- L1d 32 KiB, L1i 32 KiB, L2 256 KiB, L3 55 MiB (1 instance masing-masing, shared di 1 core)

**GPU** — `nvidia-smi` + `nvcc`:
- Tesla T4, compute capability 7.5 (Turing), 15360 MiB VRAM
- Driver 580.82.07, CUDA runtime 13.0, **nvcc 12.8** (compile pakai toolkit 12.8, `-arch=sm_75`)

Implikasi desain:
- Compiler flags CPU: `-O3 -march=broadwell -mtune=broadwell -funroll-loops -fopenmp`
- SIMD width maksimum: AVX2 (256-bit, 4×u64/instruksi). Tidak boleh pakai AVX-512 intrinsics.
- Hanya 2 thread logis → paralelisme OpenMP CPU terbatas; GPU (2560 core) akan menanggung mayoritas beban kerja untuk loop O(|A|×|B|). Desain hybrid harus mengasumsikan CPU sebagai "helper kecil", bukan partner setara.
- `-arch=sm_75` wajib di nvcc (bukan default/sm_70).

---

## 3. Inventarisasi file

| File | Status | Catatan |
|---|---|---|
| `main.cpp` | dibaca penuh | entry point, argparse, 2 pipeline (1D SSP via `-f`, dan MarkShareFeas m×n lama) |
| `ter_solver.cpp` / `ter_solver.cuh` | dibaca penuh | TER solver, sudah pakai `ter_u128` untuk weights/target |
| `ter_kernel.cu` / `ter_kernel.cuh` | dibaca penuh | GPU path TER, hanya meng-cover Level 1 merge |
| `cuda_kernels.cu` / `cuda_kernels.cuh` | dibaca penuh | **100% terikat ke `MarkShareFeas`** (GPU pipeline lama, dead code) |
| `markshare.hpp` | dibaca penuh | class `MarkShareFeas`, dipakai jalur lama (no `-f`) **dan** fallback `-f` non-1D. **Akan dihapus total** (keputusan di §10) |
| `pairs_tuple.hpp`, `profiler.hpp` | dibaca penuh | utilitas kecil, tetap dipakai |
| `zero_sum_swap_berlapis.h` | dibaca penuh | **bukan kode native program ini** — berasal dari program lain, tergantung `solver_header.h` milik program asalnya. Hanya **logikanya** yang dipakai (BFS bertingkat via swap ≤4in/≤4out, verifikasi presisi penuh), ditulis ulang native dengan tipe project ini (§4.2 baru). **File aslinya tidak ada di upload terakhir** — implementasi berpegang pada deskripsi §4.2 |
| `solver_header.h` | **tidak relevan lagi** | milik program lain (asal `zero_sum_swap_berlapis.h`), tidak perlu diupload — logika di-porting native pakai tipe `main.cpp` sendiri (`u128 = unsigned __int128`, `u64 = uint64_t`) |
| `argparse.hpp` | **teridentifikasi** | library `p-ranav/argparse` (single-header, API publik dikenal: `add_argument().help().default_value().store_into()/.flag()`). Disetup di Colab via `!git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse`. Bukan lagi blocker. |

---

## 4. Bug correctness — wajib fix duluan, sebelum apa pun di bawah ini

### 4.1 Bug 1 — `TerEntry::psum` signed, UB (scope diperluas setelah diverifikasi ke kode)

**Fakta dari kode:** penjumlahan/pengurangan `int64_t` psum terjadi di lebih dari satu tempat, bukan hanya root:
- `merge_root_and_solve`: `u.psum + v.psum`, `target_signed - uv_sum`, comparator sort & `lower_bound` (matching **exact**, tanpa mask).
- `merge_level2`: `combined.psum = u.psum + it->psum`.
- `merge_level1` (CPU) dan `level1_merge_kernel` (GPU): `(uint64_t)(u.psum + v.psum) & mask` — penjumlahan dilakukan **signed dulu**, baru di-cast.
- `check_and_add_ternary` (CPU) dan `check_and_add_ternary_gpu`: `out.psum = u.psum + v.psum + w.psum`.
- `generate_half_base_pool`: semua `e.psum = (int64_t)(...)`.

Matching di L1/L2 memang memakai `(uint64_t)psum & mask`, tapi operasi tambahnya tetap signed — jadi klaim lama "L1/L2 sudah benar" hanya berlaku untuk perbandingannya, bukan penjumlahannya.

**Status praktis saat ini:** `CMakeLists.txt` memakai `-fwrapv` (C++ dan host-side nvcc via `-Xcompiler`), sehingga overflow signed *terdefinisi* dan bug ini tidak muncul di build sekarang. Flag baru di §7 **tidak** menyertakan `-fwrapv`, jadi fix ini wajib sebelum flag diganti. `-fwrapv` boleh tetap dipertahankan sebagai lapisan pengaman tambahan, tapi bukan pengganti fix.

**Fix (satu perubahan tipe membereskan semua titik di atas):**
- `TerEntry::psum`: `int64_t` → `uint64_t` (`ter_solver.cuh`), dan perbarui komentar header yang menyebut "psum signed/int64".
- `generate_half_base_pool`: buang cast `(int64_t)`, assign langsung `uint64_t` (wraparound unsigned terdefinisi).
- `merge_root_and_solve`:
  - `int64_t target_signed = (int64_t)lo64(target);` → `uint64_t target_u64 = lo64(target);`
  - `int64_t uv_sum = u.psum + v.psum;` → `uint64_t uv_sum = ...;`
  - `int64_t req_w = target_signed - uv_sum;` → `uint64_t req_w = target_u64 - uv_sum;`
  - lambda `lower_bound`: `int64_t val` → `uint64_t val`. Comparator sort ikut unsigned otomatis.
- Cast `(uint64_t)x.psum & mask` di L1/L2/kernel GPU jadi redundan; boleh dibersihkan, tidak wajib. Logika kernel tidak berubah.

**Verifikasi:** urutan sort di root berubah dari signed ke unsigned, tapi yang dipakai hanya kesamaan (equality) dan `lower_bound` dengan comparator yang konsisten, jadi hasilnya harus identik. Bandingkan output sebelum/sesudah (`run_idx`, ukuran L1, solusi) dengan `OMP_NUM_THREADS=1` (deterministik; `rng(1337)` tetap). Jalur GPU tidak deterministik urutannya (`atomicAdd`), jadi bandingkan ukuran & hasil akhir saja.

### 4.2 Zero-sum-swap explorer — porting LOGIKA saja, bukan kode, ke tipe native project ini

**Klarifikasi penting:** `zero_sum_swap_berlapis.h` yang dibaca sebelumnya bukan bagian dari codebase ini — itu kode dari program lain (butuh `solver_header.h` milik program asalnya, yang memang tidak akan pernah kita punya, dan memang tidak perlu). Yang diambil dari situ hanya **logikanya**:

- Struktur BFS bertingkat (tiered/wavefront): Tier 0 = solusi awal dari TER/SS, Tier N = hasil swap dari Tier N-1, berhenti saat closure (tidak ada solusi baru) atau `max_tiers`/`max_solutions` tercapai.
- Per node: generate subset ≤`max_swap_size` (default 4) elemen dari sisi "in" solusi dan sisi "out" (bukan bagian solusi), cari pasangan in/out yang jumlahnya sama (net-zero swap), match cepat pakai residue/hash lalu **wajib direverifikasi presisi penuh** sebelum diterima.
- Dedup solusi via `std::set<std::vector<int>>` dari indeks terurut.

**Implementasi baru (native, file sendiri — mis. `zero_sum_swap.h` di project ini, atau ditanam langsung di `main.cpp`):**
- Tipe dipakai konsisten dengan project ini: `u128 = unsigned __int128` (sudah ada di `main.cpp`), `u64 = uint64_t` (dideklarasikan lokal, bukan diimpor dari header asing).
- Struct minimal yang benar-benar dibutuhkan, didesain dari nol sesuai kebutuhan TER/SS di sini (bukan disalin dari `Element`/`SolutionWitness`/`Instance`/`SolverBudget`/`ExecutionStats` milik program lain):
  ```cpp
  struct SwapElement { size_t orig_idx; u128 val; };         // representasi 1 elemen instance
  struct SwapWitness {                                        // 1 solusi (set indeks + sum tervalidasi)
      std::vector<size_t> indices;                            // original_indices, sudah sorted
      u128 sum;                                                // presisi penuh, WAJIB == target sebelum diterima
  };
  struct SwapExploreResult {
      std::vector<SwapWitness> all_solutions;
      int tiers_run;
      bool capped;                                             // true kalau berhenti karena batas, bukan closure
  };
  ```
- Matching cepat level `u64` (residue dari `lo64(val)`, sama pola dengan `psum` di TER) sebagai filter awal — **tidak berubah dari ide aslinya**, ini optimisasi valid.
- **Fix bug utama (wajib, ini alasan kenapa logika lama tidak boleh disalin mentah):** setelah kandidat lolos filter `u64`, hitung sum presisi penuh dari `weights` asli (bukan dari residue) pakai `u128`, lalu **verifikasi wajib** `sum == target` (target instance, bukan cuma `base.sum`) sebelum masuk `all_solutions` — persis pola `ver_sum == target` yang sudah benar di `merge_root_and_solve` TER solver. Ini yang hilang total di kode sumber, jadi di implementasi native ini langsung ditulis benar dari awal, bukan "ditambal".
- Karena ditulis native dari nol: tidak ada lagi risiko Bug 3 (truncation `Element::val` u128→u64) — tipe elemen langsung didesain `u128` sejak awal supaya konsisten dengan `ter_u128`/`u128` yang sudah dipakai `ter_solve()`, tidak ada asumsi "elemen pasti muat u64" yang diam-diam salah untuk instance besar (`n96.prb`).

### 4.3 Bug 2 (baru, dugaan kuat — belum diuji pada instance nyata) — overflow jalur SS `uint64_t`

`can_fit_size_t()` di `main.cpp` hanya mengecek bahwa **tiap nilai** dan **target** muat `size_t`. Tapi jalur `shroeppel_shamir_1d<uint64_t>` menjumlahkan banyak nilai: `P[k]`/`S[k]` di `compute_k_window` (dan `compute_k_priority_profile`), semua subset-sum hasil `generate_subsets`, skor pasangan di heap, dan `sum` di `print_and_write_1d_ss_solution`. Kalau jumlah total > 2^64, semuanya wrap diam-diam: perbandingan `score < target` jadi salah (solusi terlewat), dan verifikasi akhir `sum != target` ikut dihitung modulo 2^64 sehingga bisa **lolos palsu**. Jalur TER tidak terdampak (verifikasi u128 penuh).

**Fix (murah):** perluas `can_fit_size_t()` — hitung total semua `values` dalam `u128`; pakai jalur `uint64_t` hanya kalau total ≤ `UINT64_MAX` (target otomatis ikut aman). Selain itu pakai `shroeppel_shamir_1d<u128>`. Konsekuensi: instance besar jatuh ke jalur u128 yang lebih lambat dan belum ter-GPU-kan (lihat §5.2) — ini harga kebenaran.

**Verifikasi:** `n64.prb` belum ada di upload. Buat instance uji sendiri (planted solution, nilai ~2^58–2^63) yang memang membuat total > 2^64; jalur lama harus gagal/keliru, jalur baru benar.

### 4.4 Batas exactness solver (wajib didokumentasikan & tampil di output CLI)

**TER — heuristik/probabilistik, bukan exhaustive.** Tiap restart memilih `s1[k]`, `s2[m]` acak; "tidak ketemu" setelah N run atau timeout **bukan** bukti infeasible, dan `--autorestart` (`max_restarts=0`) hanya berarti "jalan terus", tanpa jaminan matematis. Selain sifat acak algoritmanya, implementasi ini punya pemotongan tambahan: pool L3 (`generate_half_base_pool`) berhenti di komposisi ≤3 satu / ≤1 minus-satu dan dipotong di `target_size*2`; L2/L1 dipotong di `max_cap`; hasil L1 dari GPU terpotong lewat `atomicAdd` tanpa urutan deterministik. Solusi yang ditemukan tetap **benar** (`ver_sum == target` presisi penuh); yang tidak dijamin adalah **ditemukannya**.

**Parameter `eps*` di `ter_default_params()` tidak berpengaruh apa pun** (diverifikasi dengan grep). `eps01/eps11/eps02/eps12/eps22` hanya dipakai `compute_derived()` untuk menghitung `w1`/`w2`, dan `w1`/`w2` tidak dibaca di mana pun. Yang benar-benar mengatur solver adalah konstanta hardcode `l1=0.2221, l2=0.2147, l3=0.1922, r1=0.57365, r2=0.1698` (→ `target_L*`, `b1`, `b2`), bergantung pada `n` hanya lewat `floor(r*n)`. Skrip `ter_run.py` sumber angka-angka itu tidak ada di codebase, dan belum ada bukti nilainya optimal untuk n=64/96 pada implementasi ini.

**Perkiraan kasar (dihitung dari kode, BELUM diukur):**
- n=64: `b2=10`, `b1=36`, pool L3 ≈ 6,5k/sisi, L2 ≈ 27k (kena cap), ≈ 7,5×10^8 pasangan per merge L1, hasil L1 hanya ≈ ratusan entri (≈ |L2|³/2^36).
- n=96: `b2=16`, `b1=55`, pool L3 hanya ≈ 72k dari target ≈ 356k (generator kehabisan komposisi), sehingga L1 kemungkinan hampir selalu kosong → peluang sukses per restart di n=96 bisa sangat kecil.
Ini hipotesis yang diuji di langkah 4b, bukan kesimpulan.

**SS — exhaustive, dengan dua pengecualian yang harus ditandai:**
- `--k_radius ≥ 0` melewati `k` di luar `peak_k ± r` → tidak exhaustive. Padahal `main()` mencetak "Instance was infeasible or not found by current solver" untuk semua kasus tidak ketemu.
- Jalur `uint64_t` bisa overflow (Bug 2, §4.3).

`shroeppel_shamir_1d` belum diaudit dengan tes, jadi belum boleh disebut exact. Titik cek (**belum terbukti bug**): (a) galloping skip dengan batas `pos+1 < size` di kedua heap — elemen terakhir list tidak pernah ikut di-skip; (b) `return print_and_write_1d_ss_solution(...)` menghentikan **seluruh** loop `k` bila verifikasi gagal; (c) penanganan nilai duplikat di `extract_pairs_from_heap`. Bukti diminta dari tes diferensial, bukan dari dugaan.

**Tindakan:**
1. Pesan output dibedakan: TER → "tidak ditemukan dalam N run (bukan bukti infeasible)"; SS dengan `k_radius = -1` dan lolos audit → "infeasible (exhaustive)"; SS dengan `k_radius ≥ 0` → "tidak ditemukan pada rentang k yang dicoba".
2. Tes diferensial (bisa di sandbox): instance acak n ≤ 24 (nilai duplikat, target 0/total, total > 2^64 untuk Bug 2) — SS vs brute force bitmask; TER dengan planted solution n=32–48, catat success rate per run.
3. `eps*`/`w1`/`w2`: putuskan setelah 4b — hapus (jujur), atau ganti generator pool agar benar-benar mengikuti distribusi paper. Selama belum diputuskan, jangan menyebut parameter ini "optimal".

---

## 5. Arsitektur eksekusi baru: CPU + GPU digabung, hapus `--gpu`

Keputusan desain baru (bukan pilihan manual via flag): setiap solve — baik `--solver ter` maupun `--solver ss` — **selalu memakai CPU dan GPU bersamaan**, porsi kerja dibagi otomatis untuk memaksimalkan throughput, bukan salah satu saja.

### 5.1 TER solver — hybrid execution

Kondisi saat ini: hanya `merge_level1` yang punya GPU path (`run_level1_merge_gpu`), itu pun all-or-nothing (`use_gpu` boolean) — kalau GPU dipakai, CPU menganggur total di level itu; kalau tidak, GPU menganggur total.

**Desain baru — workload split + overlap, bukan all-or-nothing:**

1. **Split ruang pasangan `A×B` di `merge_level1`** antara CPU dan GPU berdasarkan rasio kalibrasi (bukan hardcode 50/50 — GPU T4 jauh lebih kuat dari 2 thread Broadwell untuk loop O(|A|×|B|), rasio realistis kemungkinan ~90-97% ke GPU / 3-10% ke CPU, tapi harus dikalibrasi, lihat poin 3).
2. **Jalankan GPU secara async** (`cudaMemcpyAsync` + kernel launch di stream terpisah, tidak `cudaDeviceSynchronize()` langsung) sehingga CPU bisa mengerjakan porsi slice-nya sendiri **sambil** GPU jalan, baru di-join di akhir. Ini bedanya dari kode lama yang blocking (`cudaDeviceSynchronize()` dipanggil segera setelah launch di `run_level1_merge_gpu`).
3. **Kalibrasi rasio split:** jalankan micro-benchmark singkat sekali di awal proses (bukan per-run) — ukur throughput CPU merge_level1 vs GPU merge_level1 pada ukuran kecil representatif, simpan rasio, pakai untuk semua run berikutnya. Alternatif lebih sederhana: rasio statis berbasis spek diketahui (T4 2560 core vs CPU 2 thread), dikoreksi via 1-2 run kalibrasi nyata di awal `ter_solve()`.
4. **Pipelining lintas-level (opsional, fase lanjutan):** selagi GPU mengerjakan `merge_level1` untuk slot L1[j], CPU bisa mulai kerjakan `merge_level2` untuk run berikutnya atau post-processing root search dari run sebelumnya — overlap lintas iterasi restart loop, bukan cuma lintas A×B split.
5. `merge_level2` (9× per run, CPU-only saat ini) — evaluasi opsional offload ke GPU juga, tapi hanya kalau `|L3_left| × |sorted_right|` melebihi ambang batas tertentu (list kecil, overhead async launch GPU bisa lebih mahal dari kerjaan itu sendiri di CPU).

6. **Buffer & transfer persisten (hygiene murah — bukan bottleneck utama).** `run_level1_merge_gpu` melakukan `cudaMalloc` → 3× `cudaMemcpy` → kernel → `cudaFree` tiap panggilan (3× per run). Alokasikan `d_A/d_B/d_C/d_out/d_out_count` sekali (kapasitas maksimum, resize hanya bila perlu), pakai staging buffer pinned (`cudaHostAlloc`; `std::vector` biasa itu pageable) + `cudaMemcpyAsync` di stream. **Koreksi klaim Revisi 2:** perkiraan kasar n=64 — overhead alokasi+transfer ≈ ms, kernel ≈ ratusan ms untuk 7,5×10^8 pasangan, jadi manfaatnya kecil (kemungkinan <5%); di n=96 kernel lebih dominan lagi. Tetap dikerjakan karena murah dan dibutuhkan untuk overlap async dengan CPU (5b), tapi **ukur dulu dengan `cudaEvent`/`nsys`** sebelum menjanjikan hasil.
7. **Aturan split:** bagi **baris A** (bukan indeks pasangan datar): CPU mengerjakan `A[0:k)`, GPU `A[k:|A|)`, keduanya terhadap seluruh `B` dan `sorted_C`. Kapasitas `max_cap` dibagi/dijumlahkan secara eksplisit (kernel saat ini memotong lewat `atomicAdd` tanpa urutan deterministik). Pada 2 thread logis (1 core fisik), satu thread sebaiknya jadi "orkestrator" (launch, sync, copy hasil) dan yang lain mengerjakan porsi CPU — jangan buat `omp parallel` dengan 2 thread kerja sambil mengharapkan GPU tidak terganggu.
8. **Catatan kernel:** `level1_merge_kernel` belum punya quick-reject inkompatibilitas seperti versi CPU (`(u.pos & v.pos) & (u.neg | v.neg)`), dan `grid_size` di-cast `int` (aman untuk ukuran sekarang, tapi beri `assert`/guard). Binary search pada struct 40 B tidak ramah cache di GPU — lihat SoA di §7 (dimajukan).

9. **Kalibrasi rasio split — keputusan (bukan lagi terbuka): adaptif proporsional per merge**, tanpa benchmark offline. Inisialisasi 5% baris A ke CPU. Setelah tiap `merge_level1`, ukur waktu dinding sisi CPU dan sisi GPU, hitung throughput (baris/detik) masing-masing, perbarui rasio ke proporsi throughput dengan EMA (α≈0,5), clamp ke [1%, 50%]. Target: kedua sisi selesai bersamaan. Log rasio tiap run (verbose). Jika `|A|` kecil (< ~4096) → GPU saja tanpa split.
10. **Lookup pasangan lebih cepat dari binary search (usul tambahan → langkah 4c).** `sorted_C` diurut menurut `psum & mask_m1` yang kira-kira seragam di `[0, 2^b1)`. Bangun *bucket table*: `start[bucket]`, `bucket = key >> (b1 − bits)`, `bits ≈ ceil(log2 |C|)`. Pencarian per pasangan = baca `start[bucket]` dan `start[bucket+1]`, lalu binary search **di dalam bucket** (rata-rata ~1–2 elemen) — menggantikan ~15 langkah dependent-load. Tetap benar untuk distribusi tidak seragam (hanya lebih lambat). Padukan dengan **array kunci sidecar** (`c_keys[]`, dan `psum` A/B sebagai array uint64 terpisah agar load per-thread coalesced): n=64 → 27k×8 B ≈ 216 KB muat L2 CPU 256 KiB, sedangkan `TerEntry` 40 B ≈ 1,1 MB tidak. Entri AoS hanya disentuh saat ada kecocokan. Berlaku untuk CPU dan GPU, dan menjawab keluhan akses tersebar di kernel; hasil harus identik dengan versi binary search (tes diferensial).
11. **Root merge:** `merge_root_and_solve` CPU-only. Di n=64 perkiraan L1 hanya ratusan entri sehingga murah; kalau profil 4b menunjukkan L1 besar, kernel yang sama (match exact + verifikasi) dapat dipakai di GPU. Jangan dikerjakan sebelum ada data.

### 5.2 SS solver (`shroeppel_shamir_1d`) — DIGANTI: enumerasi per rentang skor di GPU

**Diagnosis:** `generate_subsets` hanya O(2^(n/4)) dan memory-bound; bottleneck sebenarnya adalah traversal dua heap (~O(2^(n/2)) langkah, tiap langkah bergantung pada langkah sebelumnya, cache-miss). Jadi **Kandidat 1 lama (doubling di GPU) dan Kandidat 2 lama (port heap/matching 4-list) dibatalkan** — yang pertama tidak berdampak, yang kedua tidak bisa diparalelkan mentah-mentah. `cuda_kernels.cu/.cuh` tetap tidak bisa dipakai (terikat `MarkShareFeas`).

**Desain baru** (mempertahankan loop `k`, filter kardinalitas, dan `S2` terurut naik yang sudah ada; `x = a+b` dengan `a∈S1, b∈S2`; `y = c+d` dengan `c∈S3, d∈S4`; cari `x + y = target`):

Bagi rentang `x` menjadi potongan `[lo, hi)`. Sisi kedua otomatis `y ∈ (target−hi, target−lo]`. Per potongan:
1. **count** — satu thread per `a∈S1`: jumlah `b` dengan `a+b ∈ [lo,hi)` = dua binary search di `S2` (hati-hati underflow saat `lo < a`; clamp). Sisi kedua sama untuk `c∈S3` terhadap `S4`.
2. **scan** — exclusive prefix sum (`cub::DeviceScan`) → offset tulis dan total pasangan `Px`, `Py`.
3. **expand** — tiap thread menulis pasangan `(kunci u64, payload = indeks (a,b) dipadatkan)` ke array; sisi `y` ditulis sebagai kunci `target − y` supaya dapat dicocokkan naik-naik.
4. **sort** — `cub::DeviceRadixSort::SortPairs` untuk kedua array, lalu match (binary search atau merge) → daftar kandidat `(a,b,c,d)`.
5. **verify** — kandidat direkonstruksi jadi bitmask dan dijumlah ulang pada presisi penuh (pola `print_and_write_1d_ss_solution`); solusi pertama yang lolos menghentikan semuanya.

**Pemilihan potongan:** jumlah pasangan per rentang tidak merata, jadi jangan pakai lebar tetap. Jalankan kernel `count` pada grid rentang kasar dulu (murah: |S1|·log|S2| per batas), lalu gabungkan rentang berdekatan sampai batas memori per potongan (target awal ≤ ~2^26–2^27 pasangan per sisi; 16 B/pasangan, ×2 dengan double-buffer sort, ×2 sisi — harus muat T4 15 GB). Ini juga alasan MITM penuh tidak dipakai: 2^32 entri sekaligus ≈ 34+ GB.

**Total kerja** kira-kira sama dengan traversal sekarang (~2·2^(n/2) pasangan), tapi massal-paralel dan sequential-access. **Perkiraan kasar (BELUM diukur):** CPU ≈ 8,6 miliar langkah heap untuk n=64 (menit sampai belasan menit); radix sort 64-bit di T4 ≈ 0,3–1 miliar pasang/detik → percepatan orde puluhan kali. Wajib dibenchmark di Colab sebelum dijadikan target.

**Batasan & catatan:**
- **Fase awal hanya jalur `uint64_t`** (dan hanya benar untuk instance yang lolos cek Bug 2 §4.3). Jalur `u128` tetap CPU-only; versi GPU u128 (filter `lo64` + perbandingan batas 128-bit hi/lo) adalah pekerjaan terpisah.
- Sebaran waktu-sampai-solusi kurang lebih setara traversal ascending sekarang (potongan diproses berurutan, berhenti di solusi pertama).
- **KEPUTUSAN TERBUKA (perlu persetujuan):** peran CPU di sini kecil — orkestrasi + verifikasi. Membagi sebagian potongan ke heap CPU secara teknis mungkin tapi kontribusinya ≈1% (1 core fisik) dan tidak sepadan dengan kompleksitasnya. Ini menyimpang tipis dari tujuan #2 ("CPU dan GPU selalu bersamaan"); default saya: CPU = orkestrator/verifier untuk SS, hybrid penuh hanya di TER.
- Scope besar (3–4 kernel + `cub`), hanya bisa diuji di Colab. Dipecah jadi sub-langkah di §11 (6a–6f).

### 5.3 CLI baru

- **Hapus** `program.add_argument("--gpu")` sepenuhnya dan `TerParams::use_gpu`. Kode diasumsikan **selalu** berjalan di hardware §2 (Xeon Broadwell + Tesla T4). Tidak ada `#ifdef WITH_GPU`, tidak ada jalur CPU-only, tidak ada build tanpa CUDA. Kode GPU jadi bagian wajib dari build.
- **Flag ikut dihapus** (hanya melayani pipeline `MarkShareFeas` lama): `-m/--m`, `-n/--n`, `-k/--k`, `-i/--iter`, `-s/--seed`, `--reduce`.
- **Flag yang DIPERTAHANKAN:** `-f/--file` (jadi **wajib**), `-t/--threads`, `--check_only`, `--write_prb`, `--k_radius`, `--runs`, `--autorestart`, `--timeout`, `--solver`, **`--max_pairs`, `--mem_budget_gb`** (keduanya dipakai `extract_pairs_from_heap` lewat `max_pairs_per_chunk` — jalur SS 1D hidup).
- **Flag baru:** `--extsol` (belum ada di `main.cpp` sekarang — harus dibuat, lihat §9).
- Jika `-f` tidak diberikan atau formatnya bukan 1D subset-sum: cetak error jelas dan keluar (pengganti fallback `MarkShareFeas(path)`).
- Setelah `argparse.hpp` tersedia: cek help text lain yang menyinggung `--gpu`/GPU/CPU-only dan sesuaikan.

---

## 6. Dead code removal (diverifikasi dengan grep pemakaian; konfirmasi akhir lewat compile)

**Genuinely dead** (hanya reachable dari cabang `else` di `main()` atau fallback `MarkShareFeas(path)`):
- `shroeppel_shamir_dim_reduced`, `compute_k_setup`, struct `KSetupResult`
- Checkpoint: `checkpoint_path_for_instance`, `load_checkpoint_completed_k`, `append_checkpoint_completed_k`, `clear_checkpoint`
- `evaluate_cpu`, `evaluate_gpu`, `evaluate_gpu_or_cpu`, struct `PipelineBuffer`, struct `EvalResult`
- `print_and_verify_solution`, `verify_solution`
- `flatten_and_encode_tuples_cpu_kv`, struct `HashIdx`, `custom_hash_cpu`, `compute_scores_cpu`, `combine_scores_cpu`
- **`find_matching_pairs_cpu`** — terkonfirmasi dead: satu-satunya pemanggil adalah `evaluate_cpu`, parameternya `HashIdx` (tidak ada overload untuk SS 1D).
- Seluruh `cuda_kernels.cu` + `cuda_kernels.cuh` (class `GpuData`, `combine_and_encode_tuples_*_gpu`, `find_matching_pairs_gpu`, `find_equal_hashes`, `find_hash_positions_gpu`, `sort_required_gpu`)
- Cabang `else` di `main()` + `instance.write_as_prb(...)` + `shroeppel_shamir_dim_reduced<uint64_t>(...)`
- **Tambahan hasil grep (belum ada di rencana lama):** `print_subset_and_compute_sum`, `extract_subset`, `concat_vectors`, `print_four_list_solution`, `write_four_list_solution_to_file`, `append_solution_to_file`, `max_encodable_dimension`, `highestSetBit`, `countSetBits`, `print_bits`, `funcTime`, `print_vector`, `compute_peak_k` (hanya definisi, tak pernah dipanggil), serta **`unpack_mask` dan `unpack_mask_into`** (pemanggilnya hanya `verify_solution`, `print_and_verify_solution`, `compute_scores_cpu` — semuanya dead).
- Include/macro yang ikut ditinjau: `<nmmintrin.h>`/`<emmintrin.h>` (→ `<immintrin.h>`, §7), `<execution>` + `PSORT`/`USE_PARALLEL_SORT` (cek apakah masih ada pemakai setelah `find_matching_pairs_cpu` hilang), `<future>` dan sejenisnya.

**BUKAN dead — jangan dihapus:**
- Dipanggil langsung oleh `shroeppel_shamir_1d`: `generate_subsets`, `filter_by_cardinality`, `sort_indices`, `apply_permutation`, `compute_k_window`, `compute_k_priority_profile`, `quarter_k_bounds`, struct `HeapNode`, `extract_pairs_from_heap` (hanya satu definisi, tidak ada duplikasi), `print_and_write_1d_ss_solution`, struct `PairsTuple`.
- **`max_pairs_per_chunk`, `detect_system_ram_bytes`, `compute_default_max_pairs`, `BYTES_PER_PAIR_HOST_ESTIMATE`** — dipakai `extract_pairs_from_heap` (`assert`/batas chunk) dan penghitungan default `--max_pairs`.
- `number_to_string`, `u128_to_string`/`operator<<`, `parse_u128_*`, `load_hgj_txt_instance`, `load_prb_1d128_instance`, `write_1d_prb`, `can_fit_size_t`, `SubsetSum1D128`, `get_filename_without_extension`, `ScopedProfiler`.
- SSE di `generate_subsets` juga hidup → diganti AVX2 (§7), bukan dihapus.

**Perlu dirapikan sebelum bisa menghapus `cuda_kernels.cuh`:** `print_info_line` hidup (dipanggil SS 1D) tetapi parameternya `const GpuData&` + `bool run_on_gpu` di bawah `#ifdef WITH_GPU`, dan `shroeppel_shamir_1d` memanggilnya dengan `GpuData{}`. Lepas keduanya dan hapus cabang cetak "GB allocated".

**`class MarkShareFeas` / `markshare.hpp` — keputusan default: dihapus total.** Konstruktor `MarkShareFeas(path)` **masih reachable** dari `main()` (file `-f` yang bukan 1D jatuh ke `instance = MarkShareFeas(path)` lalu ke `dim_reduced`), jadi konsekuensinya: dukungan input matriks m×n hilang. Ini perlu tetap dicatat sebagai keputusan eksplisit (§10). `pairs_tuple.hpp` dan `profiler.hpp` tetap.

---

## 7. Optimasi CPU (Xeon Broadwell)

- Compiler flags: `-O3 -march=broadwell -mtune=broadwell -funroll-loops -fopenmp`
- **SSE → AVX2** di `generate_subsets<uint64_t>` (`main.cpp`, cabang `if constexpr (std::is_same_v<T, uint64_t>)`). **Catatan dampak:** doubling ini O(2^(n/4)) dan memory-bound, jadi percepatan total kecil (traversal heap mendominasi). Tetap dikerjakan karena murah. Butuh `-mavx2` (dari `-march=broadwell`); cabang `u128` tidak punya SIMD dan tidak diubah.
  - Ganti `_mm_set1_epi64x/_mm_loadu_si128/_mm_storeu_si128/_mm_add_epi64` (SSE2, 2×u64/instruksi) → `_mm256_set1_epi64x/_mm256_loadu_si256/_mm256_storeu_si256/_mm256_add_epi64` (AVX2, 4×u64/instruksi).
  - Include: ganti `<nmmintrin.h>` + `<emmintrin.h>` → `<immintrin.h>` (satu header untuk AVX2/BMI2/FMA/ADX/POPCNT; `_mm_popcnt_u64` yang sudah dipakai di `filter_by_cardinality` tetap tersedia lewat `<immintrin.h>` selama dikompilasi dengan `-march=broadwell`/`-mpopcnt`).
- **Sidecar key array + bucket lookup (langkah 4c)** menggantikan "SoA penuh": tanpa mengubah layout `TerEntry` (tidak menyentuh seluruh kode), tambahkan `c_keys[]`, `psum` A/B sebagai array uint64, dan bucket table untuk `sorted_C` (detail §5.1 poin 10). Ini kandidat leverage CPU terbesar yang tersisa karena thread hanya 2: keys ≈ 216 KB muat L2, dan ~15 langkah binary search turun jadi ~2. SoA penuh baru dipertimbangkan bila profil masih menunjuk ke akses `TerEntry`. **Bukan opsional**; masuk milestone 4c, sebelum hybrid (langkah 5).
- Cache blocking/tiling di `merge_level1` CPU-side split: prioritas di bawah 4c. Setelah kunci muat L2, tile A/B supaya blok `TerEntry` yang disentuh saat ada kecocokan juga muat L2; ukur dulu.
- BMI2 (`_pext_u64`/`__builtin_ctzll`): **koreksi** — `unpack_mask`/`unpack_mask_into` hanya dipanggil dead code (§6) dan akan terhapus, jadi bukan contoh pemakaian yang hidup. Evaluasi ulang hot loop yang benar-benar tersisa; kemungkinan tidak ada yang layak.

---

## 8. Optimasi GPU (Tesla T4, sm_75)

- `nvcc -arch=sm_75` wajib di build flags.
- Workload-split hybrid TER `merge_level1` (§5.1) — kalibrasi rasio, async streams.
- Kernel GPU baru untuk SS solver: desain enumerasi per rentang skor di §5.2 (count → scan → expand → sort → match → verify), bukan port `generate_subsets`. Prototipe `uint64_t` dulu.
- Prasyarat hybrid TER: buffer persisten + pinned + stream (§5.1 poin 6) sebelum split.
- Sort `sorted_pool_right` / `sorted_C` / `root_sorted_C`: **keputusan default = tetap `std::sort` CPU.** Perkiraan n=64: `sorted_C` ≈ 27k entri → beberapa ms vs merge L1 ≈ ratusan ms; `sorted_pool_right` diurut sekali per solve; `root_sorted_C` ≈ ratusan entri. Pindah ke `thrust::sort`/`cub` hanya bila profil 4b menunjukkan sort > ~5–10% waktu run, atau bila L2 nanti dibangun di GPU sehingga datanya sudah ada di device.

---

## 9. Flag `--extsol`

**Status:** flag ini **belum ada** di `main.cpp` (tidak ada di argparse maupun kode lain) — jadi dibuat baru, bukan "disambungkan". File asal `zero_sum_swap_berlapis.h` tidak tersedia di upload terakhir; default yang saya asumsikan untuk explorer native (bisa dikoreksi): `max_swap_size=4`, ukuran in/out boleh berbeda selama jumlahnya sama, `max_tiers` dan `max_solutions` sebagai parameter dengan batas default konservatif, dedup via `std::set<std::vector<size_t>>`.

Tidak lagi bergantung pada file eksternal. Setelah implementasi native (§4.2) selesai:
1. Buat file baru `zero_sum_swap.h` (atau ditanam langsung di `main.cpp` kalau ukurannya kecil — keputusan saat implementasi) berisi explorer BFS bertingkat native, self-contained, tanpa `#include "solver_header.h"`.
2. `main.cpp` include file itu, panggil fungsi explore (nama disesuaikan, mis. `explore_zero_sum_swaps(weights, target, solution_indices, ...)`) otomatis setelah `ter_solve()`/`shroeppel_shamir_1d()` mengembalikan `found == true`, hanya kalau flag `--extsol` diaktifkan.
3. Output solusi tambahan ditulis ke file `.sol` terpisah atau append ke `.sol` yang sudah ada (format sama seperti solusi utama: string biner `0101...`).

---

## 10. Setup build & keputusan/blocker

**Argparse (Colab):**
```bash
!git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse
```
`CMakeLists.txt` sudah menambahkan `external/argparse` dan include dirnya; `main.cpp` memakai `#include "argparse.hpp"`. API dikenal (`add_argument().help().default_value().store_into()/.flag()`, `program["--flag"] == true`, `program.is_used("--x")`).

**Blocker lama — sudah terjawab:**
1. `find_matching_pairs_cpu` dead, `extract_pairs_from_heap` satu definisi hidup (§6). ✔
2. `MarkShareFeas(path)` masih reachable lewat fallback `-f` non-1D (§6). ✔ (jadi keputusan di bawah)

**Keputusan yang perlu dikonfirmasi (default saya tertulis):**
- **K1 — input matriks m×n:** dibuang (default). Konsekuensi: `-f` wajib 1D; `markshare.hpp` dihapus.
- **K2 — peran CPU di SS GPU:** orkestrator + verifier saja (default), hybrid penuh hanya TER (§5.2).
- **K3 — default explorer `--extsol`:** lihat §9; koreksi kalau file aslinya beda.
- **K4 — kalibrasi split TER:** adaptif proporsional per merge (§5.1 poin 9); tanpa benchmark offline.
- **K5 — sort ke GPU:** default tetap CPU sampai profil 4b menyatakan lain (§8).
- **K6 — `eps*`/`w1`/`w2`:** diputuskan setelah langkah 4b (hapus vs implementasi paper-faithful), §4.4.
- Desain final `SwapElement`/`SwapWitness` (§4.2) dicocokkan dengan `TerResult::solution_indices` (`std::vector<size_t>`) — sudah cocok.

**Lingkungan verifikasi:** sandbox Claude hanya punya g++ 13, 1 core, **tanpa nvcc/GPU/cmake** → yang bisa diverifikasi di sana: build & tes CPU-only (TER/SS/zero-sum-swap). Semua yang menyentuh `.cu`, kalibrasi rasio, dan benchmark T4 diuji di Colab oleh pengguna.

**CMake baru (dikerjakan bersama langkah 3):** `project(... LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit REQUIRED)`, `CUDA_ARCHITECTURES 75`, `-march=broadwell -mtune=broadwell -funroll-loops` (tanpa `-march=native`), hapus opsi `USE_CUDA`/`ENABLE_NATIVE_OPT`/`WITH_GPU`, hapus `src/cuda_kernels.cu`. Setelah tinggal satu `.cu`, evaluasi apakah `CUDA_SEPARABLE_COMPILATION` masih perlu dan apakah LTO (yang dimatikan karena tabrakan `fatbinData` multi-`.cu`) bisa dinyalakan lagi — uji di Colab.

---

## 11. Urutan pengerjaan (milestone) — satu langkah = satu titik aman

| # | Isi | Diuji di sandbox? | Status |
|---|---|---|---|
| 0 | Setup `external/argparse` (git clone) + simpan baseline output versi asli | ya | belum |
| 1 | Bug 1 (`psum` → `uint64_t`, §4.1) + Bug 2 (cek overflow SS, §4.3) | ya | belum |
| 1b | Audit exactness SS + tes diferensial vs brute force; pesan output CLI dibedakan (§4.4) | ya | belum |
| 2 | Dead code removal (§6, termasuk lepas `GpuData` dari `print_info_line`), hapus `--gpu` & flag mati (§5.3), SSE→AVX2 (§7) | ya (build CPU-only, bandingkan output) | belum |
| 3 | CMake baru (§10), CUDA wajib, `sm_75` | sebagian (perlu Colab) | belum |
| 4 | `zero_sum_swap.h` native (§4.2) + flag `--extsol` (§9) | ya | belum |
| 4b | Profiling per fase TER (gen pool / L2 / sort / L1 / root) + success rate per run pada planted instance n=32–64; putuskan K6 (`eps*`) dan tuning `l1..r2` | CPU di sandbox; GPU di Colab | belum |
| 4c | Sidecar key array + bucket lookup (§5.1 poin 10): CPU dulu, hasil identik dengan binary search; lalu dipakai kernel GPU | CPU ya; kernel di Colab | belum |
| 5 | Hybrid TER: **5a** buffer persisten+pinned+stream (ukur dulu); **5b** split baris A adaptif (K4); **5c** log & uji rasio; **5d** (opsional, sesuai profil) pipelining lintas-level / root di GPU / offload `merge_level2` | hanya Colab | belum |
| 6 | GPU SS per rentang skor (§5.2): **6a** kernel `count` + tes acuan CPU; **6b** `scan`+`expand`; **6c** `sort`+`match`+`verify`; **6d** perencana potongan; **6e** integrasi dengan loop `k`; **6f** (opsional) jalur u128 | hanya Colab | belum |
| 7 | Cache blocking + tuning lanjutan berdasarkan profil | hanya Colab | belum |

Tiap baris = satu sesi kerja (checklist rinci ada di `PROGRESS.md`); sub-langkah 5a–5d dan 6a–6f dipisah supaya kode selalu tetap bisa dikompilasi di antara sesi.

---

## 12. Workflow kerja (hemat token, tahan putus di tengah)

Konteks: dipakai di paket gratis; percakapan bisa terputus kapan saja karena batas token.

1. **Satu langkah = satu chat baru.** Riwayat panjang dibaca ulang di setiap balasan, jadi chat pendek per langkah lebih hemat dan lebih andal.
2. **Edit terarah, bukan tulis ulang file.** `main.cpp` 2.300+ baris; perubahan dilakukan lewat edit lokal di sandbox lalu file lengkap hasilnya dikirim sebagai download. Jangan minta "satu file dengan semua optimasi" — kalau terputus di tengah, hasilnya terpotong dan tak bisa dipakai.
3. **Akhir tiap langkah:** verifikasi yang bisa dilakukan di sandbox → serahkan file yang berubah → perbarui `PROGRESS.md`.
4. **`PROGRESS.md` = memori antar-chat.** Isi minimal: langkah selesai/berikutnya (mengacu nomor §11), keputusan yang sudah diambil (K1–K3), daftar file + versi terbaru, perintah tes + output yang diharapkan, masalah yang belum beres. Di awal langkah, status ditandai "sedang dikerjakan" supaya jelas kalau terputus.
5. **Prompt lanjut di chat baru** (upload `rencana.md`, `PROGRESS.md`, dan file sumber terbaru):
   ```
   Baca PROGRESS.md dan rencana.md. Lanjutkan langkah berikutnya sesuai PROGRESS.md.
   Jangan ulangi analisis yang sudah ada. Di akhir, kirim file hasil dan PROGRESS.md yang diperbarui.
   ```
6. **Git di Colab:** `git init` sekali, `git commit` tiap selesai satu langkah → bisa kembali ke titik aman tanpa bergantung pada chat.
7. **Catatan:** file di sandbox Claude hilang antar-chat, jadi file terbaru harus diunduh dan diunggah ulang di chat berikutnya.
