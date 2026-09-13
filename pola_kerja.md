# SOP & Pola Kerja Optimasi Solver Subset-Sum (Roadmap Menuju N=96 < 60 Detik)

Dokumen ini mendokumentasikan metodologi, disiplin kerja, dan siklus iterasi eksperimental yang digunakan dalam riset optimasi solver Subset-Sum hingga mencapai target **N=96 terselesaikan di bawah 60 detik** pada mesin lokal.

---

## 1. Visi dan Target Utama

* **Target Akhir**: Solver mampu menyelesaikan instance Subset-Sum acak hingga skala **$N = 96$** dengan waktu eksekusi **$< 60$ detik**.
* **Batasan Memori (Safety Net)**: Penggunaan memori harus tetap terkendali (**Peak RAM $< 1$ GB**, idealnya $< 100$ MB) dan **bebas dari crash / Out-of-Memory (`std::bad_alloc`)** pada laptop berkapasitas RAM 8 GB.
* **Integritas Solusi**: Setiap solusi wajib melalui verifikasi independen 100% presisi (`sum == target`, indeks unik, tanpa collision palsu).

---

## 2. Siklus Iterasi 6 Langkah (Pola Eksperimen Berulang)

Pola berikut dijalankan secara berulang pada setiap pengujian ide baru dan kenaikan ukuran instance ($N$):

```
+-------------------------------------------------------------------+
| 1. Formulasi Hipotesis Algoritma                                  |
|    (Identifikasi bottleneck komputasi / memori pada N aktif)      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 2. Implementasi Langsung pada adaptssp                            |
|    (Modifikasi langsung di adaptssp.cpp tanpa membuat varian      |
|     terpisah seperti adaptssp2)                                   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 3. Pengujian Empiris Bertahap                                     |
|    - Instance pengujian WAJIB mengikuti instruksi User            |
|    - Jalankan secara SEKUANSIAL (satu per satu, bukan sekaligus)   |
|    - WAJIB 3x pengulangan per instance untuk validitas data       |
|    - Catat 3 metrik inti: Status Solusi, Runtime (s), Peak RAM (MB)|
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 4. Evaluasi Hasil vs Baseline Sebelumnya                          |
|    (Bandingkan performa hasil 3x run: apakah makin cepat?         |
|     apakah ada regresi atau gejala lonjakan RAM / OOM?)           |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 5. Keputusan: Pertahankan atau Revert                             |
|    - JIKA LEBIH BAIK: Pertahankan modifikasi kode                 |
|    - JIKA REGRESI / MEMBURUK / OOM: Segera batalkan (revert)      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 6. Menunggu Instruksi Commit dari User                            |
|    - HANYA perubahan terbukti lebih baik yang dapat dicommit      |
|    - DILARANG commit otomatis; tunggu instruksi eksplisit User    |
|    - DILARANG melakukan git push (eksklusif dilakukan User)       |
+-------------------------------------------------------------------+
```

---

## 3. Disiplin & Aturan Operasional (Operational Rules)

1. **Implementasi Langsung Tanpa Duplikasi File**:
   * Perubahan kode dilakukan langsung pada `adaptssp.cpp` (dan build ke `adaptssp.exe`).
   * Tidak lagi membuat file varian cadangan terpisah seperti `adaptssp2.cpp` atau `adaptssp2.exe`.

2. **Pemilihan Instance Pengujian Terarah**:
   * Instance yang digunakan untuk uji coba/benchmark **wajib mengikuti instruksi eksplisit dari User**.
   * Jangan menentukan atau memilih sendiri daftar instance tanpa arahan.

3. **Kemandirian Pengujian (*Deterministic Benchmark*)**:
   * Setiap instance diuji menggunakan perintah terminal standar dengan parameter yang terkontrol (misal: `threads=0`, batas waktu `60s` atau `120s`, `--calibrate_s=0`).
   * Jangan menguji secara paralel massal; jalankan sekuensial agar pengukuran CPU time dan RAM murni merefleksikan 1 proses.

4. **Validasi Multi-Repetisi (Minimal 3x Run)**:
   * Satu kali jalan tidak cukup karena adanya faktor acak pengacakan partisi (*random permutation*). 3x run wajib dilakukan untuk mengamati stabilitas throughput, runtime, dan konsistensi sukses.

5. **Aturan Ketat Git (Commit & Push Policy)**:
   * **DILARANG AUTO-COMMIT**: AI tidak boleh melakukan `git commit` otomatis. Kapan commit dilakukan dan pesan commit harus menunggu instruksi dari User.
   * **SYARAT COMMIT**: Hanya perubahan yang **terbukti lebih baik secara konsisten setelah 3x run** yang boleh dicommit saat diinstruksikan. Jika hasil uji memburuk atau memicu OOM, perubahan wajib di-*revert*.
   * **DILARANG KERAS GIT PUSH**: AI dilarang keras menjalankan perintah `git push`. Seluruh operasi push ke remote repository menjadi hak prerogatif dan dijalankan sendiri oleh User.


