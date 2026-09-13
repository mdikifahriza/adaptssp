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
| 2. Implementasi Terisolasi                                        |
|    (Buat varian terpisah misal: adaptssp2.cpp / adaptssp2.exe, tanpa        |
|     merusak kode baseline utama)                                  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 3. Pengujian Empiris Bertahap                                     |
|    - Jalankan secara SEKUANSIAL (satu per satu, bukan sekaligus)   |
|    - WAJIB 3x pengulangan per instance untuk validitas data       |
|    - Catat 3 metrik inti: Status Solusi, Runtime (s), Peak RAM (MB)|
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 4. Komparasi Head-to-Head vs Baseline                             |
|    (Bandingkan adaptssp2 vs adaptssp: apakah makin cepat? di mana titik OOM?)|
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 5. Sintesis Pragmatis & Desain Adaptif                            |
|    - Ambil bagian yang mempercepat (misal: Two-Pointer di N kecil)|
|    - Buang/batasi bagian yang memicu OOM (pertahankan Modulo M    |
|      di N besar)                                                  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| 6. Promosi ke Baseline & Pembersihan Workspace                    |
|    - Timpa adaptssp.cpp & adaptssp.exe dengan versi adaptif terbaik         |
|    - Hapus file eksperimen (adaptssp2) agar workspace selalu bersih    |
|    - Siap naik ke target skala N berikutnya                       |
+-------------------------------------------------------------------+
```

---

## 3. Disiplin & Aturan Eksekusi (Operational Rules)

1. **Disiplin Lingkup Ketat (*Strict Scope Control*)**:
   * Jangan membuka, membaca, atau memodifikasi file di luar yang diminta.
   * Tidak melakukan perubahan tanpa instruksi eksplisit.
2. **Kemandirian Pengujian (*Deterministic Benchmark*)**:
   * Setiap instance diuji menggunakan perintah terminal standar dengan parameter yang terkontrol (misal: `threads=0`, batas waktu `60s` atau `120s`, `--calibrate_s=0`).
   * Jangan menguji secara paralel massal; jalankan sekuensial agar pengukuran CPU time dan RAM murni merefleksikan 1 proses.
3. **Validasi Multi-Repetisi (Minimal 3x Run)**:
   * Satu kali jalan tidak cukup karena adanya faktor acak pengacakan partisi (*random permutation*). 3x run wajib dilakukan untuk mengamati stabilitas throughput dan peluang sukses.

