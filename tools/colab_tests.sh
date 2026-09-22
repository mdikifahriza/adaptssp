#!/usr/bin/env bash
# Satu perintah untuk semua tes di Colab (runtime GPU T4).
#
#   bash tools/colab_tests.sh              # tes standar (beberapa menit)
#   QUICK=1 bash tools/colab_tests.sh      # versi cepat (run/instance lebih sedikit)
#   FULL=1  bash tools/colab_tests.sh      # + compute-sanitizer memcheck dan nsys (lambat)
#
# Jalankan dari akar proyek (yang berisi CMakeLists.txt, main.cpp, tools/).
# Keluaran juga disimpan di colab_tests.log; data profil TER di work/profil_*.txt.
# Skrip TIDAK berhenti di kegagalan pertama; ringkasan PASS/FAIL/SKIP ada di akhir.
set -u
ROOT="$(pwd)"
[ -f "$ROOT/CMakeLists.txt" ] || { echo "Jalankan dari akar proyek (CMakeLists.txt tidak ditemukan)"; exit 1; }
exec > >(tee "$ROOT/colab_tests.log") 2>&1

QUICK="${QUICK:-0}"; FULL="${FULL:-0}"
RUNS_STAT=20; DIFF_N=300; DIFF_BIG=80
if [ "$QUICK" = "1" ]; then RUNS_STAT=5; DIFF_N=60; DIFF_BIG=20; fi

PASS=0; FAIL=0; SKIP=0
ok()   { echo "[PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }
skip() { echo "[SKIP] $1"; SKIP=$((SKIP+1)); }
hdr()  { echo; echo "================ $1 ================"; }
# check "deskripsi" perintah... -> PASS bila rc=0, selain itu FAIL dan tampilkan ekor keluaran
check() {
  local d="$1"; shift
  local o; o="$("$@" 2>&1)"; local rc=$?
  if [ $rc -eq 0 ]; then ok "$d"; else bad "$d (rc=$rc)"; echo "$o" | tail -15; fi
}
now_s() { date +%s.%N; }

hdr "0. Lingkungan"
nvidia-smi -L 2>&1 | head -2
nvcc --version 2>&1 | tail -1
cmake --version 2>&1 | head -1
lscpu | grep -o -wE "avx2|bmi2" | sort -u | tr '\n' ' '; echo
python3 --version

hdr "1. Build GPU (CMake) dan CPU-only (g++)"
[ -f external/argparse/include/argparse/argparse.hpp ] || \
  git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse
SRC="$ROOT"; [ -f "$ROOT/src/main.cpp" ] && SRC="$ROOT/src"
if cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j2 2>&1 | tail -5 && [ -x build/adapt_ssp ]; then
  ok "build GPU build/adapt_ssp"
else
  bad "build GPU gagal -> tes GPU dilewati"; GPU_OK=0
fi
GPU_OK="${GPU_OK:-1}"
mkdir -p build work
if g++ -std=c++17 -O3 -march=broadwell -fopenmp -I"$SRC" -Iexternal/argparse/include/argparse \
     "$SRC/main.cpp" "$SRC/ter_solver.cpp" -o build/adapt_ssp_cpu 2>&1 | tail -5 && [ -x build/adapt_ssp_cpu ]; then
  ok "build CPU-only build/adapt_ssp_cpu"
else
  bad "build CPU-only gagal"
fi
GPU=./build/adapt_ssp; CPU=./build/adapt_ssp_cpu

hdr "2. Tes unit explorer zero-sum swap (tools/test_swap.cpp, ASan+UBSan)"
if g++ -std=c++17 -O2 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined \
     -I"$SRC" -o build/test_swap tools/test_swap.cpp 2>&1 | tail -5; [ -x build/test_swap ]; then
  check "test_swap (diferensial vs oracle BFS, error handling, cap)" ./build/test_swap
else
  bad "test_swap tidak bisa dikompilasi"
fi

hdr "3. Instance uji (tools/genplant.py)"
python3 tools/genplant.py work/p24.prb 24 20 7   | head -1
python3 tools/genplant.py work/p28.prb 28 24 3   | head -1
python3 tools/genplant.py work/big32.prb 32 63 5 | head -1
python3 tools/genplant.py work/dup20.prb 20 10 4 --dups 8 | head -1
python3 tools/genplant.py work/dup24.prb 24 10 9 --dups 10 | head -1
for f in dup20 dup24; do python3 tools/bruteforce.py work/$f.prb | head -1; done

hdr "4. TER end-to-end (GPU) + verifikasi .sol"
if [ "$GPU_OK" = "1" ]; then
  for inst in p24 p28 big32; do
    ( cd work && ../$GPU -f $inst.prb --solver ter --runs 3 > ter_$inst.out 2>&1 )
    if grep -q "SOLUTION FOUND" work/ter_$inst.out; then
      check "TER GPU $inst: .sol valid" python3 tools/verify_sol.py work/$inst.prb work/$inst.sol
    else
      bad "TER GPU $inst: tidak menemukan solusi dalam 3 run (bukan bukti infeasible)"
    fi
    grep -E "Execution mode|PERINGATAN|\[GPU\]" work/ter_$inst.out | head -3
  done
else skip "TER GPU (build GPU gagal)"; fi

hdr "5. Langkah 4c: --ter_bucket harus identik dengan binary search"
# CPU-only, 1 thread => deterministik (rng(1337)); ukuran L2/L1 tiap run harus sama persis.
core() { grep -o "L2 avg=[0-9]*, L1=([0-9, ]*)" "$1"; }
for inst in p28 big32; do
  ( cd work && OMP_NUM_THREADS=1 ../$CPU -f $inst.prb --solver ter --runs 4 --ter_stats > cmp_${inst}_bs.out 2>&1 )
  ( cd work && OMP_NUM_THREADS=1 ../$CPU -f $inst.prb --solver ter --runs 4 --ter_stats --ter_bucket > cmp_${inst}_bk.out 2>&1 )
  core work/cmp_${inst}_bs.out > work/cmp_${inst}_bs.core; core work/cmp_${inst}_bk.out > work/cmp_${inst}_bk.core
  if [ -s work/cmp_${inst}_bs.core ] && diff -q work/cmp_${inst}_bs.core work/cmp_${inst}_bk.core >/dev/null; then
    ok "CPU $inst: ukuran L2/L1 per run identik (binary search vs bucket)"
  else
    bad "CPU $inst: ukuran beda / kosong"; diff work/cmp_${inst}_bs.core work/cmp_${inst}_bk.core | head
  fi
  a=$(grep -c "SOLUTION FOUND" work/cmp_${inst}_bs.out); b=$(grep -c "SOLUTION FOUND" work/cmp_${inst}_bk.out)
  [ "$a" = "$b" ] && ok "CPU $inst: jumlah run sukses sama ($a)" || bad "CPU $inst: run sukses beda ($a vs $b)"
  echo "   waktu L1 per run (ms):"; grep "TER-PROFIL\] rata-rata per run" work/cmp_${inst}_bs.out | sed 's/^/   bs: /'; grep "TER-PROFIL\] rata-rata per run" work/cmp_${inst}_bk.out | sed 's/^/   bk: /'
done
if [ "$GPU_OK" = "1" ]; then
  for inst in p28 big32; do
    ( cd work && ../$GPU -f $inst.prb --solver ter --runs 4 --ter_stats > gcmp_${inst}_bs.out 2>&1 )
    ( cd work && ../$GPU -f $inst.prb --solver ter --runs 4 --ter_stats --ter_bucket > gcmp_${inst}_bk.out 2>&1 )
    core work/gcmp_${inst}_bs.out > work/gcmp_${inst}_bs.core; core work/gcmp_${inst}_bk.out > work/gcmp_${inst}_bk.core
    if diff -q work/gcmp_${inst}_bs.core work/gcmp_${inst}_bk.core >/dev/null; then
      ok "GPU $inst: ukuran L2/L1 identik (kernel lama vs kernel bucket)"
    elif grep -q "L1 kena cap=0/" work/gcmp_${inst}_bs.out; then
      bad "GPU $inst: ukuran beda padahal tidak ada cap"; diff work/gcmp_${inst}_bs.core work/gcmp_${inst}_bk.core | head
    else
      echo "[WARN] GPU $inst: ukuran beda dan L1 kena cap (urutan atomicAdd tidak deterministik) -> periksa manual"
    fi
    grep -E "TER-PROFIL\] (L1|GPU)" work/gcmp_${inst}_bs.out | sed 's/^/   bs: /'
    grep -E "TER-PROFIL\] (L1|GPU)" work/gcmp_${inst}_bk.out | sed 's/^/   bk: /'
  done
fi

hdr "6. --extsol end-to-end (SS dan TER) + oracle komponen"
run_ext() { # solver instance swap_size
  local solver="$1" inst="$2" m="$3" bin="$4" extra="${5:-}"
  ( cd work && rm -f $inst.sol $inst.extsol && ../$bin -f $inst.prb --solver $solver $extra --extsol --extsol_swap_size $m --extsol_max_solutions 0 > ext_${solver}_${inst}_m$m.out 2>&1 )
  if [ ! -s work/$inst.extsol ]; then skip "extsol $solver $inst m=$m: tidak ada solusi awal / berkas kosong"; return; fi
  check "extsol $solver $inst m=$m: semua baris valid+unik, baris 1 == .sol" python3 tools/verify_sol.py work/$inst.prb --extsol work/$inst.extsol
  python3 tools/bruteforce.py work/$inst.prb --component work/$inst.sol --m $m --extsol work/$inst.extsol > work/oracle_${solver}_${inst}_m$m.txt 2>&1
  rc=$?
  tail -2 work/oracle_${solver}_${inst}_m$m.txt
  case $rc in 0) ok "extsol $solver $inst m=$m == komponen oracle";; 3) skip "oracle tidak bisa memutuskan ($solver $inst m=$m)";; *) bad "extsol $solver $inst m=$m != oracle";; esac
}
BIN_EXT=$CPU; [ "$GPU_OK" = "1" ] && BIN_EXT=$GPU
for inst in dup20 dup24; do for m in 2 4; do run_ext ss $inst $m $CPU; done; done
run_ext ter dup24 3 $BIN_EXT "--runs 30"
grep -h "extsol\] \(selesai\|closure\|hasil\)" work/ext_ss_dup20_m4.out 2>/dev/null | head -3

hdr "7. Diferensial SS vs brute force (audit exactness, langkah 1b)"
BIN_D=$CPU; [ "$GPU_OK" = "1" ] && BIN_D=$GPU
check "difftest $DIFF_N instance (seed 1)" python3 tools/difftest.py $BIN_D $DIFF_N 1
check "difftest --big $DIFF_BIG instance (seed 11)" python3 tools/difftest.py $BIN_D $DIFF_BIG 11 --big

hdr "8. Langkah 4b: success rate TER per run + profil per fase (planted n=32/40/48)"
for spec in "32 24 5" "40 30 6" "48 40 7"; do
  set -- $spec; n=$1; bits=$2; sd=$3
  python3 tools/genplant.py work/pl$n.prb $n $bits $sd | head -1
  BIN_P=$CPU; [ "$GPU_OK" = "1" ] && BIN_P=$GPU
  ( cd work && timeout 1500 ../$BIN_P -f pl$n.prb --solver ter --ter_stats --runs $RUNS_STAT > profil_full_$n.txt 2>&1 )
  grep -h "TER-PROFIL" work/profil_full_$n.txt > work/profil_$n.txt
  echo "--- n=$n bits=$bits (RUNS=$RUNS_STAT)"; cat work/profil_$n.txt
done
echo "(data untuk keputusan K6 / tuning l1..r2 ada di work/profil_*.txt)"

hdr "9. Waktu GPU vs CPU-only (mesin sama, jam dinding)"
if [ "$GPU_OK" = "1" ]; then
  for inst in p28 big32; do
    for who in GPU CPU; do
      bin=$GPU; [ $who = CPU ] && bin=$CPU
      t0=$(now_s); ( cd work && ../$bin -f $inst.prb --solver ter --runs 3 > /dev/null 2>&1 ); t1=$(now_s)
      printf "  %-6s %-5s %s s\n" $inst $who "$(python3 -c "print('%.2f' % ($t1 - $t0))")"
    done
  done
  echo "  (run pertama GPU memuat inisialisasi konteks CUDA; lihat baris [TER-PROFIL] GPU untuk pisahnya)"
else skip "waktu GPU (build GPU gagal)"; fi

if [ "$FULL" = "1" ] && [ "$GPU_OK" = "1" ]; then
  hdr "10. FULL: compute-sanitizer dan nsys"
  ( cd work && compute-sanitizer --tool memcheck ../$GPU -f p24.prb --solver ter --runs 1 > sanitizer.out 2>&1 )
  grep "ERROR SUMMARY" work/sanitizer.out
  grep -q "ERROR SUMMARY: 0 errors" work/sanitizer.out && ok "compute-sanitizer memcheck 0 error" || bad "compute-sanitizer melaporkan error"
  ( cd work && compute-sanitizer --tool memcheck ../$GPU -f p24.prb --solver ter --runs 1 --ter_bucket > sanitizer_bk.out 2>&1 )
  grep "ERROR SUMMARY" work/sanitizer_bk.out
  grep -q "ERROR SUMMARY: 0 errors" work/sanitizer_bk.out && ok "compute-sanitizer memcheck 0 error (--ter_bucket)" || bad "compute-sanitizer error (--ter_bucket)"
  if command -v nsys >/dev/null; then
    ( cd work && nsys profile -t cuda --stats=true -o nsys_bs -f true ../$GPU -f p28.prb --solver ter --runs 1 > nsys_bs.txt 2>&1 )
    ( cd work && nsys profile -t cuda --stats=true -o nsys_bk -f true ../$GPU -f p28.prb --solver ter --runs 1 --ter_bucket > nsys_bk.txt 2>&1 )
    grep -q level1_merge_kernel work/nsys_bs.txt && ok "nsys: level1_merge_kernel muncul" || bad "nsys: kernel tidak muncul"
    grep -q level1_merge_kernel_bucket work/nsys_bk.txt && ok "nsys: level1_merge_kernel_bucket muncul" || bad "nsys: kernel bucket tidak muncul"
  else skip "nsys tidak terpasang"; fi
fi

hdr "RINGKASAN"
echo "PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP   (log: colab_tests.log, data: work/)"
[ $FAIL -eq 0 ]
