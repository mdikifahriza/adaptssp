@echo off
echo ================================================================
echo Kompilasi Zweydinger-Esser SSP Solver (C++17 + SSE4.1 + AES-NI)
echo ================================================================
g++ -O3 -march=native -msse4.1 -msse4.2 -mpopcnt -maes -std=c++17 zweydingerCli.cpp -o zweydingerSolver.exe
if %ERRORLEVEL% EQU 0 (
    echo [SUKSES] Kompilasi berhasil: zweydingerSolver.exe terbentuk!
) else (
    echo [GAGAL] Kompilasi gagal. Periksa pesan error di atas.
)
