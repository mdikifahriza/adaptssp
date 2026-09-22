cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 75)

project(adapt_ssp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
else()
    set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(ARGPARSE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/argparse/include/argparse")
if(NOT EXISTS "${ARGPARSE_INCLUDE_DIR}/argparse.hpp")
    message(FATAL_ERROR
        "argparse.hpp tidak ditemukan di ${ARGPARSE_INCLUDE_DIR}\n"
        "Jalankan: git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse")
endif()

add_executable(adapt_ssp
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/ter_solver.cpp
    ${SRC_DIR}/ter_kernel.cu
    ${SRC_DIR}/ss_kernel.cu
)

target_include_directories(adapt_ssp PRIVATE
    ${SRC_DIR}
    ${ARGPARSE_INCLUDE_DIR}
)

# GPU wajib: makro ini dipakai guard #ifdef di ter_solver.cpp / ter_kernel.*
target_compile_definitions(adapt_ssp PRIVATE WITH_GPU)

# Xeon Broadwell (AVX2/BMI2/FMA, tanpa AVX-512) dan Tesla T4 (sm_75)
target_compile_options(adapt_ssp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=broadwell>
    $<$<COMPILE_LANGUAGE:CXX>:-mtune=broadwell>
    $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=broadwell,-mtune=broadwell,-funroll-loops>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Dua file .cu (ter_kernel.cu, ss_kernel.cu), tidak ada simbol __device__/__constant__
# yang dibagi lintas file, jadi separable compilation tidak diperlukan.
set_target_properties(adapt_ssp PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION OFF
)

# LTO (rencana.md §10): default OFF karena belum pernah diuji dengan nvcc. Sekarang hanya ada satu
# .cu (tabrakan fatbinData multi-.cu yang dulu mematikan LTO sudah tidak relevan). Coba di Colab:
#   cmake -S . -B build -DENABLE_LTO=ON && cmake --build build -j2
# LTO hanya mencakup unit C++ (bukan .cu); bandingkan hasil/waktu dengan build biasa.
option(ENABLE_LTO "Aktifkan LTO untuk unit C++ (uji dulu di Colab)" OFF)
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg LANGUAGES CXX)
    if(ipo_ok)
        set_property(TARGET adapt_ssp PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO aktif")
    else()
        message(WARNING "LTO tidak didukung: ${ipo_msg}")
    endif()
endif()

target_link_libraries(adapt_ssp PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    Threads::Threads
)
