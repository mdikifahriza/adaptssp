cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 75)

project(adaptssp LANGUAGES CXX CUDA)

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

add_executable(adaptssp
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/ter_solver.cpp
    ${SRC_DIR}/ter_kernel.cu
    ${SRC_DIR}/ss_kernel.cu
)

target_include_directories(adaptssp PRIVATE
    ${SRC_DIR}
    ${ARGPARSE_INCLUDE_DIR}
)

target_compile_definitions(adaptssp PRIVATE WITH_GPU)

target_compile_options(adaptssp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=broadwell>
    $<$<COMPILE_LANGUAGE:CXX>:-mtune=broadwell>
    $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=broadwell,-mtune=broadwell,-funroll-loops>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(adaptssp PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION OFF
)

option(ENABLE_LTO "Aktifkan LTO untuk unit C++ (uji dulu di Colab)" OFF)
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg LANGUAGES CXX)
    if(ipo_ok)
        set_property(TARGET adaptssp PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO aktif")
    else()
        message(WARNING "LTO tidak didukung: ${ipo_msg}")
    endif()
endif()

target_link_libraries(adaptssp PRIVATE
    CUDA::cudart
    OpenMP::OpenMP_CXX
    Threads::Threads
)
