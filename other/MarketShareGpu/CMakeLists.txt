cmake_minimum_required(VERSION 3.16.3)

project(markshare_gpu LANGUAGES CXX CUDA)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add argparse library
add_subdirectory(external/argparse)

# Add an option to toggle CUDA usage
option(USE_CUDA "Enable CUDA support" ON)
# Find OpenMP
find_package(OpenMP REQUIRED)

if (USE_CUDA)
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled.")
else (USE_CUDA)
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA support disabled.")
endif (USE_CUDA)

# If CUDA is found, add definitions and link CUDA libraries
if (CUDA_ENABLED)
    add_executable (markshare_main src/main.cpp src/cuda_kernels.cu)

    set_target_properties(markshare_main PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;90") # Replace these with the desired compute capabilities
    set_target_properties(markshare_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(markshare_main PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall> # Warnings for CUDA
    )

    target_link_libraries(markshare_main PRIVATE CUDA::cudart)

    target_compile_definitions(markshare_main PRIVATE WITH_GPU)
else (CUDA_ENABLED)
    add_executable (markshare_main src/main.cpp)
endif(CUDA_ENABLED)

target_compile_options(markshare_main PRIVATE -Wall -Wextra) # Warnings for C++

# target_compile_options(markshare_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

target_compile_features(markshare_main PUBLIC cxx_std_17)

target_include_directories(markshare_main PRIVATE external/argparse/include/argparse)
target_link_libraries(markshare_main PRIVATE OpenMP::OpenMP_CXX)
