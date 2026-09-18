cmake_minimum_required(VERSION 3.16.3)

project(markshare_gpu LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add argparse library
add_subdirectory(external/argparse)

# Add an option to toggle CUDA usage
option(USE_CUDA "Enable CUDA support" ON)
option(ENABLE_NATIVE_OPT "Enable native CPU optimizations for this machine" ON)
# Find OpenMP
find_package(OpenMP REQUIRED)

if (USE_CUDA)
    enable_language(CUDA OPTIONAL)

    # Find CUDA Toolkit
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        set(CUDA_ENABLED TRUE)
        message(STATUS "CUDA support enabled.")
    else ()
        set(CUDA_ENABLED FALSE)
        message(STATUS "CUDA toolkit not found, falling back to CPU.")
    endif ()
else (USE_CUDA)
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA support disabled.")
endif (USE_CUDA)

# If CUDA is found, add definitions and link CUDA libraries
if (CUDA_ENABLED)
    add_executable (markshare_main src/main.cpp src/cuda_kernels.cu src/ter_solver.cpp src/ter_kernel.cu)

    set_target_properties(markshare_main PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;90") # Replace these with the desired compute capabilities
    set_target_properties(markshare_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(markshare_main PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall> # Warnings for CUDA
    )

    target_link_libraries(markshare_main PRIVATE CUDA::cudart)

    target_compile_definitions(markshare_main PRIVATE WITH_GPU)
else (CUDA_ENABLED)
    add_executable (markshare_main src/main.cpp src/ter_solver.cpp)
endif(CUDA_ENABLED)

target_compile_options(markshare_main PRIVATE -Wall -Wextra) # Warnings for C++

if (ENABLE_NATIVE_OPT)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(markshare_main PRIVATE
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -mtune=native -msse4.2 -mpopcnt>
        )

        # NOTE: -flto=auto is intentionally NOT enabled when CUDA is on.
        # With CUDA_SEPARABLE_COMPILATION + multiple .cu translation units,
        # nvcc device-linking embeds an internal `fatbinData` symbol per .cu
        # file. GCC's LTO then tries to merge all object files (including
        # nvcc's host-side stub objects) into one optimization unit at link
        # time, and the two `fatbinData` symbols collide ("symbol
        # `fatbinData' is already defined"), aborting the link. This is a
        # known incompatibility between GCC LTO and nvcc's device-link step
        # when there's more than one .cu file — not a bug in this project's
        # code. LTO is still fully enabled for CPU-only builds.
        if (NOT CUDA_ENABLED)
            target_compile_options(markshare_main PRIVATE $<$<CONFIG:Release>:-flto=auto>)
            target_link_options(markshare_main PRIVATE $<$<CONFIG:Release>:-flto=auto>)
        else()
            message(STATUS "LTO disabled for Release build: incompatible with CUDA device-linking (multiple .cu TUs).")
        endif()
    elseif (MSVC)
        target_compile_options(markshare_main PRIVATE
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
        )
    endif()
endif()

# target_compile_options(markshare_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

set_target_properties(markshare_main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(markshare_main PRIVATE external/argparse/include/argparse)
target_link_libraries(markshare_main PRIVATE OpenMP::OpenMP_CXX)
