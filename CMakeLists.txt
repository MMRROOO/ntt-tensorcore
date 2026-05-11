cmake_minimum_required(VERSION 3.18)
project(ntt_tensorcore CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 120)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# NTT library
add_library(ntt_lib STATIC
    src/ntt_basic.cu
    src/ntt_four_step.cu
    src/ntt_optimized.cu
)

target_compile_options(ntt_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Test executable
add_executable(test_ntt tests/test_ntt.cu)
target_link_libraries(test_ntt ntt_lib)

# Inner NTT test
add_executable(test_inner tests/test_inner_ntt.cu)
target_link_libraries(test_inner ntt_lib)

# Debug inner kernel
add_executable(debug_inner tests/debug_inner.cu)
target_link_libraries(debug_inner ntt_lib)
add_executable(debug_rowmaj tests/debug_rowmaj.cu)
target_link_libraries(debug_rowmaj ntt_lib)

# Benchmark executable
add_executable(benchmark tests/benchmark.cu)
target_link_libraries(benchmark ntt_lib)

# Enable testing
enable_testing()
add_test(NAME NTTTest COMMAND test_ntt)
