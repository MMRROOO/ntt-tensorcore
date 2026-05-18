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

# Generic K-limb FP64-MMA NTT (templated over K, supports arbitrary prime sizes).
# Self-contained: has its own main() with correctness + timing.
add_executable(ntt_arbitrary src/ntt_mma_arbitrary.cu)
target_compile_options(ntt_arbitrary PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Timing comparison: MMA optimized vs Radix-2 for K=2 and K=11
add_executable(timing_compare tests/timing_compare.cu)
target_compile_options(timing_compare PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ============================================================================
# Optional: ICICLE NTT comparison (https://github.com/ingonyama-zk/icicle)
# ----------------------------------------------------------------------------
# Build ICICLE separately first via:  ./scripts/build_icicle.sh
# Then enable here with:
#   cmake -DICICLE_ENABLED=ON \
#         -DICICLE_INSTALL_DIR=$HOME/.local/icicle ..
# ============================================================================
option(ICICLE_ENABLED "Enable ICICLE NTT comparison binary" OFF)
if(ICICLE_ENABLED)
    if(NOT DEFINED ICICLE_INSTALL_DIR OR NOT EXISTS "${ICICLE_INSTALL_DIR}")
        message(FATAL_ERROR
            "ICICLE_ENABLED=ON but ICICLE_INSTALL_DIR is not set or doesn't exist.\n"
            "Either point at an existing ICICLE install:\n"
            "  -DICICLE_INSTALL_DIR=$ENV{HOME}/git-repos/icicle-install/icicle\n"
            "Or build ICICLE first:  ./scripts/build_icicle.sh")
    endif()

    # Auto-detect layout. The official tarball untars to ./icicle/{include,lib};
    # but our build_icicle.sh installs flat as <dir>/{include,lib}. Support both.
    if(EXISTS "${ICICLE_INSTALL_DIR}/include/icicle/ntt.h")
        set(ICICLE_INC "${ICICLE_INSTALL_DIR}/include")
        set(ICICLE_LIB "${ICICLE_INSTALL_DIR}/lib")
    elseif(EXISTS "${ICICLE_INSTALL_DIR}/icicle/include/icicle/ntt.h")
        set(ICICLE_INC "${ICICLE_INSTALL_DIR}/icicle/include")
        set(ICICLE_LIB "${ICICLE_INSTALL_DIR}/icicle/lib")
    else()
        message(FATAL_ERROR
            "Could not find icicle/ntt.h under ${ICICLE_INSTALL_DIR}.\n"
            "Expected layout: <dir>/include/icicle/ntt.h or <dir>/icicle/include/icicle/ntt.h")
    endif()
    message(STATUS "ICICLE include : ${ICICLE_INC}")
    message(STATUS "ICICLE lib     : ${ICICLE_LIB}")
    message(STATUS "ICICLE backend : ${ICICLE_LIB}/backend  (set ICICLE_BACKEND_INSTALL_DIR at runtime)")

    find_library(ICICLE_DEVICE_LIB icicle_device
                 PATHS "${ICICLE_LIB}" NO_DEFAULT_PATH REQUIRED)
    find_library(ICICLE_FIELD_LIB  icicle_field_babybear
                 PATHS "${ICICLE_LIB}" NO_DEFAULT_PATH REQUIRED)

    add_executable(icicle_compare tests/icicle_compare.cu)
    target_include_directories(icicle_compare PRIVATE "${ICICLE_INC}")
    target_link_libraries(icicle_compare PRIVATE
        ntt_lib
        ${ICICLE_FIELD_LIB}
        ${ICICLE_DEVICE_LIB})
    # ICICLE's headers branch on FIELD_ID for the field_config alias.
    target_compile_definitions(icicle_compare PRIVATE FIELD_ID=BABY_BEAR)
    # Embed ICICLE's lib dir as RPATH so we don't need LD_LIBRARY_PATH.
    set_target_properties(icicle_compare PROPERTIES
        BUILD_RPATH   "${ICICLE_LIB}"
        INSTALL_RPATH "${ICICLE_LIB}")

    # ---- BN254 (254-bit) head-to-head: our K=11 MMA NTT vs ICICLE BN254. ----
    # ICICLE's BN254 scalar field NTT lives in libicicle_field_bn254.so; the
    # curve_bn254 lib only carries the EC-projective NTT variant.
    find_library(ICICLE_BN254_FIELD_LIB icicle_field_bn254
                 PATHS "${ICICLE_LIB}" NO_DEFAULT_PATH REQUIRED)

    add_executable(icicle_arbitrary_compare tests/icicle_arbitrary_compare.cu)
    target_include_directories(icicle_arbitrary_compare PRIVATE "${ICICLE_INC}")
    target_link_libraries(icicle_arbitrary_compare PRIVATE
        ${ICICLE_BN254_FIELD_LIB}
        ${ICICLE_DEVICE_LIB})
    target_compile_definitions(icicle_arbitrary_compare PRIVATE
        FIELD_ID=BN254
        CURVE_ID=BN254)
    target_compile_options(icicle_arbitrary_compare PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(icicle_arbitrary_compare PROPERTIES
        BUILD_RPATH   "${ICICLE_LIB}"
        INSTALL_RPATH "${ICICLE_LIB}")
endif()

# Enable testing
enable_testing()
add_test(NAME NTTTest COMMAND test_ntt)
