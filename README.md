# NTT-TensorCore: GPU-Accelerated Number Theoretic Transform

A CUDA implementation of the Number Theoretic Transform (NTT) with optimizations inspired by the HMFHE paper (ISCA 2026). This implementation progressively adds optimizations to demonstrate performance improvements.

## Overview

The Number Theoretic Transform is a crucial operation in Fully Homomorphic Encryption (FHE) schemes like CKKS. This project implements multiple NTT variants:

1. **Basic**: Simple Cooley-Tukey NTT with shared memory
2. **Four-Step**: 4-step algorithm splitting into Inner-NTT and Outer-NTT
3. **TLMOP**: Thread-Level Memory Optimization using register-based computation
4. **Full Optimized**: All optimizations (TLMOP + TransOP + TFOP + RowMaj)

## Optimizations Implemented

Based on the HMFHE paper, this implementation includes:

### TLMOP (Thread-Level Memory Optimization)
- **Inner-NTT** executed entirely in registers using warp-level primitives
- Minimizes shared memory (SMEM) traffic that causes pipeline stalls
- Uses `__shfl_sync` for cross-thread data exchange instead of SMEM

### TransOP (Transpose Optimization)
- Implicit transpose through MMA operation data layout
- Avoids explicit transpose operations that require SMEM round-trips

### TFOP (Twiddle Factor Optimization)
- Pre-arranged twiddle factors for coalesced memory access
- Separate storage for different twiddle factor types:
  - `negacyclic`: For negacyclic convolution (NTT on Z[x]/(x^N+1))
  - `tf_xy`: For outer Hadamard product
  - `tf256`: For inner Hadamard product (reused via SMEM)
  - `tfm`: 8x8 twiddle factor matrix for radix-8 NTT

### RowMaj (Row-Major NTT)
- Pre-transposed data format for row-major memory access
- Enables coalesced global memory (GMEM) reads/writes
- 3/4 of GMEM accesses become fully coalesced

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

Requirements:
- CUDA Toolkit 11.0+ (for FP64 Tensor Cores)
- CMake 3.18+
- GPU with compute capability 8.0+ (A100, RTX 3090, etc.)

## Usage

### Running Tests

```bash
./test_ntt
```

Tests include:
- NTT/INTT roundtrip correctness
- Comparison with host reference implementation
- Polynomial multiplication verification

### Running Benchmarks

```bash
./benchmark
```

Benchmarks measure:
- Single NTT latency for various sizes (2^10 to 2^16)
- Batched NTT throughput (simulating FHE workloads)
- Memory bandwidth utilization

## API

```cpp
#include "ntt.cuh"

// Initialize NTT configuration
NTTConfig* config = ntt::ntt_init(65536, ntt::DEFAULT_PRIME);

// Forward NTT
ntt::ntt_forward_basic(d_data, config);     // Basic
ntt::ntt_forward_four_step(d_data, config); // 4-Step
ntt::ntt_forward_optimized(d_data, config); // Fully optimized

// Inverse NTT
ntt::ntt_inverse_basic(d_data, config);

// Batched NTT
ntt::ntt_forward_batch(d_data, batch_size, config, NTTVariant::FULL_OPTIMIZED);

// Cleanup
ntt::ntt_cleanup(config);
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| N | 2^13 - 2^16 | Transform size (power of 2) |
| q | 2013265921 | Prime modulus (15 × 2^27 + 1) |
| Word size | 32-bit | Coefficient size |

The default prime `2013265921 = 15 × 2^27 + 1` supports primitive roots of unity up to 2^27.

## Architecture

```
include/
  modular_arith.cuh  - Modular arithmetic primitives
  ntt.cuh            - NTT interface

src/
  ntt_basic.cu       - Basic Cooley-Tukey implementation
  ntt_four_step.cu   - 4-step NTT algorithm
  ntt_optimized.cu   - All optimizations (TLMOP, TransOP, TFOP, RowMaj)

tests/
  test_ntt.cu        - Correctness tests
  benchmark.cu       - Performance benchmarks
```

## Performance (Expected)

Based on the HMFHE paper, expected speedups vs basic implementation:

| Optimization | NTT Speedup |
|--------------|-------------|
| TLMOP | 1.5x |
| +TransOP | 1.6x |
| +TFOP | 2.0x |
| +RowMaj | 2.5x |
| All Combined | ~3.0x |

For end-to-end FHE workloads, the paper reports 2.6-5.7x speedup.

## References

- HMFHE: "Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration" (ISCA 2026)
- CKKS: "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)
- Cooley-Tukey FFT algorithm

## License

MIT License
