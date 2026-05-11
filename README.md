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
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..   # 80 = A100, 86 = A4000, 89 = L4/4090, 90 = H100
make -j
```

Requirements:
- CUDA Toolkit 11.0+ (for FP64 Tensor Cores)
- CMake 3.18+
- GPU with compute capability 8.0+ (A100, A40, RTX 3090, etc.)

> **Important:** the kernel uses the FP64 `mma.sync.aligned.m8n8k4` tensor-core
> instruction (introduced in Ampere, sm_80). Full-rate FP64 tensor cores only
> exist on **datacenter** Ampere/Hopper (A100, H100, A30). On consumer GPUs
> (RTX 30/40/50 series, L4) the instruction works but FP64 throughput is
> heavily throttled — you'll see correctness pass but timings will be
> dominated by FP64 emulation cost.

## Lab 6 (MIT 6.S894) optimizations applied to the optimized kernel

The optimized variant (`src/ntt_optimized.cu`) applies the techniques from
[6.S894 Lab 6](https://accelerated-computing.academy/fall24/labs/lab6/):

| Technique | Implementation |
|---|---|
| **Avoid runtime divides** | Barrett reduction with precomputed `mu = ⌊2⁶⁴/q⌋` (`mod_q_barrett`) replaces every `% q` with `__umul64hi`+sub+1-cmov |
| **Cheap-arithmetic core** | `__int128` modmuls dropped — since q < 2³¹, `a*b` always fits in uint64 |
| **Skip redundant reductions** | Bit-merge fused: only reduce `D_high` before shifting; sum with raw `D_low` (still < 2⁵⁰) and Barrett-reduce once |
| **Power-of-2 mod** | `% n` in hot path replaced with `& (n-1)` since N is always a power of 2 |
| **Avoid bank conflicts** | Per-warp SMEM scratch padded from stride 8 → 9 doubles (eliminates 2-way conflicts on stride-8 column reads) |
| **Overlap data movement w/ compute** | `cp.async.ca.shared.global` 8B copies for the per-warp input load when no pre-twist is needed; data streams into SMEM in parallel with previous-iter dependency chain |

## ICICLE comparison (apples-to-apples)

This project's BabyBear NTT (q = 15·2²⁷+1 = 2013265921) is benchmarked
head-to-head against [Ingonyama's ICICLE](https://github.com/ingonyama-zk/icicle)
which uses the **same prime**, on the **same GPU**, with **identical input
data, sizes and timing harness** (`tests/icicle_compare.cu`).

The MMA kernel and ICICLE go through one shared `time_kernel(...)` micro-bench
that records cudaEvents around each call and forces a `cudaDeviceSynchronize()`
on both sides of every iteration so async stream pools cannot mask kernel
work. Domain-init / twiddle-precompute is performed once outside the timed
region for both implementations.

### Build

```bash
./scripts/build_icicle.sh                                  # downloads the
                                                            # ICICLE 4.0.0 release
                                                            # tarballs (~16 MB +
                                                            # ~370 MB) into
                                                            # ~/.local/icicle
cd build
cmake -DICICLE_ENABLED=ON \
      -DICICLE_INSTALL_DIR=$HOME/.local/icicle \
      ..
make -j icicle_compare benchmark
```

If you already have an ICICLE install (e.g. from another project), point
`-DICICLE_INSTALL_DIR=` at it -- the CMake auto-detects either layout
(`<dir>/include` or `<dir>/icicle/include`).

### Run

```bash
ICICLE_BACKEND_INSTALL_DIR=$HOME/.local/icicle/lib/backend ./icicle_compare
# or pass log2(N) values as arguments:
ICICLE_BACKEND_INSTALL_DIR=$HOME/.local/icicle/lib/backend ./icicle_compare 16 18 20
```

### Sample results — RTX 5060 (consumer Blackwell, sm_120, 30 SMs)

```
N             MMA min   MMA med   MMA avg  ICICLE min  ICICLE med  ICICLE avg  med ratio
                 (us)      (us)      (us)        (us)        (us)        (us)  ICICLE/MMA
------------------------------------------------------------------------------------------
4096            13.60     15.01     14.98       14.59       65.38      247.26      4.36x
16384           25.47     27.17     31.46       14.98       66.50      195.09      2.45x
65536           60.83     62.43     79.72       16.64       68.67      259.83      1.10x
131072         103.39    104.29    105.77       21.47       71.14      275.00      0.68x
262144         252.19    257.47    260.19       30.21       78.53      211.06      0.30x
524288         487.71    491.14    502.51       50.24      102.66      317.99      0.21x
1048576        961.12    963.65    979.66       94.78      141.25      422.02      0.15x
```

Reading the table:

* **N ≤ 64K**: the MMA kernel beats ICICLE on this consumer Blackwell GPU,
  by up to 4.4× at 4K — the per-launch overhead and per-stage setup of
  ICICLE's mixed-radix engine doesn't amortize at small sizes.
* **N ≥ 128K**: ICICLE pulls ahead because it uses **uint32 modular
  arithmetic** that runs at full-rate on every consumer SM, while our MMA
  path is bottlenecked by the heavily throttled FP64 tensor cores
  (sm_120 consumer FP64 is ~1/64 the rate of sm_80 datacenter FP64).

The interesting test is the same comparison on a **datacenter A100/H100**:
on those GPUs FP64 MMA runs at 19.5+ TFLOPS, which is the design point
of this kernel and is expected to flip large-N results in MMA's favour.
The exact same `./icicle_compare` binary will give that data on those
machines without any code changes.

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
