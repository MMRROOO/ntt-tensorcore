#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "modular_arith.cuh"

namespace ntt {

// NTT Configuration
struct NTTConfig {
    uint64_t n;           // Transform size (power of 2)
    uint64_t log_n;       // log2(n)
    uint64_t q;           // Prime modulus
    uint64_t mu;          // Barrett constant: floor(2^64 / q). Used by the
                          // optimized kernel to replace runtime u64 %  q
                          // (slow div) with __umul64hi + 1 conditional sub.

    // Device pointers for twiddle factors
    uint64_t* d_twiddles;      // Forward NTT twiddle factors
    uint64_t* d_twiddles_inv;  // Inverse NTT twiddle factors
    
    ModularParams params;
};

// NTT Implementation variants
enum class NTTVariant {
    BASIC,           // Simple Cooley-Tukey with shared memory
    FOUR_STEP,       // 4-step algorithm with Inner/Outer split
    TLMOP,           // Thread-level memory optimization (register-based)
    FULL_OPTIMIZED   // All optimizations (TLMOP + TransOP + TFOP + RowMaj)
};

// Initialize NTT configuration
NTTConfig* ntt_init(uint64_t n, uint64_t q = DEFAULT_PRIME);

// Cleanup
void ntt_cleanup(NTTConfig* config);

// Forward NTT: polynomial to point-value representation
// Basic implementation using Cooley-Tukey
void ntt_forward_basic(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);

// Inverse NTT: point-value to polynomial representation  
void ntt_inverse_basic(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);

// 4-Step NTT for large N (splits into Inner-NTT and Outer-NTT)
void ntt_forward_four_step(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);
void ntt_inverse_four_step(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);

// Optimized NTT with TLMOP (register-based inner NTT)
void ntt_forward_tlmop(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);
void ntt_inverse_tlmop(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);

// Fully optimized NTT (all paper optimizations)
void ntt_forward_optimized(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);
void ntt_inverse_optimized(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);

// 4-step RowMaj NTT (paper § IV-B1 outer-NTT optimization).
// Input/output must be in pre-transposed format. Use the helpers below to
// convert to/from canonical layout. Currently supports N = 4096 (N1=N2=64);
// other sizes fall back to ntt_forward_optimized.
void ntt_forward_rowmaj(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream = 0);
void rowmaj_to_pretransposed(uint64_t* d_dst, const uint64_t* d_src,
                             int N1, int N2, cudaStream_t stream = 0);
void rowmaj_from_pretransposed(uint64_t* d_dst, const uint64_t* d_src,
                               int N1, int N2, cudaStream_t stream = 0);

// Batched NTT for multiple polynomials (common in FHE)
void ntt_forward_batch(uint64_t* d_data, uint64_t batch_size, 
                       const NTTConfig* config, NTTVariant variant = NTTVariant::BASIC,
                       cudaStream_t stream = 0);
void ntt_inverse_batch(uint64_t* d_data, uint64_t batch_size,
                       const NTTConfig* config, NTTVariant variant = NTTVariant::BASIC, 
                       cudaStream_t stream = 0);

// Element-wise polynomial multiplication in NTT domain
void poly_multiply_ntt(uint64_t* d_result, const uint64_t* d_a, const uint64_t* d_b,
                       const NTTConfig* config, cudaStream_t stream = 0);

// Utility: bit-reverse permutation
void bit_reverse_permute(uint64_t* d_data, uint64_t n, cudaStream_t stream = 0);

// Host-side NTT for verification
void ntt_host(std::vector<uint64_t>& data, const ModularParams& params, bool inverse = false);

} // namespace ntt
