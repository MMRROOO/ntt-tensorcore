// ============================================================================
// ntt_arbitrary_radix64.cuh - Scalar CUDA Radix-64 NTT (no tensor cores)
// ============================================================================
// This implements the same radix-64 algorithm as the MMA version but uses
// regular scalar Montgomery multiplications instead of FP64 tensor cores.
//
// Algorithm: 4-step radix-64 NTT
//   Step 1: Column-wise DFT-8 (8 parallel 8-point NTTs on columns)
//   Step 2: Hadamard multiply with twiddle factors ω_64^(i*j)
//   Step 3: Row-wise DFT-8 (8 parallel 8-point NTTs on rows)
//
// This serves as a comparison baseline to measure the actual benefit of
// tensor core acceleration vs the algorithmic benefit of radix-64 fusion.
// ============================================================================
#pragma once

#include "ntt_arbitrary.cuh"

namespace ntt_arb {

// ============================================================================
// Scalar DFT-8 using Cooley-Tukey radix-2
// ============================================================================
// Computes 8-point DFT in-place using 3 stages of radix-2 butterflies.
// Input/output in natural order (no bit-reversal needed for small N=8).
// ============================================================================
template <int K>
__device__ __forceinline__
void scalar_dft8_inplace(DScalar<K>* x, const DScalar<K>* omega8_powers,
                         const DScalar<K>& prime, uint32_t np) {
    // omega8_powers[k] = ω_8^k for k = 0..7
    // Stage 0: butterflies with stride 1, groups of 2
    // Stage 1: butterflies with stride 2, groups of 4  
    // Stage 2: butterflies with stride 4, groups of 8

    // Stage 0: m=2, half_m=1
    for (int g = 0; g < 8; g += 2) {
        DScalar<K> u = x[g];
        DScalar<K> v = x[g + 1];
        // ω_8^0 = 1 for all butterflies in stage 0
        x[g]     = mod_add_d<K>(u, v, prime);
        x[g + 1] = mod_sub_d<K>(u, v, prime);
    }

    // Stage 1: m=4, half_m=2
    for (int g = 0; g < 8; g += 4) {
        for (int k = 0; k < 2; k++) {
            DScalar<K> u = x[g + k];
            DScalar<K> v = x[g + k + 2];
            // twiddle = ω_8^(k * 8/4) = ω_8^(k*2)
            DScalar<K> w = omega8_powers[k * 2];
            DScalar<K> t = mont_mul<K>(v, w, prime, np);
            x[g + k]     = mod_add_d<K>(u, t, prime);
            x[g + k + 2] = mod_sub_d<K>(u, t, prime);
        }
    }

    // Stage 2: m=8, half_m=4
    for (int k = 0; k < 4; k++) {
        DScalar<K> u = x[k];
        DScalar<K> v = x[k + 4];
        // twiddle = ω_8^(k * 8/8) = ω_8^k
        DScalar<K> w = omega8_powers[k];
        DScalar<K> t = mont_mul<K>(v, w, prime, np);
        x[k]     = mod_add_d<K>(u, t, prime);
        x[k + 4] = mod_sub_d<K>(u, t, prime);
    }

    // Bit-reversal permutation for N=8: {0,4,2,6,1,5,3,7}
    // Swap pairs: (1,4), (3,6)
    DScalar<K> tmp;
    tmp = x[1]; x[1] = x[4]; x[4] = tmp;
    tmp = x[3]; x[3] = x[6]; x[6] = tmp;
}

// ============================================================================
// Scalar 4-Step Radix-64 NTT Kernel
// ============================================================================
// Each warp processes one 64-point NTT using scalar Montgomery arithmetic.
// Data layout: 8x8 matrix stored row-major in shared memory.
// ============================================================================
template <int K>
__global__ __launch_bounds__(TCU_BLOCK_SIZE)
void scalar_radix64_kernel(
    DScalar<K>* __restrict__ data,
    const DScalar<K>* __restrict__ twiddles,  // Full twiddle table (for outer stages)
    const DScalar<K>* __restrict__ omega8,    // ω_8^k for k=0..7
    const DScalar<K>* __restrict__ omega64,   // ω_64^k for k=0..63
    uint32_t n,
    int stage_start,
    DScalar<K> prime,
    uint32_t np
) {
    extern __shared__ char smem_raw[];
    
    // Shared memory layout: omega8[8] + omega64[64] + warp_scratch[warps * 72]
    DScalar<K>* omega8_smem  = (DScalar<K>*)smem_raw;
    DScalar<K>* omega64_smem = omega8_smem + 8;
    DScalar<K>* warp_scratch_base = omega64_smem + 64;

    int tid      = threadIdx.x;
    int warp_id  = tid / TCU_WARP_SIZE;
    int lane     = tid % TCU_WARP_SIZE;
    int warps_per_block = blockDim.x / TCU_WARP_SIZE;
    int global_warp = blockIdx.x * warps_per_block + warp_id;

    // Load omega tables into SMEM (cooperative load)
    if (tid < 8) {
        omega8_smem[tid] = omega8[tid];
    }
    if (tid < 64) {
        omega64_smem[tid] = omega64[tid];
    }
    __syncthreads();

    uint32_t total_ntts = n / TCU_INNER_SIZE;
    if ((uint32_t)global_warp >= total_ntts) return;

    DScalar<K>* warp_smem = warp_scratch_base + (uint64_t)warp_id * TCU_WARP_TOTAL;

    if (n == TCU_INNER_SIZE) {
        // N == 64: Direct 4-step NTT (natural order input → natural order output)
        
        // Load 64 elements into SMEM (each lane loads 2 elements)
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            warp_smem[wsm_idx(j)] = data[j];
        }
        __syncwarp();

        // =====================================================================
        // Step 1: Column DFT-8 (8 columns, each is an 8-point NTT)
        // M[i,j] stored at warp_smem[i*8 + j], so column j has elements at
        // positions j, j+8, j+16, ..., j+56
        // =====================================================================
        // Each lane handles one column (lanes 0-7 handle columns 0-7)
        // Lanes 8-31 help by handling columns in parallel batches
        if (lane < 8) {
            int col = lane;
            DScalar<K> col_data[8];
            
            // Load column
            #pragma unroll
            for (int row = 0; row < 8; row++) {
                col_data[row] = warp_smem[wsm_idx(row * 8 + col)];
            }
            
            // Compute DFT-8 on column
            scalar_dft8_inplace<K>(col_data, omega8_smem, prime, np);
            
            // Store column back
            #pragma unroll
            for (int row = 0; row < 8; row++) {
                warp_smem[wsm_idx(row * 8 + col)] = col_data[row];
            }
        }
        __syncwarp();

        // =====================================================================
        // Step 2: Hadamard multiply with twiddles ω_64^(i*j)
        // =====================================================================
        // Each lane handles 2 elements
        #pragma unroll
        for (int idx = lane; idx < 64; idx += TCU_WARP_SIZE) {
            int row = idx / 8;
            int col = idx % 8;
            int twiddle_idx = (row * col) & 63;  // (i * j) mod 64
            DScalar<K> val = warp_smem[wsm_idx(idx)];
            DScalar<K> tw  = omega64_smem[twiddle_idx];
            warp_smem[wsm_idx(idx)] = mont_mul<K>(val, tw, prime, np);
        }
        __syncwarp();

        // =====================================================================
        // Step 3: Row DFT-8 (8 rows, each is an 8-point NTT)
        // =====================================================================
        if (lane < 8) {
            int row = lane;
            DScalar<K> row_data[8];
            
            // Load row
            #pragma unroll
            for (int col = 0; col < 8; col++) {
                row_data[col] = warp_smem[wsm_idx(row * 8 + col)];
            }
            
            // Compute DFT-8 on row
            scalar_dft8_inplace<K>(row_data, omega8_smem, prime, np);
            
            // Store row back
            #pragma unroll
            for (int col = 0; col < 8; col++) {
                warp_smem[wsm_idx(row * 8 + col)] = row_data[col];
            }
        }
        __syncwarp();

        // Store 64 elements back to global memory
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            data[j] = warp_smem[wsm_idx(j)];
        }

    } else {
        // N > 64: Fused inner stages with strided access
        uint32_t S          = 1u << stage_start;
        uint32_t macro_size = TCU_INNER_SIZE * S;
        uint32_t macro_g    = global_warp / S;
        uint32_t offset_o   = global_warp % S;
        uint32_t base       = macro_g * macro_size + offset_o;

        // Load 64 elements with stride S
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            uint32_t global_idx = base + (uint32_t)j * S;
            DScalar<K> val = data[global_idx];

            // Pre-twist for offset > 0
            if (offset_o > 0) {
                uint32_t tw_stride = n / macro_size;
                uint32_t tw_idx = ((uint64_t)offset_o * (uint64_t)bit_rev_6(j) * tw_stride) % n;
                val = mont_mul<K>(val, twiddles[tw_idx], prime, np);
            }
            warp_smem[wsm_idx(j)] = val;
        }
        __syncwarp();

        // =====================================================================
        // Radix-64 NTT using scalar operations (same 4-step structure)
        // =====================================================================
        
        // Step 1: Column DFT-8
        if (lane < 8) {
            int col = lane;
            DScalar<K> col_data[8];
            #pragma unroll
            for (int row = 0; row < 8; row++) {
                col_data[row] = warp_smem[wsm_idx(row * 8 + col)];
            }
            scalar_dft8_inplace<K>(col_data, omega8_smem, prime, np);
            #pragma unroll
            for (int row = 0; row < 8; row++) {
                warp_smem[wsm_idx(row * 8 + col)] = col_data[row];
            }
        }
        __syncwarp();

        // Step 2: Hadamard multiply
        #pragma unroll
        for (int idx = lane; idx < 64; idx += TCU_WARP_SIZE) {
            int row = idx / 8;
            int col = idx % 8;
            int twiddle_idx = (row * col) & 63;
            DScalar<K> val = warp_smem[wsm_idx(idx)];
            DScalar<K> tw  = omega64_smem[twiddle_idx];
            warp_smem[wsm_idx(idx)] = mont_mul<K>(val, tw, prime, np);
        }
        __syncwarp();

        // Step 3: Row DFT-8
        if (lane < 8) {
            int row = lane;
            DScalar<K> row_data[8];
            #pragma unroll
            for (int col = 0; col < 8; col++) {
                row_data[col] = warp_smem[wsm_idx(row * 8 + col)];
            }
            scalar_dft8_inplace<K>(row_data, omega8_smem, prime, np);
            #pragma unroll
            for (int col = 0; col < 8; col++) {
                warp_smem[wsm_idx(row * 8 + col)] = row_data[col];
            }
        }
        __syncwarp();

        // Store back with stride S
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            uint32_t global_idx = base + (uint32_t)j * S;
            data[global_idx] = warp_smem[wsm_idx(j)];
        }
    }
}

// ============================================================================
// Radix-64 NTT Buffers (extends NTTBuffers with omega8/omega64 tables)
// ============================================================================
template <int K>
struct Radix64Buffers {
    DScalar<K>* d_omega8  = nullptr;   // ω_8^k for k=0..7
    DScalar<K>* d_omega64 = nullptr;   // ω_64^k for k=0..63
};

template <int K>
static inline Radix64Buffers<K> setup_radix64_tables(const NTTBuffers<K>& B) {
    Radix64Buffers<K> R;
    
    if (B.n < 64) return R;

    BigInt<K> omega_64 = B.omega_n;
    // omega_64 = omega_n^(N/64)
    for (int t = B.n; t > 64; t >>= 1) {
        omega_64 = mod_mul<K>(omega_64, omega_64, B.prime);
    }

    // Build omega_8 = omega_64^8
    BigInt<K> omega_8 = mod_pow<K>(omega_64, 8, B.prime);

    // omega8[k] = omega_8^k
    std::vector<DScalar<K>> omega8_host(8);
    BigInt<K> w8 = BigInt<K>::one();
    for (int k = 0; k < 8; k++) {
        BigInt<K> w8_mont = mont_mul_big<K>(w8, B.R2, B.prime, B.np);
        omega8_host[k] = load_scalar<K>(w8_mont);
        w8 = mod_mul<K>(w8, omega_8, B.prime);
    }

    // omega64[k] = omega_64^k
    std::vector<DScalar<K>> omega64_host(64);
    BigInt<K> w64 = BigInt<K>::one();
    for (int k = 0; k < 64; k++) {
        BigInt<K> w64_mont = mont_mul_big<K>(w64, B.R2, B.prime, B.np);
        omega64_host[k] = load_scalar<K>(w64_mont);
        w64 = mod_mul<K>(w64, omega_64, B.prime);
    }

    cudaMalloc(&R.d_omega8,  8  * sizeof(DScalar<K>));
    cudaMalloc(&R.d_omega64, 64 * sizeof(DScalar<K>));
    cudaMemcpy(R.d_omega8,  omega8_host.data(),  8  * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    cudaMemcpy(R.d_omega64, omega64_host.data(), 64 * sizeof(DScalar<K>), cudaMemcpyHostToDevice);

    return R;
}

template <int K>
static inline void teardown_radix64_tables(Radix64Buffers<K>& R) {
    if (R.d_omega8)  cudaFree(R.d_omega8);
    if (R.d_omega64) cudaFree(R.d_omega64);
    R.d_omega8 = nullptr;
    R.d_omega64 = nullptr;
}

// ============================================================================
// Forward NTT using scalar radix-64 (no tensor cores)
// ============================================================================
template <int K>
static inline void run_forward_radix64_scalar(
    DScalar<K>* d_data,
    const DScalar<K>* d_tw,
    const DScalar<K>* d_omega8,
    const DScalar<K>* d_omega64,
    uint32_t n, int log_n,
    DScalar<K> prime_d,
    uint32_t np,
    cudaStream_t stream = 0
) {
    int t = 256;
    int half = n / 2;
    int b_st = (half + t - 1) / t;
    int b_br = (n + t - 1) / t;

    if (log_n == TCU_LOG_INNER && d_omega8 && d_omega64) {
        // Exact N=64: Pure scalar radix-64
        size_t smem_bytes = (8 + 64 + TCU_WARPS_PER_BLK * TCU_WARP_TOTAL) * sizeof(DScalar<K>);
        scalar_radix64_kernel<K><<<1, TCU_BLOCK_SIZE, smem_bytes, stream>>>(
            d_data, d_tw, d_omega8, d_omega64, n, 0, prime_d, np);
            
    } else if (log_n > TCU_LOG_INNER && d_omega8 && d_omega64) {
        // N > 64: Bit-reverse + inner radix-64 stages + outer radix-2 stages
        bitrev_kernel_d<K><<<b_br, t, 0, stream>>>(d_data, n, log_n);

        // Inner 6 stages using scalar radix-64
        size_t smem_bytes = (8 + 64 + TCU_WARPS_PER_BLK * TCU_WARP_TOTAL) * sizeof(DScalar<K>);
        uint32_t total_ntts = n / TCU_INNER_SIZE;
        uint32_t blocks = (total_ntts + TCU_WARPS_PER_BLK - 1) / TCU_WARPS_PER_BLK;

        scalar_radix64_kernel<K><<<blocks, TCU_BLOCK_SIZE, smem_bytes, stream>>>(
            d_data, d_tw, d_omega8, d_omega64, n, 0, prime_d, np);

        // Outer stages 6+ using per-element CT butterflies
        for (int st = TCU_LOG_INNER; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
    } else {
        // Fallback: standard CT-DIT radix-2
        bitrev_kernel_d<K><<<b_br, t, 0, stream>>>(d_data, n, log_n);
        for (int st = 0; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
    }
}

} // namespace ntt_arb
