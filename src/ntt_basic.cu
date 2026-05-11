#include "ntt.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>

namespace ntt {

// Maximum shared memory elements (32KB / 8 bytes = 4096 elements to be safe)
constexpr int MAX_SMEM_ELEMENTS = 4096;
constexpr int WARP_SIZE = 32;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Bit-reverse an index
__device__ __forceinline__
uint32_t bit_reverse(uint32_t x, int log_n) {
    uint32_t result = 0;
    for (int i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Kernel: Bit-reverse permutation
__global__ void bit_reverse_kernel(uint64_t* data, uint64_t n, int log_n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t rev_idx = bit_reverse(idx, log_n);
    if (idx < rev_idx) {
        uint64_t tmp = data[idx];
        data[idx] = data[rev_idx];
        data[rev_idx] = tmp;
    }
}

// Basic NTT kernel - processes one stage at a time
// Each thread handles one butterfly operation
__global__ void ntt_stage_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint64_t n,
    uint64_t stage,  // Current stage (0 to log_n - 1)
    uint64_t q
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half_size = 1ULL << stage;
    uint64_t full_size = half_size << 1;
    
    // Total number of butterfly operations = n/2
    uint64_t num_butterflies = n >> 1;
    if (idx >= num_butterflies) return;
    
    // Determine which butterfly group and position within group
    uint64_t group = idx / half_size;
    uint64_t pos = idx % half_size;
    
    // Indices for butterfly
    uint64_t i = group * full_size + pos;
    uint64_t j = i + half_size;
    
    // Twiddle factor index
    uint64_t tw_idx = pos * (n / full_size);
    uint64_t w = twiddles[tw_idx];
    
    // Load values
    uint64_t u = data[i];
    uint64_t v = data[j];
    
    // Butterfly: (u, v) -> (u + w*v, u - w*v)
    uint64_t wv = (unsigned __int128)w * v % q;
    
    data[i] = (u + wv) % q;
    data[j] = (u >= wv) ? (u - wv) : (u + q - wv);
}

// Shared memory NTT kernel - processes multiple stages in one kernel
// Works for N <= MAX_SMEM_ELEMENTS
__global__ void ntt_smem_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint64_t n,
    int log_n,
    uint64_t q,
    bool inverse
) {
    extern __shared__ uint64_t smem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    
    // Load data into shared memory with bit-reversal
    for (int i = tid; i < n; i += block_size) {
        uint32_t rev_i = bit_reverse(i, log_n);
        smem[i] = data[bid * n + rev_i];
    }
    __syncthreads();
    
    // Perform NTT stages
    for (int stage = 0; stage < log_n; stage++) {
        uint64_t half_size = 1ULL << stage;
        uint64_t full_size = half_size << 1;
        uint64_t tw_stride = n / full_size;
        
        for (int idx = tid; idx < (n >> 1); idx += block_size) {
            uint64_t group = idx / half_size;
            uint64_t pos = idx % half_size;
            
            uint64_t i = group * full_size + pos;
            uint64_t j = i + half_size;
            
            uint64_t tw_idx = pos * tw_stride;
            uint64_t w = twiddles[tw_idx];
            
            uint64_t u = smem[i];
            uint64_t v = smem[j];
            
            uint64_t wv = (unsigned __int128)w * v % q;
            
            smem[i] = (u + wv) % q;
            smem[j] = (u >= wv) ? (u - wv) : (u + q - wv);
        }
        __syncthreads();
    }
    
    // Write back to global memory
    for (int i = tid; i < n; i += block_size) {
        data[bid * n + i] = smem[i];
    }
}

// Multi-stage kernel for larger NTTs
// Processes as many stages as fit in shared memory, then switches to global memory
__global__ void ntt_hybrid_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint64_t n,
    int log_n,
    int start_stage,
    int end_stage,
    uint64_t q
) {
    extern __shared__ uint64_t smem[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Each block processes a contiguous chunk
    uint64_t chunk_size = 1ULL << end_stage;
    uint64_t num_chunks = n / chunk_size;
    uint64_t chunk_id = blockIdx.x % num_chunks;
    uint64_t base_idx = chunk_id * chunk_size;
    
    // Load chunk into shared memory
    for (int i = tid; i < chunk_size; i += block_size) {
        smem[i] = data[base_idx + i];
    }
    __syncthreads();
    
    // Process stages
    for (int stage = start_stage; stage < end_stage; stage++) {
        uint64_t half_size = 1ULL << stage;
        uint64_t full_size = half_size << 1;
        uint64_t tw_stride = n / full_size;
        
        for (int idx = tid; idx < (chunk_size >> 1); idx += block_size) {
            uint64_t local_group = idx / half_size;
            uint64_t pos = idx % half_size;
            
            uint64_t i = local_group * full_size + pos;
            uint64_t j = i + half_size;
            
            // Global twiddle factor index
            uint64_t global_group = (base_idx / full_size) + local_group;
            uint64_t tw_idx = (global_group * half_size + pos) * tw_stride % n;
            tw_idx = pos * tw_stride;  // For standard NTT
            
            uint64_t w = twiddles[tw_idx];
            
            uint64_t u = smem[i];
            uint64_t v = smem[j];
            
            uint64_t wv = (unsigned __int128)w * v % q;
            
            smem[i] = (u + wv) % q;
            smem[j] = (u >= wv) ? (u - wv) : (u + q - wv);
        }
        __syncthreads();
    }
    
    // Write back
    for (int i = tid; i < chunk_size; i += block_size) {
        data[base_idx + i] = smem[i];
    }
}

// Scale by N^(-1) for inverse NTT
__global__ void scale_kernel(uint64_t* data, uint64_t n, uint64_t n_inv, uint64_t q) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    data[idx] = (unsigned __int128)data[idx] * n_inv % q;
}

// Initialize NTT configuration
NTTConfig* ntt_init(uint64_t n, uint64_t q) {
    assert((n & (n - 1)) == 0 && "N must be a power of 2");
    
    NTTConfig* config = new NTTConfig();
    config->n = n;
    config->log_n = __builtin_ctzll(n);
    config->q = q;
    // Barrett constant mu = floor(2^64 / q). Computing on host once is exact.
    // Used by ntt_optimized kernels for fast modular reduction.
    {
        unsigned __int128 num = (unsigned __int128)1 << 64;
        config->mu = (uint64_t)(num / q);
    }
    config->params = create_params(q, n);
    
    // Allocate and compute twiddle factors on host
    std::vector<uint64_t> twiddles(n);
    std::vector<uint64_t> twiddles_inv(n);
    
    compute_twiddle_factors(twiddles.data(), n, config->params.omega, q);
    compute_twiddle_factors(twiddles_inv.data(), n, config->params.omega_inv, q);
    
    // Copy to device
    CUDA_CHECK(cudaMalloc(&config->d_twiddles, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&config->d_twiddles_inv, n * sizeof(uint64_t)));
    
    CUDA_CHECK(cudaMemcpy(config->d_twiddles, twiddles.data(), 
                          n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(config->d_twiddles_inv, twiddles_inv.data(),
                          n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    return config;
}

void ntt_cleanup(NTTConfig* config) {
    if (config) {
        if (config->d_twiddles) cudaFree(config->d_twiddles);
        if (config->d_twiddles_inv) cudaFree(config->d_twiddles_inv);
        delete config;
    }
}

void bit_reverse_permute(uint64_t* d_data, uint64_t n, cudaStream_t stream) {
    int log_n = __builtin_ctzll(n);
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    
    bit_reverse_kernel<<<num_blocks, block_size, 0, stream>>>(d_data, n, log_n);
}

// Basic forward NTT implementation
void ntt_forward_basic(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n = config->n;
    int log_n = config->log_n;
    uint64_t q = config->q;
    
    if (n <= MAX_SMEM_ELEMENTS) {
        // Use shared memory kernel for small N
        int block_size = min(512, (int)(n / 2));
        size_t smem_size = n * sizeof(uint64_t);
        
        ntt_smem_kernel<<<1, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles, n, log_n, q, false
        );
    } else {
        // Bit-reverse permutation first
        bit_reverse_permute(d_data, n, stream);
        
        // Process stage by stage
        int block_size = 256;
        uint64_t num_butterflies = n >> 1;
        int num_blocks = (num_butterflies + block_size - 1) / block_size;
        
        for (int stage = 0; stage < log_n; stage++) {
            ntt_stage_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_data, config->d_twiddles, n, stage, q
            );
        }
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void ntt_inverse_basic(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n = config->n;
    int log_n = config->log_n;
    uint64_t q = config->q;
    
    if (n <= MAX_SMEM_ELEMENTS) {
        int block_size = min(512, (int)(n / 2));
        size_t smem_size = n * sizeof(uint64_t);
        
        ntt_smem_kernel<<<1, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles_inv, n, log_n, q, true
        );
    } else {
        bit_reverse_permute(d_data, n, stream);
        
        int block_size = 256;
        uint64_t num_butterflies = n >> 1;
        int num_blocks = (num_butterflies + block_size - 1) / block_size;
        
        for (int stage = 0; stage < log_n; stage++) {
            ntt_stage_kernel<<<num_blocks, block_size, 0, stream>>>(
                d_data, config->d_twiddles_inv, n, stage, q
            );
        }
    }
    
    // Scale by N^(-1)
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    scale_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_data, n, config->params.n_inv, q
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Batched NTT
void ntt_forward_batch(uint64_t* d_data, uint64_t batch_size, 
                       const NTTConfig* config, NTTVariant variant,
                       cudaStream_t stream) {
    switch (variant) {
        case NTTVariant::BASIC:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_forward_basic(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::FOUR_STEP:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_forward_four_step(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::TLMOP:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_forward_tlmop(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::FULL_OPTIMIZED:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_forward_optimized(d_data + i * config->n, config, stream);
            }
            break;
    }
}

void ntt_inverse_batch(uint64_t* d_data, uint64_t batch_size,
                       const NTTConfig* config, NTTVariant variant, 
                       cudaStream_t stream) {
    switch (variant) {
        case NTTVariant::BASIC:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_inverse_basic(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::FOUR_STEP:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_inverse_four_step(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::TLMOP:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_inverse_tlmop(d_data + i * config->n, config, stream);
            }
            break;
        case NTTVariant::FULL_OPTIMIZED:
            for (uint64_t i = 0; i < batch_size; i++) {
                ntt_inverse_optimized(d_data + i * config->n, config, stream);
            }
            break;
    }
}

// Element-wise multiplication in NTT domain
__global__ void poly_mul_kernel(uint64_t* result, const uint64_t* a, const uint64_t* b, 
                                 uint64_t n, uint64_t q) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    result[idx] = (unsigned __int128)a[idx] * b[idx] % q;
}

void poly_multiply_ntt(uint64_t* d_result, const uint64_t* d_a, const uint64_t* d_b,
                       const NTTConfig* config, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (config->n + block_size - 1) / block_size;
    
    poly_mul_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_result, d_a, d_b, config->n, config->q
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host-side NTT for verification
void ntt_host(std::vector<uint64_t>& data, const ModularParams& params, bool inverse) {
    uint64_t n = data.size();
    int log_n = __builtin_ctzll(n);
    uint64_t q = params.q;
    uint64_t omega = inverse ? params.omega_inv : params.omega;
    
    // Bit-reverse permutation
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0;
        uint64_t tmp = i;
        for (int j = 0; j < log_n; j++) {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }
    
    // Cooley-Tukey NTT
    for (int stage = 0; stage < log_n; stage++) {
        uint64_t half_size = 1ULL << stage;
        uint64_t full_size = half_size << 1;
        uint64_t tw_stride = n / full_size;
        
        // Compute twiddle factor for this stage
        uint64_t w_base = mod_pow(omega, tw_stride, q);
        
        for (uint64_t group = 0; group < n / full_size; group++) {
            uint64_t w = 1;
            for (uint64_t pos = 0; pos < half_size; pos++) {
                uint64_t i = group * full_size + pos;
                uint64_t j = i + half_size;
                
                uint64_t u = data[i];
                uint64_t v = (unsigned __int128)data[j] * w % q;
                
                data[i] = (u + v) % q;
                data[j] = (u >= v) ? (u - v) : (u + q - v);
                
                w = (unsigned __int128)w * w_base % q;
            }
        }
    }
    
    // Scale by N^(-1) for inverse
    if (inverse) {
        for (uint64_t i = 0; i < n; i++) {
            data[i] = (unsigned __int128)data[i] * params.n_inv % q;
        }
    }
}

} // namespace ntt
