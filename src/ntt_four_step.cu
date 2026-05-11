#include "ntt.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace ntt {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants for 4-step NTT
constexpr int INNER_NTT_SIZE = 64;  // Radix-64 inner NTT (8x8 for tensor cores)
constexpr int LOG_INNER_SIZE = 6;

// Bit-reverse within a range
__device__ __forceinline__
uint32_t bit_reverse_n(uint32_t x, int bits) {
    uint32_t result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Inner NTT kernel: radix-64 NTT in shared memory
// Processes multiple "pads" (inner NTTs) per block for coalesced access
__global__ void inner_ntt_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint64_t n,
    uint64_t n1,        // Inner dimension (typically 64)
    uint64_t n2,        // Outer dimension (n / n1)
    int log_n1,
    uint64_t q,
    bool is_column_wise  // True for step 1, false for step 4
) {
    extern __shared__ uint64_t smem[];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    
    // Each block processes multiple columns (pads)
    int pads_per_block = blockDim.x / (n1 / 2);  // Threads per butterfly = n1/2
    int pad_in_block = tid / (n1 / 2);
    int local_tid = tid % (n1 / 2);
    
    uint64_t global_pad_idx = blockIdx.x * pads_per_block + pad_in_block;
    if (global_pad_idx >= n2) return;
    
    // Compute base index
    uint64_t base_idx;
    if (is_column_wise) {
        // Column-wise: data[col * n2 + row] for row in [0, n1)
        base_idx = global_pad_idx;  // Column index
    } else {
        // Row-wise: data[row * n1 + col] for col in [0, n1)
        base_idx = global_pad_idx * n1;
    }
    
    // Load data into shared memory
    uint64_t* local_smem = smem + pad_in_block * n1;
    
    if (is_column_wise) {
        // Strided load for column-wise access
        for (int i = local_tid; i < n1; i += (n1 / 2)) {
            local_smem[i] = data[i * n2 + global_pad_idx];
        }
        for (int i = local_tid + (n1 / 2); i < n1; i += (n1 / 2)) {
            local_smem[i] = data[i * n2 + global_pad_idx];
        }
    } else {
        // Contiguous load for row-wise access
        for (int i = local_tid; i < n1; i += (n1 / 2)) {
            local_smem[i] = data[base_idx + i];
        }
        for (int i = local_tid + (n1 / 2); i < n1; i += (n1 / 2)) {
            local_smem[i] = data[base_idx + i];
        }
    }
    __syncwarp();
    
    // Bit-reverse permutation within shared memory
    for (int i = local_tid; i < n1; i += (n1 / 2)) {
        uint32_t rev_i = bit_reverse_n(i, log_n1);
        if (i < rev_i) {
            uint64_t tmp = local_smem[i];
            local_smem[i] = local_smem[rev_i];
            local_smem[rev_i] = tmp;
        }
    }
    __syncwarp();
    
    // Perform radix-n1 NTT using Cooley-Tukey
    for (int stage = 0; stage < log_n1; stage++) {
        uint64_t half_size = 1ULL << stage;
        uint64_t full_size = half_size << 1;
        uint64_t tw_stride = n1 / full_size;
        
        for (int idx = local_tid; idx < (n1 >> 1); idx += (n1 / 2)) {
            uint64_t group = idx / half_size;
            uint64_t pos = idx % half_size;
            
            uint64_t i = group * full_size + pos;
            uint64_t j = i + half_size;
            
            // Twiddle factor for inner NTT
            uint64_t tw_idx = pos * tw_stride;
            uint64_t omega_power = tw_idx * (n / n1);  // Scale to full twiddle table
            uint64_t w = twiddles[omega_power % n];
            
            uint64_t u = local_smem[i];
            uint64_t v = local_smem[j];
            
            uint64_t wv = (unsigned __int128)w * v % q;
            
            local_smem[i] = (u + wv) % q;
            local_smem[j] = (u >= wv) ? (u - wv) : (u + q - wv);
        }
        __syncwarp();
    }
    
    // Write back
    if (is_column_wise) {
        for (int i = local_tid; i < n1; i += (n1 / 2)) {
            data[i * n2 + global_pad_idx] = local_smem[i];
        }
        for (int i = local_tid + (n1 / 2); i < n1; i += (n1 / 2)) {
            data[i * n2 + global_pad_idx] = local_smem[i];
        }
    } else {
        for (int i = local_tid; i < n1; i += (n1 / 2)) {
            data[base_idx + i] = local_smem[i];
        }
        for (int i = local_tid + (n1 / 2); i < n1; i += (n1 / 2)) {
            data[base_idx + i] = local_smem[i];
        }
    }
}

// Element-wise twiddle factor multiplication (Hadamard product)
// data[i,j] *= omega^(i*j) where i is row, j is column
__global__ void twiddle_multiply_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint64_t n,
    uint64_t n1,
    uint64_t n2,
    uint64_t q
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Convert linear index to 2D (row-major)
    uint64_t row = idx / n2;
    uint64_t col = idx % n2;
    
    // Twiddle factor: omega^(row * col)
    uint64_t tw_idx = ((row * col) % n);
    uint64_t w = twiddles[tw_idx];
    
    data[idx] = (unsigned __int128)data[idx] * w % q;
}

// Matrix transpose kernel
// Transposes n1 x n2 matrix to n2 x n1
__global__ void transpose_kernel(
    uint64_t* output,
    const uint64_t* input,
    uint64_t n1,
    uint64_t n2
) {
    // Use shared memory for coalesced access
    extern __shared__ uint64_t tile[];
    
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile into shared memory (coalesced read)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n2 && (y + j) < n1) {
            tile[(threadIdx.y + j) * (TILE_DIM + 1) + threadIdx.x] = 
                input[(y + j) * n2 + x];
        }
    }
    __syncthreads();
    
    // Write transposed tile (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n1 && (y + j) < n2) {
            output[(y + j) * n1 + x] = 
                tile[threadIdx.x * (TILE_DIM + 1) + threadIdx.y + j];
        }
    }
}

// In-place transpose using a temporary buffer
void transpose_inplace(uint64_t* d_data, uint64_t n1, uint64_t n2, cudaStream_t stream) {
    uint64_t* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n1 * n2 * sizeof(uint64_t)));
    
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((n2 + TILE_DIM - 1) / TILE_DIM, (n1 + TILE_DIM - 1) / TILE_DIM);
    size_t smem_size = TILE_DIM * (TILE_DIM + 1) * sizeof(uint64_t);
    
    transpose_kernel<<<grid, block, smem_size, stream>>>(d_temp, d_data, n1, n2);
    
    CUDA_CHECK(cudaMemcpyAsync(d_data, d_temp, n1 * n2 * sizeof(uint64_t), 
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFree(d_temp));
}

// Scale by N^(-1) for inverse NTT
__global__ void scale_inverse_kernel(uint64_t* data, uint64_t n, uint64_t n_inv, uint64_t q) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    data[idx] = (unsigned __int128)data[idx] * n_inv % q;
}

// 4-Step Forward NTT
// Algorithm:
// 1. Column-wise NTTs (n2 NTTs of size n1)
// 2. Element-wise twiddle multiplication
// 3. Transpose (n1 x n2 -> n2 x n1)
// 4. Row-wise NTTs (n1 NTTs of size n2)
void ntt_forward_four_step(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n = config->n;
    uint64_t q = config->q;
    
    // Choose decomposition: N = N1 * N2
    // For balanced load, make N1 ≈ N2
    // For tensor cores: N1 should be multiple of 64
    uint64_t log_n = config->log_n;
    uint64_t log_n1 = log_n / 2;
    if (log_n1 < 6) log_n1 = 6;  // Minimum 64 for tensor core compatibility
    if (log_n1 > log_n) log_n1 = log_n;
    
    uint64_t n1 = 1ULL << log_n1;
    uint64_t n2 = n / n1;
    uint64_t log_n2 = log_n - log_n1;
    
    // Step 1: Column-wise NTTs
    // Process n2 columns, each of size n1
    {
        int threads_per_ntt = n1 / 2;
        int pads_per_block = 1;  // Can increase for more parallelism
        while (pads_per_block * threads_per_ntt <= 256 && pads_per_block < 4) {
            pads_per_block++;
        }
        pads_per_block--;
        if (pads_per_block < 1) pads_per_block = 1;
        
        int block_size = pads_per_block * threads_per_ntt;
        int num_blocks = (n2 + pads_per_block - 1) / pads_per_block;
        size_t smem_size = pads_per_block * n1 * sizeof(uint64_t);
        
        inner_ntt_kernel<<<num_blocks, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles, n, n1, n2, log_n1, q, true
        );
    }
    
    // Step 2: Twiddle factor multiplication
    {
        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        
        twiddle_multiply_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_data, config->d_twiddles, n, n1, n2, q
        );
    }
    
    // Step 3: Transpose
    transpose_inplace(d_data, n1, n2, stream);
    
    // Step 4: Row-wise NTTs (now n1 rows of size n2)
    // After transpose, dimensions are swapped
    {
        int threads_per_ntt = n2 / 2;
        int pads_per_block = 1;
        while (pads_per_block * threads_per_ntt <= 256 && pads_per_block < 4) {
            pads_per_block++;
        }
        pads_per_block--;
        if (pads_per_block < 1) pads_per_block = 1;
        
        int block_size = pads_per_block * threads_per_ntt;
        int num_blocks = (n1 + pads_per_block - 1) / pads_per_block;
        size_t smem_size = pads_per_block * n2 * sizeof(uint64_t);
        
        inner_ntt_kernel<<<num_blocks, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles, n, n2, n1, log_n2, q, false
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// 4-Step Inverse NTT
void ntt_inverse_four_step(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n = config->n;
    uint64_t q = config->q;
    
    uint64_t log_n = config->log_n;
    uint64_t log_n1 = log_n / 2;
    if (log_n1 < 6) log_n1 = 6;
    if (log_n1 > log_n) log_n1 = log_n;
    
    uint64_t n1 = 1ULL << log_n1;
    uint64_t n2 = n / n1;
    uint64_t log_n2 = log_n - log_n1;
    
    // Inverse is similar but uses inverse twiddles
    // Step 1: Row-wise inverse NTTs
    {
        int threads_per_ntt = n2 / 2;
        int pads_per_block = 1;
        while (pads_per_block * threads_per_ntt <= 256 && pads_per_block < 4) {
            pads_per_block++;
        }
        pads_per_block--;
        if (pads_per_block < 1) pads_per_block = 1;
        
        int block_size = pads_per_block * threads_per_ntt;
        int num_blocks = (n1 + pads_per_block - 1) / pads_per_block;
        size_t smem_size = pads_per_block * n2 * sizeof(uint64_t);
        
        inner_ntt_kernel<<<num_blocks, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles_inv, n, n2, n1, log_n2, q, false
        );
    }
    
    // Step 2: Transpose back
    transpose_inplace(d_data, n2, n1, stream);
    
    // Step 3: Inverse twiddle multiplication
    {
        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        
        twiddle_multiply_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_data, config->d_twiddles_inv, n, n1, n2, q
        );
    }
    
    // Step 4: Column-wise inverse NTTs
    {
        int threads_per_ntt = n1 / 2;
        int pads_per_block = 1;
        while (pads_per_block * threads_per_ntt <= 256 && pads_per_block < 4) {
            pads_per_block++;
        }
        pads_per_block--;
        if (pads_per_block < 1) pads_per_block = 1;
        
        int block_size = pads_per_block * threads_per_ntt;
        int num_blocks = (n2 + pads_per_block - 1) / pads_per_block;
        size_t smem_size = pads_per_block * n1 * sizeof(uint64_t);
        
        inner_ntt_kernel<<<num_blocks, block_size, smem_size, stream>>>(
            d_data, config->d_twiddles_inv, n, n1, n2, log_n1, q, true
        );
    }
    
    // Scale by N^(-1)
    {
        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        scale_inverse_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_data, n, config->params.n_inv, q
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ntt
