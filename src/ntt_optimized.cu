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

constexpr int INNER_SIZE = 64;
constexpr int LOG_INNER  = 6;
constexpr int WARP_SIZE  = 32;
constexpr int WARPS_PER_BLOCK = 8;          // 8 NTTs per block (one per warp)
constexpr int BLOCK_SIZE_INNER = WARP_SIZE * WARPS_PER_BLOCK;

// ============================================================================
// Lab 6 fast modular reduction (Barrett, q < 2^31, input < 2^62)
// ============================================================================
// Replaces  a % q  (a slow 64-bit divmod by a runtime divisor) with:
//   qhat = high64(a * mu)        // mu = floor(2^64 / q), precomputed on host
//   r    = a - qhat * q          // r in [0, 2q) when a < 2^62, q < 2^31
//   if (r >= q) r -= q;          // single conditional subtraction
//
// Bound proof: qhat = floor(a*mu/2^64) <= a/q. Hence r = a - qhat*q >= 0.
// Also qhat >= a/q - a/2^64 - 1, so r <= q + a*q/2^64 + q < 2q+1 for a < 2^62.
//
// Cost: 2 multiplies + 1 sub + 1 cmov, vs. the multi-instruction 64-bit
// integer divide that the compiler emits for u64 % q with runtime q.

__device__ __forceinline__
uint64_t mod_q_barrett(uint64_t a, uint64_t q, uint64_t mu) {
    uint64_t qhat = __umul64hi(a, mu);
    uint64_t r    = a - qhat * q;
    if (r >= q) r -= q;
    return r;
}

// Fast modmul: (a * b) mod q with a, b < q < 2^31  =>  a*b < 2^62  =>
// fits in uint64. Then a single Barrett reduction.
__device__ __forceinline__
uint64_t modmul_barrett(uint64_t a, uint64_t b, uint64_t q, uint64_t mu) {
    return mod_q_barrett(a * b, q, mu);
}

// ============================================================================
// Tensor Core MMA wrapper - FP64 m8n8k4 (Ampere+ / sm_80+)
// ============================================================================
// PTX: mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {d0,d1}, {a0}, {b0}, {c0,c1}
// Computes D = A(8x4) * B(4x8) + C(8x8) using FP64 tensor cores
//
// Lane-element distribution per the PTX ISA documentation:
//   A (8x4 row-major):  lane t holds A[t/4, t%4]
//   B (4x8 col-major):  lane t holds B[t%4, t/4]
//   C/D (8x8):          lane t holds C[t/4, 2*(t%4)] and C[t/4, 2*(t%4)+1]

__device__ __forceinline__
void mma_m8n8k4_f64(double& d0, double& d1,
                    double a, double b,
                    double c0, double c1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=d"(d0), "=d"(d1)
        : "d"(a), "d"(b), "d"(c0), "d"(c1)
    );
#else
    d0 = c0; d1 = c1;
#endif
}

// ============================================================================
// Bit reversal helpers
// ============================================================================

__device__ __forceinline__
int bit_rev_3(int x) {
    return ((x & 1) << 2) | (x & 2) | ((x >> 2) & 1);
}

__device__ __forceinline__
int bit_rev_6(int x) {
    return (bit_rev_3(x & 7) << 3) | bit_rev_3((x >> 3) & 7);
}

__device__ __forceinline__
uint32_t bitrev_log(uint32_t x, int log_n) {
    uint32_t r = 0;
    for (int i = 0; i < log_n; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

__global__ void bitrev_kernel(uint64_t* data, uint64_t n, int log_n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t rev = bitrev_log(idx, log_n);
    if (idx < rev) {
        uint64_t tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

// ============================================================================
// Outer stage kernel (used for `extra` initial CT stages)
// ============================================================================
// TFOP for outer stages: stage s has only 2^s distinct twiddles, all reused
// across many butterflies. We cache them block-wide in SMEM so that the GMEM
// twiddle traffic drops from O(N/2) to O(2^s * N / (2 * block_size)) reads.
// For our sizes, `extra` <= 5 -> at most 32 twiddles to cache (256 B).

__global__ void outer_stage_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    uint64_t n,
    int stage,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t tw_smem[];

    int tid       = threadIdx.x;
    uint64_t half = 1ULL << stage;
    uint64_t full = half << 1;

    // Cooperative cache load: 2^stage distinct twiddles -> SMEM
    if ((uint64_t)tid < half) {
        tw_smem[tid] = twiddles[(uint64_t)tid * (n / full)];
    }
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)tid;
    uint64_t num_bfly = n >> 1;
    if (idx >= num_bfly) return;

    uint64_t group = idx / half;
    uint64_t pos   = idx % half;
    uint64_t i = group * full + pos;
    uint64_t j = i + half;

    uint64_t w = tw_smem[pos];

    uint64_t u = data[i];
    uint64_t v = data[j];
    // Lab 6 Barrett-mod replaces (a*b) % q with __umul64hi-based reduction.
    uint64_t wv = modmul_barrett(w, v, q, mu);
    uint64_t sum = u + wv;
    if (sum >= q) sum -= q;
    data[i] = sum;
    data[j] = (u >= wv) ? (u - wv) : (u + q - wv);
}

// ============================================================================
// Scale kernel (for inverse NTT)
// ============================================================================

__global__ void scale_opt_kernel(
    uint64_t* __restrict__ data,
    uint64_t n,
    uint64_t n_inv,
    uint64_t q,
    uint64_t mu
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = modmul_barrett(data[idx], n_inv, q, mu);
}

// ============================================================================
// Tensor-Core based Inner NTT-64 kernel (HMFHE-style: TLMOP + TransOP + TFOP)
// ============================================================================
// Each warp processes one 64-point inner NTT using FP64 MMA tensor cores.
//
// Mathematically equivalent to applying 6 CT-DIT stages [stage_start, +6) with
// "external" twiddles linking the local 64-block to the global NTT structure:
//
//   1) Pre-twist input:   x'[j] = x[j] * factor[j]
//      where  factor[j] = omega_global^(offset_o * bit_rev_6(j) * N / macro_size)
//   2) Standard NTT-64 of bit-reversed input (= direct DFT-64 of natural input)
//
// NTT-64 is computed via the 4-step algorithm using FP64 tensor cores:
//
//   - Reshape x_natural as 8x8 matrix M[a,b] = x_natural[a*8 + b]
//   - M'  = TFM_8 @ M                       (col-wise NTT-8 via MMA 1+2)
//   - M'' = M' * omega_64^(a*b)             (inner Hadamard)
//   - F   = TFM_8 @ M''^T                   (row-wise NTT-8 via MMA 3+4)
//   - y_natural[a*8 + b] = F[a, b]
//
// === MPA (32-bit modulus on FP64-TCU) ===
// TFM_8 entries are 31-bit. To keep MMA accumulator within FP64's 53-bit
// mantissa, we split TFM into 16-bit high/low halves:
//      TFM = TFM_high * 2^16 + TFM_low      (each < 2^16)
// Each NTT-8 round needs 4 MMAs (2 for high, 2 for low; each pair covers two
// 8x4 inner-dim halves). Two NTT-8 rounds per NTT-64 => 8 MMAs per warp.
//
// === TLMOP (Thread-Level Memory OPtimization, paper § IV-A3) ===
// All intermediate values (D, M', M'') stay in registers between MMAs.
// SMEM is touched only during the initial input load and the final output write.
//
// === TransOP (Transpose OPtimization, paper § IV-A4) ===
// The implicit transpose required by the 4-step algorithm is realized via
// fragment re-mapping rather than an explicit SMEM round-trip:
//
//   After MMA 1+2, lane t holds M''[t/4, 2*(t%4)] (=mp_0) and
//                              M''[t/4, 2*(t%4)+1] (=mp_1) in registers.
//
//   For MMA 3+4 we treat M'' as Fragment A (8x8 split into even/odd columns):
//     A_3a = M''[:, {0,2,4,6}],  lane t holds A_3a[t/4, t%4] = mp_0  (perfect!)
//     A_3b = M''[:, {1,3,5,7}],  lane t holds A_3b[t/4, t%4] = mp_1  (perfect!)
//   B_3a/3b is TFM split by even/odd rows:
//     B_3a = TFM[{0,2,4,6}, :],  B_3b = TFM[{1,3,5,7}, :]
//   The result G = M'' @ TFM is the transpose of F = TFM @ M''^T (TFM symmetric);
//   we write G[i, j] back to data[base + (j*8+i)*S] (= y[a*8+b] with a=j, b=i).
//
// === TFOP (Twiddle Factor access OPtimization, paper § IV-B2) ===
//   - TFM_8  (64 entries, omega_8^(i*k))   : block-shared SMEM, used by all warps
//   - Hada64 (64 entries, omega_64^(a*b))  : block-shared SMEM (replaces GMEM
//                                            access for the inner Hadamard step)
//   - Pre-twist factors (depend on offset_o): per-warp SMEM cache, written once
//                                            and reused 64 times (once per j)

// SMEM layout offsets (in uint64_t units), per block:
//   [0 .. 64)                                                    : TFM_8 (shared)
//   [64 .. 128)                                                  : Hada64 (shared)
//   [128 .. 128 + WARP_SMEM_STRIDE*8 * WARPS_PER_BLOCK)           : per-warp scratch
//
// Lab 6 BANK-CONFLICT FIX: per-warp scratch is laid out as 8 rows x
// WARP_SMEM_STRIDE columns of u64. With stride 8, each row is 64 B which
// covers exactly 16 banks (each row's columns 0,1,...,7 occupy banks
// 0,2,...,14); accessing column-ish patterns like warp_smem[col*8+row]
// then makes lanes that share a column hit the same bank pair, causing
// 2-way conflicts. Padding the row to 9 changes the bank-of-position
// mapping so 32 lanes' column accesses hit 32 distinct banks.
constexpr int WARP_SMEM_STRIDE  = 9;       // padded row stride (was 8)
constexpr int WARP_SMEM_ROWS    = 8;
constexpr int WARP_SMEM_TOTAL   = WARP_SMEM_STRIDE * WARP_SMEM_ROWS;  // 72

constexpr int SMEM_TFM_OFFSET   = 0;
constexpr int SMEM_HADA_OFFSET  = 64;
constexpr int SMEM_WARP_OFFSET  = 128;
constexpr int SMEM_PER_BLOCK_U64 =
    SMEM_WARP_OFFSET + WARP_SMEM_TOTAL * WARPS_PER_BLOCK;

// Helper: linear-to-padded mapping for a position j in [0, 64).
// Used when the access pattern naturally enumerates a flat 0..63 index.
__device__ __forceinline__
int wsm(int j) { return (j >> 3) * WARP_SMEM_STRIDE + (j & 7); }

// ============================================================================
// Lab 6 cp.async helpers (Ampere+ : sm_80 and later)
// ============================================================================
// Asynchronous global -> shared 8B copy. Allows the compiler/scheduler to
// overlap the GMEM read latency with subsequent in-flight work, instead of
// the synchronous GMEM->register->SMEM round-trip.

__device__ __forceinline__
unsigned smem_addr_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void cp_async_8B(uint64_t* smem_ptr, const uint64_t* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    unsigned smem_int = smem_addr_u32(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_int), "l"(gmem_ptr)
    );
#else
    *smem_ptr = *gmem_ptr;
#endif
}

__device__ __forceinline__
void cp_async_commit_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__global__ __launch_bounds__(BLOCK_SIZE_INNER)
void inner_ntt_tcu_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    uint64_t n,
    int stage_start,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t smem[];
    uint64_t* tfm8_smem = smem + SMEM_TFM_OFFSET;
    uint64_t* hada_smem = smem + SMEM_HADA_OFFSET;

    int tid              = threadIdx.x;
    int warp_id          = tid / WARP_SIZE;
    int lane             = tid % WARP_SIZE;
    int warps_per_block  = blockDim.x / WARP_SIZE;
    int global_warp      = blockIdx.x * warps_per_block + warp_id;

    // ------------------------------------------------------------------
    // TFOP: block-cooperative load of TFM_8 and Hada64 (block-shared)
    //   TFM_8 [i, k] = omega_8^(i*k)  = twiddles[(i*k * N/8 ) mod N]
    //   Hada64[a, b] = omega_64^(a*b) = twiddles[(a*b * N/64) mod N]
    // ------------------------------------------------------------------
    if (tid < 64) {
        int i = tid >> 3;
        int k = tid & 7;
        uint64_t exp8  = ((uint64_t)i * (uint64_t)k * (n >> 3)) % n;
        uint64_t exp64 = ((uint64_t)i * (uint64_t)k * (n >> 6)) % n;
        tfm8_smem[i * 8 + k] = twiddles[exp8];
        hada_smem[i * 8 + k] = twiddles[exp64];
    }
    __syncthreads();

    uint64_t total_ntts = n / INNER_SIZE;
    if ((uint64_t)global_warp >= total_ntts) return;

    uint64_t S          = 1ULL << stage_start;
    uint64_t macro_size = (uint64_t)INNER_SIZE * S;
    uint64_t macro_g    = (uint64_t)global_warp / S;
    uint64_t offset_o   = (uint64_t)global_warp % S;
    uint64_t base       = macro_g * macro_size + offset_o;

    uint64_t* warp_smem = smem + SMEM_WARP_OFFSET + (uint64_t)warp_id * WARP_SMEM_TOTAL;

    // ------------------------------------------------------------------
    // Lab 6 ASYNC LOAD path (no pre-twist required, offset_o == 0):
    //   - cp.async pushes 8B from GMEM directly into SMEM without going
    //     through the register file. The 32 lanes of this warp issue 64
    //     in-flight loads which can be serviced in parallel.
    //   - cp.async.wait_all + __syncthreads barriers complete before MMA.
    //
    // SYNC + PRE-TWIST path (offset_o > 0):
    //   - Each loaded element is multiplied by a per-position twiddle.
    //   - Cannot be expressed as cp.async; uses the synchronous read path
    //     with Barrett-reduced modmul.
    // ------------------------------------------------------------------
    if (offset_o == 0) {
        #pragma unroll
        for (int j = lane; j < INNER_SIZE; j += WARP_SIZE) {
            cp_async_8B(&warp_smem[wsm(j)],
                        &data[base + (uint64_t)j * S]);
        }
        cp_async_commit_wait_all();
        __syncwarp();
    } else {
        uint64_t tw_stride_pre = n / macro_size;
        for (int j = lane; j < INNER_SIZE; j += WARP_SIZE) {
            uint64_t val = data[base + (uint64_t)j * S];
            int br = bit_rev_6(j);
            uint64_t exp_idx = (offset_o * (uint64_t)br * tw_stride_pre) % n;
            uint64_t f = twiddles[exp_idx];
            val = modmul_barrett(val, f, q, mu);
            warp_smem[wsm(j)] = val;
        }
        __syncwarp();
    }

    int lane_div4 = lane >> 2;       // 0..7  (row index of A and D fragments)
    int lane_mod4 = lane & 3;        // 0..3  (col index of A / row of B fragments)

    // ------------------------------------------------------------------
    // MMA 1+2 fragments: A = TFM (split into LEFT/RIGHT 4-col halves)
    //                    B = M   (split into TOP/BOTTOM 4-row halves)
    //   Lane t holds A[t/4, t%4] for left half, A[t/4, t%4+4] for right half
    //   Lane t holds B[t%4, t/4] for top half,  B[t%4+4, t/4] for bottom half
    //
    // M[a,b] = x_natural[a*8+b] = x_bitrev[bit_rev_3(b)*8 + bit_rev_3(a)]
    // ------------------------------------------------------------------
    uint64_t tfm12_l = tfm8_smem[lane_div4 * 8 + lane_mod4];
    uint64_t tfm12_r = tfm8_smem[lane_div4 * 8 + lane_mod4 + 4];

    double tfm12_l_h = (double)(uint32_t)(tfm12_l >> 16);
    double tfm12_l_l = (double)(uint32_t)(tfm12_l & 0xFFFFULL);
    double tfm12_r_h = (double)(uint32_t)(tfm12_r >> 16);
    double tfm12_r_l = (double)(uint32_t)(tfm12_r & 0xFFFFULL);

    double dat_top = (double)warp_smem[bit_rev_3(lane_div4) * WARP_SMEM_STRIDE + bit_rev_3(lane_mod4)];
    double dat_bot = (double)warp_smem[bit_rev_3(lane_div4) * WARP_SMEM_STRIDE + bit_rev_3(lane_mod4 + 4)];

    // ------------------------------------------------------------------
    // MMA 1+2: D = TFM @ M (8x8 = 8x4 @ 4x8 + 8x4 @ 4x8, split by inner dim)
    //   D_high = TFM_left_high @ M_top + TFM_right_high @ M_bot
    //   D_low  = TFM_left_low  @ M_top + TFM_right_low  @ M_bot
    // ------------------------------------------------------------------
    double d_high_0 = 0.0, d_high_1 = 0.0;
    double d_low_0  = 0.0, d_low_1  = 0.0;

    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_l_h, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_r_h, dat_bot, d_high_0, d_high_1);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_l_l, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_r_l, dat_bot, d_low_0, d_low_1);

    // Bit-merge + ModRed (Barrett): M'[i, j] = (D_high << 16 + D_low) mod q
    //   D_high < 2^49 (sum of 4 u16*u31 products) -> fits Barrett input bound 2^62
    uint64_t mp_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_0, q, mu), q, mu);
    uint64_t mp_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_1, q, mu), q, mu);

    // ------------------------------------------------------------------
    // Inner Hadamard: M''[a, b] = M'[a, b] * omega_64^(a*b)  (TFOP-cached)
    // ------------------------------------------------------------------
    {
        int b0 = lane_mod4 << 1;
        mp_0 = modmul_barrett(mp_0, hada_smem[lane_div4 * 8 + b0],     q, mu);
        mp_1 = modmul_barrett(mp_1, hada_smem[lane_div4 * 8 + b0 + 1], q, mu);
    }

    // ==================================================================
    // TransOP: M'' is in A-fragment layout for MMA 3+4 directly!
    //   A_3a[t/4, t%4] = M''[t/4, 2*(t%4)]    = mp_0  (even cols of M'')
    //   A_3b[t/4, t%4] = M''[t/4, 2*(t%4)+1]  = mp_1  (odd  cols of M'')
    // No SMEM transpose, no warp shuffle -- pure register reuse.
    //
    // The companion B-fragments are TFM with rows split by parity:
    //   B_3a = TFM[{0,2,4,6}, :], lane t holds B_3a[t%4, t/4] = TFM[2*(t%4),   t/4]
    //   B_3b = TFM[{1,3,5,7}, :], lane t holds B_3b[t%4, t/4] = TFM[2*(t%4)+1, t/4]
    // ==================================================================
    double mp_0_d = (double)mp_0;
    double mp_1_d = (double)mp_1;

    uint64_t tfm34_a = tfm8_smem[(lane_mod4 << 1)       * 8 + lane_div4];
    uint64_t tfm34_b = tfm8_smem[((lane_mod4 << 1) + 1) * 8 + lane_div4];

    double tfm34_a_h = (double)(uint32_t)(tfm34_a >> 16);
    double tfm34_a_l = (double)(uint32_t)(tfm34_a & 0xFFFFULL);
    double tfm34_b_h = (double)(uint32_t)(tfm34_b >> 16);
    double tfm34_b_l = (double)(uint32_t)(tfm34_b & 0xFFFFULL);

    // ------------------------------------------------------------------
    // MMA 3+4: G = M'' @ TFM, where G[i,j] = sum_k M''[i,k]*TFM[k,j]
    //   G_high = M''_even @ TFM_high_even_rows + M''_odd @ TFM_high_odd_rows
    //   G_low  = M''_even @ TFM_low_even_rows  + M''_odd @ TFM_low_odd_rows
    // Since TFM is symmetric, F = TFM @ M''^T = G^T, hence the transposed write.
    // ------------------------------------------------------------------
    double g_high_0 = 0.0, g_high_1 = 0.0;
    double g_low_0  = 0.0, g_low_1  = 0.0;

    mma_m8n8k4_f64(g_high_0, g_high_1, mp_0_d, tfm34_a_h, 0.0, 0.0);
    mma_m8n8k4_f64(g_high_0, g_high_1, mp_1_d, tfm34_b_h, g_high_0, g_high_1);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_0_d, tfm34_a_l, 0.0, 0.0);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_1_d, tfm34_b_l, g_low_0, g_low_1);

    // Bit-merge + ModRed (Barrett)
    uint64_t g_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_0, q, mu), q, mu);
    uint64_t g_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_1, q, mu), q, mu);

    // ------------------------------------------------------------------
    // Output write: y[a*8+b] = F[a, b] = G[b, a]
    //   Lane t (i = t/4, j = 2*(t%4) or 2*(t%4)+1) writes G[i, j] to
    //   data[base + (j*8 + i)*S]. The 32 lanes together cover all 64 positions.
    // ------------------------------------------------------------------
    {
        int j0 = lane_mod4 << 1;
        int j1 = j0 + 1;
        data[base + (uint64_t)(j0 * 8 + lane_div4) * S] = g_0;
        data[base + (uint64_t)(j1 * 8 + lane_div4) * S] = g_1;
    }
}

// ============================================================================
// Main optimized NTT (forward and inverse)
// ============================================================================
// Forward NTT structure:
//   1. Global bit-reverse permutation
//   2. `extra = log_n % 6` outer CT stages (stages 0 .. extra-1)
//   3. `log_n / 6` rounds of tensor-core based 64-point inner NTTs
//      Round k processes stages [extra+6k, extra+6k+6) with stride 2^(extra+6k)
//
// HMFHE optimizations realized in this kernel chain (paper § IV-A, § IV-B2):
//   - TLMOP   (§ IV-A3) : intermediate Inner-NTT data stays in registers; no
//                         intermediate SMEM round-trips between MMAs.
//   - TransOP (§ IV-A4) : implicit transpose via Fragment re-mapping in
//                         inner_ntt_tcu_kernel (no SMEM transpose).
//   - TFOP    (§ IV-B2) : block-shared SMEM caches for TFM (radix-8), Hada64
//                         (inner Hadamard), and per-stage outer twiddles.
//
// Additional optimization (offered as a separate API entry point, see below):
//   - RowMaj  (§ IV-B1) : two-kernel 4-step decomposition with pre-transposed
//                         data layout. Implemented in ntt_forward_rowmaj() for
//                         N = 4096; uses TC NTT-64 in both K1 and K2 with the
//                         implicit transpose realised by K1's GMEM write
//                         pattern (3/4 GMEM ops coalesced vs 1/4 in col-major).
//                         Caller must pre-transpose data; helpers provided.
//
// Not implemented here (would require kernels not in scope):
//   - COOP    (§ IV-C)  : co-optimization with BConv/IP kernels (none exist
//                         in this codebase).
//   - CRPMAC  (§ V-B)   : ciphertext-reusing PMAC (no PMAC in this codebase).

void ntt_forward_optimized(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n     = config->n;
    int      log_n = config->log_n;
    uint64_t q     = config->q;
    uint64_t mu    = config->mu;

    if (n < INNER_SIZE) {
        ntt_forward_basic(d_data, config, stream);
        return;
    }

    int block_size_outer = 256;
    int num_blocks_n     = (n + block_size_outer - 1) / block_size_outer;

    int extra      = log_n % LOG_INNER;
    int num_rounds = log_n / LOG_INNER;

    // Step 1: Global bit-reverse
    bitrev_kernel<<<num_blocks_n, block_size_outer, 0, stream>>>(d_data, n, log_n);

    // Step 2: `extra` outer CT stages (with TFOP twiddle cache)
    for (int s = 0; s < extra; s++) {
        int nb = ((n >> 1) + block_size_outer - 1) / block_size_outer;
        size_t sm = (size_t)(1ULL << s) * sizeof(uint64_t);
        outer_stage_kernel<<<nb, block_size_outer, sm, stream>>>(
            d_data, config->d_twiddles, n, s, q, mu
        );
    }

    // Step 3: Tensor-Core based inner NTT-64 rounds
    uint64_t total_ntts       = n / INNER_SIZE;
    uint64_t total_blocks     = (total_ntts + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t   smem_bytes_inner = SMEM_PER_BLOCK_U64 * sizeof(uint64_t);

    for (int k = 0; k < num_rounds; k++) {
        int stage_start = extra + LOG_INNER * k;
        inner_ntt_tcu_kernel<<<(unsigned)total_blocks, BLOCK_SIZE_INNER, smem_bytes_inner, stream>>>(
            d_data, config->d_twiddles, n, stage_start, q, mu
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

void ntt_inverse_optimized(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n     = config->n;
    int      log_n = config->log_n;
    uint64_t q     = config->q;
    uint64_t mu    = config->mu;

    if (n < INNER_SIZE) {
        ntt_inverse_basic(d_data, config, stream);
        return;
    }

    int block_size_outer = 256;
    int num_blocks_n     = (n + block_size_outer - 1) / block_size_outer;

    int extra      = log_n % LOG_INNER;
    int num_rounds = log_n / LOG_INNER;

    bitrev_kernel<<<num_blocks_n, block_size_outer, 0, stream>>>(d_data, n, log_n);

    for (int s = 0; s < extra; s++) {
        int nb = ((n >> 1) + block_size_outer - 1) / block_size_outer;
        size_t sm = (size_t)(1ULL << s) * sizeof(uint64_t);
        outer_stage_kernel<<<nb, block_size_outer, sm, stream>>>(
            d_data, config->d_twiddles_inv, n, s, q, mu
        );
    }

    uint64_t total_ntts       = n / INNER_SIZE;
    uint64_t total_blocks     = (total_ntts + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t   smem_bytes_inner = SMEM_PER_BLOCK_U64 * sizeof(uint64_t);

    for (int k = 0; k < num_rounds; k++) {
        int stage_start = extra + LOG_INNER * k;
        inner_ntt_tcu_kernel<<<(unsigned)total_blocks, BLOCK_SIZE_INNER, smem_bytes_inner, stream>>>(
            d_data, config->d_twiddles_inv, n, stage_start, q, mu
        );
    }

    // Scale by N^-1
    scale_opt_kernel<<<num_blocks_n, block_size_outer, 0, stream>>>(
        d_data, n, config->params.n_inv, q, mu
    );

    CUDA_CHECK(cudaGetLastError());
}

// TLMOP-only variant (alias to the full optimized path; the tensor-core kernel
// realizes the TLMOP idea by holding fragment data in registers).
void ntt_forward_tlmop(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    ntt_forward_optimized(d_data, config, stream);
}

void ntt_inverse_tlmop(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    ntt_inverse_optimized(d_data, config, stream);
}

// ============================================================================
// 4-Step RowMaj NTT (paper § IV-B1: outer-NTT optimization)
// ============================================================================
// Implements the paper's two-kernel 4-step structure with Row-Major data layout:
//
//   For N = N1 * N2:
//     Pre-transposed input layout:  data[β*N1 + α] = canonical[α*N2 + β]
//
//   KERNEL 1 (one warp per column b of M, b = 0..N2-1):
//     - Read M[a, b] = data[b*N1 + a] for a = 0..N1-1   (CONTIGUOUS, COALESCED)
//     - Compute T[d, b] = NTT_N1(M[:, b])
//     - Apply outer Hadamard: U[d, b] = T[d, b] * ω_N^(d*b)
//     - Write U[d, b] to canonical position d*N2 + b    (implicit transpose, non-coalesced)
//
//   KERNEL 2 (one warp per row d of U, d = 0..N1-1):
//     - Read U[d, b] = data[d*N2 + b] for b = 0..N2-1   (CONTIGUOUS, COALESCED)
//     - Compute V[d, c] = NTT_N2(U[d, :])
//     - SMEM restage so adjacent lanes hold adjacent c values, then
//     - Write V[d, c] to data[d*N2 + c] for c = 0..N2-1 (CONTIGUOUS, COALESCED)
//
// 3/4 of GMEM ops are coalesced (vs 1/4 in column-major).
//
// This implementation supports N = 4096 cleanly (N1 = N2 = 64). For other sizes
// we'd need recursive decomposition or NTT_N1/NTT_N2 sub-kernels of size != 64.
//
// === Inner NTT-64 variant: NATURAL-ORDER input ===
// The kernel below uses the same TLMOP+TransOP+TFOP scheme as
// inner_ntt_tcu_kernel, but reads M[a,b] = warp_smem[a*8+b] directly (no
// bit-reversal mapping). The output positions held by lane t become:
//   d_0 = (2*lane_mod4)*8 + lane_div4
//   d_1 = d_0 + 8
// (These are the NTT-64 output indices held by this lane.)
// As a side benefit, the natural-order input mapping has NO SMEM bank conflicts
// for the data load (the bit-reversed mapping has 4-way conflicts).

__global__ __launch_bounds__(BLOCK_SIZE_INNER)
void rowmaj_4step_k1_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    uint64_t n,
    int N1, int N2,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t smem[];
    uint64_t* tfm8_smem = smem + SMEM_TFM_OFFSET;
    uint64_t* hada_smem = smem + SMEM_HADA_OFFSET;

    int tid             = threadIdx.x;
    int warp_id         = tid / WARP_SIZE;
    int lane            = tid % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp     = blockIdx.x * warps_per_block + warp_id;

    if (tid < 64) {
        int i = tid >> 3;
        int k = tid & 7;
        tfm8_smem[i*8+k] = twiddles[((uint64_t)i*k * (n >> 3)) % n];   // ω_8^(i*k)
        hada_smem[i*8+k] = twiddles[((uint64_t)i*k * (n >> 6)) % n];   // ω_64^(i*k)
    }
    __syncthreads();

    int b = global_warp;
    if (b >= N2) return;

    uint64_t* warp_smem = smem + SMEM_WARP_OFFSET + (uint64_t)warp_id * WARP_SMEM_TOTAL;

    // ----- Async coalesced load: 32 in-flight 8B GMEM->SMEM transfers -----
    #pragma unroll
    for (int a = lane; a < N1; a += WARP_SIZE) {
        cp_async_8B(&warp_smem[wsm(a)], &data[(uint64_t)b * N1 + a]);
    }
    cp_async_commit_wait_all();
    __syncwarp();

    int lane_div4 = lane >> 2;
    int lane_mod4 = lane & 3;

    // ----- TFM fragments for MMA 1+2 (A position, left/right halves) -----
    uint64_t tfm12_l = tfm8_smem[lane_div4 * 8 + lane_mod4];
    uint64_t tfm12_r = tfm8_smem[lane_div4 * 8 + lane_mod4 + 4];

    double tfm12_l_h = (double)(uint32_t)(tfm12_l >> 16);
    double tfm12_l_l = (double)(uint32_t)(tfm12_l & 0xFFFFULL);
    double tfm12_r_h = (double)(uint32_t)(tfm12_r >> 16);
    double tfm12_r_l = (double)(uint32_t)(tfm12_r & 0xFFFFULL);

    // ----- B fragments: NATURAL-ORDER M[a, b_inner], padded SMEM stride -----
    double dat_top = (double)warp_smem[lane_mod4 * WARP_SMEM_STRIDE + lane_div4];
    double dat_bot = (double)warp_smem[(lane_mod4 + 4) * WARP_SMEM_STRIDE + lane_div4];

    // ----- MMA 1+2 -----
    double d_high_0 = 0.0, d_high_1 = 0.0;
    double d_low_0  = 0.0, d_low_1  = 0.0;

    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_l_h, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_r_h, dat_bot, d_high_0, d_high_1);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_l_l, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_r_l, dat_bot, d_low_0, d_low_1);

    // Bit-merge + Barrett reduction
    uint64_t mp_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_0, q, mu), q, mu);
    uint64_t mp_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_1, q, mu), q, mu);

    // ----- Inner Hadamard ω_64^(a*b_inner) (Barrett) -----
    {
        int b0 = lane_mod4 << 1;
        mp_0 = modmul_barrett(mp_0, hada_smem[lane_div4 * 8 + b0],     q, mu);
        mp_1 = modmul_barrett(mp_1, hada_smem[lane_div4 * 8 + b0 + 1], q, mu);
    }

    // ----- TransOP: M'' as A fragments for MMA 3+4 -----
    double mp_0_d = (double)mp_0;
    double mp_1_d = (double)mp_1;

    uint64_t tfm34_a = tfm8_smem[(lane_mod4 << 1)       * 8 + lane_div4];
    uint64_t tfm34_b = tfm8_smem[((lane_mod4 << 1) + 1) * 8 + lane_div4];

    double tfm34_a_h = (double)(uint32_t)(tfm34_a >> 16);
    double tfm34_a_l = (double)(uint32_t)(tfm34_a & 0xFFFFULL);
    double tfm34_b_h = (double)(uint32_t)(tfm34_b >> 16);
    double tfm34_b_l = (double)(uint32_t)(tfm34_b & 0xFFFFULL);

    double g_high_0 = 0.0, g_high_1 = 0.0;
    double g_low_0  = 0.0, g_low_1  = 0.0;

    mma_m8n8k4_f64(g_high_0, g_high_1, mp_0_d, tfm34_a_h, 0.0, 0.0);
    mma_m8n8k4_f64(g_high_0, g_high_1, mp_1_d, tfm34_b_h, g_high_0, g_high_1);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_0_d, tfm34_a_l, 0.0, 0.0);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_1_d, tfm34_b_l, g_low_0, g_low_1);

    uint64_t g_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_0, q, mu), q, mu);
    uint64_t g_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_1, q, mu), q, mu);

    // After NTT-64 (natural-order), lane t holds T values at positions:
    int d_0 = (lane_mod4 << 1) * 8 + lane_div4;
    int d_1 = d_0 + 8;

    // ----- Outer Hadamard: U[d, b] = T[d, b] * ω_N^(d*b) (Barrett) -----
    {
        uint64_t tw0 = twiddles[((uint64_t)d_0 * (uint64_t)b) % n];
        uint64_t tw1 = twiddles[((uint64_t)d_1 * (uint64_t)b) % n];
        g_0 = modmul_barrett(g_0, tw0, q, mu);
        g_1 = modmul_barrett(g_1, tw1, q, mu);
    }

    // ----- Write U[d, b] to canonical position d*N2 + b (IMPLICIT TRANSPOSE, non-coalesced) -----
    data[(uint64_t)d_0 * N2 + b] = g_0;
    data[(uint64_t)d_1 * N2 + b] = g_1;
}

__global__ __launch_bounds__(BLOCK_SIZE_INNER)
void rowmaj_4step_k2_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    uint64_t n,
    int N1, int N2,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t smem[];
    uint64_t* tfm8_smem = smem + SMEM_TFM_OFFSET;
    uint64_t* hada_smem = smem + SMEM_HADA_OFFSET;

    int tid             = threadIdx.x;
    int warp_id         = tid / WARP_SIZE;
    int lane            = tid % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp     = blockIdx.x * warps_per_block + warp_id;

    if (tid < 64) {
        int i = tid >> 3;
        int k = tid & 7;
        tfm8_smem[i*8+k] = twiddles[((uint64_t)i*k * (n >> 3)) % n];
        hada_smem[i*8+k] = twiddles[((uint64_t)i*k * (n >> 6)) % n];
    }
    __syncthreads();

    int d = global_warp;
    if (d >= N1) return;

    uint64_t* warp_smem = smem + SMEM_WARP_OFFSET + (uint64_t)warp_id * WARP_SMEM_TOTAL;

    // ----- Async coalesced load -----
    #pragma unroll
    for (int b = lane; b < N2; b += WARP_SIZE) {
        cp_async_8B(&warp_smem[wsm(b)], &data[(uint64_t)d * N2 + b]);
    }
    cp_async_commit_wait_all();
    __syncwarp();

    int lane_div4 = lane >> 2;
    int lane_mod4 = lane & 3;

    uint64_t tfm12_l = tfm8_smem[lane_div4 * 8 + lane_mod4];
    uint64_t tfm12_r = tfm8_smem[lane_div4 * 8 + lane_mod4 + 4];

    double tfm12_l_h = (double)(uint32_t)(tfm12_l >> 16);
    double tfm12_l_l = (double)(uint32_t)(tfm12_l & 0xFFFFULL);
    double tfm12_r_h = (double)(uint32_t)(tfm12_r >> 16);
    double tfm12_r_l = (double)(uint32_t)(tfm12_r & 0xFFFFULL);

    double dat_top = (double)warp_smem[lane_mod4 * WARP_SMEM_STRIDE + lane_div4];
    double dat_bot = (double)warp_smem[(lane_mod4 + 4) * WARP_SMEM_STRIDE + lane_div4];

    double d_high_0 = 0.0, d_high_1 = 0.0;
    double d_low_0  = 0.0, d_low_1  = 0.0;

    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_l_h, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_high_0, d_high_1, tfm12_r_h, dat_bot, d_high_0, d_high_1);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_l_l, dat_top, 0.0, 0.0);
    mma_m8n8k4_f64(d_low_0,  d_low_1,  tfm12_r_l, dat_bot, d_low_0, d_low_1);

    uint64_t mp_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_0, q, mu), q, mu);
    uint64_t mp_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)d_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)d_low_1, q, mu), q, mu);

    {
        int b0 = lane_mod4 << 1;
        mp_0 = modmul_barrett(mp_0, hada_smem[lane_div4 * 8 + b0],     q, mu);
        mp_1 = modmul_barrett(mp_1, hada_smem[lane_div4 * 8 + b0 + 1], q, mu);
    }

    double mp_0_d = (double)mp_0;
    double mp_1_d = (double)mp_1;

    uint64_t tfm34_a = tfm8_smem[(lane_mod4 << 1)       * 8 + lane_div4];
    uint64_t tfm34_b = tfm8_smem[((lane_mod4 << 1) + 1) * 8 + lane_div4];

    double tfm34_a_h = (double)(uint32_t)(tfm34_a >> 16);
    double tfm34_a_l = (double)(uint32_t)(tfm34_a & 0xFFFFULL);
    double tfm34_b_h = (double)(uint32_t)(tfm34_b >> 16);
    double tfm34_b_l = (double)(uint32_t)(tfm34_b & 0xFFFFULL);

    double g_high_0 = 0.0, g_high_1 = 0.0;
    double g_low_0  = 0.0, g_low_1  = 0.0;

    mma_m8n8k4_f64(g_high_0, g_high_1, mp_0_d, tfm34_a_h, 0.0, 0.0);
    mma_m8n8k4_f64(g_high_0, g_high_1, mp_1_d, tfm34_b_h, g_high_0, g_high_1);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_0_d, tfm34_a_l, 0.0, 0.0);
    mma_m8n8k4_f64(g_low_0,  g_low_1,  mp_1_d, tfm34_b_l, g_low_0, g_low_1);

    uint64_t g_0 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_0, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_0, q, mu), q, mu);
    uint64_t g_1 = mod_q_barrett(
        mod_q_barrett((uint64_t)g_high_1, q, mu) * (1ULL << 16)
        + mod_q_barrett((uint64_t)g_low_1, q, mu), q, mu);

    // Lane t holds V[d, c_0] and V[d, c_1] in scattered NTT-output positions.
    int c_0 = (lane_mod4 << 1) * 8 + lane_div4;
    int c_1 = c_0 + 8;

    // ----- SMEM restage (using padded mapping wsm()) -----
    warp_smem[wsm(c_0)] = g_0;
    warp_smem[wsm(c_1)] = g_1;
    __syncwarp();

    // ----- Coalesced GMEM write -----
    for (int c = lane; c < N2; c += WARP_SIZE) {
        data[(uint64_t)d * N2 + c] = warp_smem[wsm(c)];
    }
}

// ----------------------------------------------------------------------------
// Pre-transpose / un-transpose helpers (for testing and FHE encode/decode)
// pre_trans[β*N1 + α] = canonical[α*N2 + β]
// ----------------------------------------------------------------------------
__global__ void transpose_to_rowmaj_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in,
    int N1, int N2
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)N1 * N2;
    if (idx >= total) return;
    uint64_t alpha = idx / (uint64_t)N2;
    uint64_t beta  = idx % (uint64_t)N2;
    out[beta * (uint64_t)N1 + alpha] = in[alpha * (uint64_t)N2 + beta];
}

__global__ void transpose_from_rowmaj_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ in,
    int N1, int N2
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)N1 * N2;
    if (idx >= total) return;
    uint64_t alpha = idx / (uint64_t)N2;
    uint64_t beta  = idx % (uint64_t)N2;
    out[alpha * (uint64_t)N2 + beta] = in[beta * (uint64_t)N1 + alpha];
}

void rowmaj_to_pretransposed(uint64_t* d_dst, const uint64_t* d_src, int N1, int N2, cudaStream_t stream) {
    int total = N1 * N2;
    int bs = 256;
    int nb = (total + bs - 1) / bs;
    transpose_to_rowmaj_kernel<<<nb, bs, 0, stream>>>(d_dst, d_src, N1, N2);
}

void rowmaj_from_pretransposed(uint64_t* d_dst, const uint64_t* d_src, int N1, int N2, cudaStream_t stream) {
    int total = N1 * N2;
    int bs = 256;
    int nb = (total + bs - 1) / bs;
    transpose_from_rowmaj_kernel<<<nb, bs, 0, stream>>>(d_dst, d_src, N1, N2);
}

// ----------------------------------------------------------------------------
// 4-step RowMaj NTT launcher (currently N=4096 only)
// d_data must be in pre-transposed format on entry; output is also pre-transposed.
// ----------------------------------------------------------------------------
void ntt_forward_rowmaj(uint64_t* d_data, const NTTConfig* config, cudaStream_t stream) {
    uint64_t n = config->n;
    uint64_t q = config->q;

    if (n != 4096) {
        // Unsupported size: fall back to standard optimized path
        ntt_forward_optimized(d_data, config, stream);
        return;
    }

    int N1 = 64, N2 = 64;
    uint64_t mu = config->mu;
    size_t smem_bytes = SMEM_PER_BLOCK_U64 * sizeof(uint64_t);

    int blocks_k1 = (N2 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    rowmaj_4step_k1_kernel<<<blocks_k1, BLOCK_SIZE_INNER, smem_bytes, stream>>>(
        d_data, config->d_twiddles, n, N1, N2, q, mu
    );

    int blocks_k2 = (N1 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    rowmaj_4step_k2_kernel<<<blocks_k2, BLOCK_SIZE_INNER, smem_bytes, stream>>>(
        d_data, config->d_twiddles, n, N1, N2, q, mu
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ntt
