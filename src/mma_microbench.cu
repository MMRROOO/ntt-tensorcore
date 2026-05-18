// =============================================================================
// mma_microbench.cu
// -----------------------------------------------------------------------------
// Empirically settles the question "should we route the K-limb modmul partial
// product through the FP64 tensor cores?".
//
// We compare two implementations of the K*K = 121 FP64 multiply-accumulates
// that make up one Montgomery partial-product matrix for K=11:
//
//   1. mma path:    one warp produces the partial products via four
//                   mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 calls.
//                   1/4 inner-dim utilization is wasted (rank-1 outer
//                   product on a contraction operator); the warp is then
//                   ENTIRELY consumed by one modmul's worth of work.
//
//   2. scalar path: each of the 32 lanes computes its own independent
//                   K*K partial product matrix using regular FP64 multiplies.
//                   32 modmuls in flight per warp.
//
// Both kernels do the same total work (121 useful FP64 muls per modmul) but
// the mma path can only execute one modmul per warp at a time, whereas the
// scalar path can execute 32. We report time per modmul for each path so
// the comparison is apples-to-apples regardless of warp-count.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
                cudaGetErrorString(err__), __FILE__, __LINE__);                \
        exit(1);                                                               \
    }                                                                          \
} while (0)
#endif

constexpr int K = 11;

__device__ __forceinline__ void mma_m8n8k4_f64(
    double& d0, double& d1, double a, double b, double c0, double c1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1},{%2},{%3},{%4,%5};\n"
        : "=d"(d0), "=d"(d1)
        : "d"(a), "d"(b), "d"(c0), "d"(c1));
#else
    d0 = c0; d1 = c1;
#endif
}

// -----------------------------------------------------------------------------
// MMA path. Each warp computes ONE 11x11 outer product per iteration.
// 11 padded to 16 -> 2x2 = 4 mma.sync calls covering 8x8 blocks each.
// Lane t in [0,32) sees A[t/4][t%4] and B[t%4][t/4]; we treat the inner-dim
// k = t%4 of A and t%4 of B as "rank-1": only k=0 carries data, k>0 are zero.
// Each lane gets two D outputs which accumulate into an FP64 sink to keep the
// compiler from eliding the work.
// -----------------------------------------------------------------------------
__launch_bounds__(128, 4)
__global__ void mma_modmul_warp(double* sink, const double* a_lim,
                                 const double* b_lim, int iters) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Per-warp inputs (one a/b set per warp -- each warp does an independent
    // sequence of mma calls so the work can't be coalesced across warps).
    int warp_global = blockIdx.x * (blockDim.x >> 5) + warp;
    double a[16], b[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        a[i] = (i < K) ? a_lim[(warp_global * 16 + i) & 0xffff] : 0.0;
        b[i] = (i < K) ? b_lim[(warp_global * 16 + i) & 0xffff] : 0.0;
    }

    double acc0 = 0.0, acc1 = 0.0;
    // Make inputs depend on (it, lane) so the compiler can't hoist them
    // outside the loop. We pipe the running accumulator back in too.
    for (int it = 0; it < iters; it++) {
        double scale = (double)(it & 0xff) + 1.0;
        #pragma unroll
        for (int bi = 0; bi < 2; bi++) {
            #pragma unroll
            for (int bj = 0; bj < 2; bj++) {
                int row = bi * 8 + (lane >> 2);
                int col = bj * 8 + (lane >> 2);
                double a_val = ((lane & 3) == 0) ? (a[row] * scale + acc0) : 0.0;
                double b_val = ((lane & 3) == 0) ? (b[col] * scale + acc1) : 0.0;
                double d0, d1;
                mma_m8n8k4_f64(d0, d1, a_val, b_val, acc0, acc1);
                acc0 = d0; acc1 = d1;
            }
        }
    }
    sink[blockIdx.x * blockDim.x + tid] = acc0 + acc1;
}

// -----------------------------------------------------------------------------
// Scalar path. Each lane independently computes a K*K partial product matrix.
// 32 modmuls per warp in flight (vs 1 per warp for the mma path).
// -----------------------------------------------------------------------------
__launch_bounds__(128, 4)
__global__ void scalar_modmul_perlane(double* sink, const double* a_lim,
                                       const double* b_lim, int iters) {
    int tid = threadIdx.x;
    int global_lane = blockIdx.x * blockDim.x + tid;
    double a[K], b[K];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        a[i] = a_lim[(global_lane * 16 + i) & 0xffff];
        b[i] = b_lim[(global_lane * 16 + i) & 0xffff];
    }
    double acc = 0.0;
    for (int it = 0; it < iters; it++) {
        double scale = (double)(it & 0xff) + 1.0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            #pragma unroll
            for (int j = 0; j < K; j++) {
                acc += (a[i] * scale + acc) * (b[j] * scale);
            }
        }
    }
    sink[global_lane] = acc;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("============================================================\n");
    printf("GPU: %s  (sm_%d%d, %d SMs)\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount);
    printf("Microbench: K=%d FP64 partial-product matrix per modmul\n", K);
    printf("============================================================\n");

    const int NUM_BLOCKS = prop.multiProcessorCount * 16;
    const int THREADS    = 128;
    const int ITERS      = 1024;

    std::vector<double> h_a(NUM_BLOCKS * 16, 0.0);
    std::vector<double> h_b(NUM_BLOCKS * 16, 0.0);
    for (size_t i = 0; i < h_a.size(); i++) {
        h_a[i] = (double)(i & 0xfffff);
        h_b[i] = (double)((i * 7) & 0xfffff);
    }
    double *d_a = nullptr, *d_b = nullptr, *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sink, NUM_BLOCKS * THREADS * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // Warmup
    scalar_modmul_perlane<<<NUM_BLOCKS, THREADS>>>(d_sink, d_a, d_b, 16);
    mma_modmul_warp     <<<NUM_BLOCKS, THREADS>>>(d_sink, d_a, d_b, 16);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Scalar path ----
    cudaEventRecord(e0);
    scalar_modmul_perlane<<<NUM_BLOCKS, THREADS>>>(d_sink, d_a, d_b, ITERS);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms_scalar = 0.f;
    cudaEventElapsedTime(&ms_scalar, e0, e1);
    // 1 modmul per lane per iter
    size_t scalar_modmuls = (size_t)NUM_BLOCKS * THREADS * ITERS;

    // ---- MMA path ----
    cudaEventRecord(e0);
    mma_modmul_warp<<<NUM_BLOCKS, THREADS>>>(d_sink, d_a, d_b, ITERS);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms_mma = 0.f;
    cudaEventElapsedTime(&ms_mma, e0, e1);
    // 1 modmul per WARP per iter (mma path is warp-cooperative)
    size_t mma_modmuls = (size_t)NUM_BLOCKS * (THREADS / 32) * ITERS;

    double ns_per_modmul_scalar = (double)ms_scalar * 1e6 / (double)scalar_modmuls;
    double ns_per_modmul_mma    = (double)ms_mma    * 1e6 / (double)mma_modmuls;

    printf("\nPer-modmul cost (partial-product matrix only, NO Montgomery reduction):\n");
    printf("  scalar (per-lane):  %7.3f ns/modmul   (%zu modmuls in %.2f ms)\n",
           ns_per_modmul_scalar, scalar_modmuls, ms_scalar);
    printf("  mma    (per-warp):  %7.3f ns/modmul   (%zu modmuls in %.2f ms)\n",
           ns_per_modmul_mma, mma_modmuls, ms_mma);
    printf("  ratio  mma/scalar:  %5.2fx\n", ns_per_modmul_mma / ns_per_modmul_scalar);

    printf("\nNotes:\n");
    printf("  * scalar path runs 32 modmuls in flight per warp (one per lane).\n");
    printf("  * mma path runs 1 modmul per warp; the m8n8k4 contraction is at\n");
    printf("    1/4 utilization because we use it for a rank-1 outer product.\n");
    printf("  * If the ratio > 1, the mma path is SLOWER per modmul and is not\n");
    printf("    a useful replacement for the scalar inner of mont_mul<K>.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sink);
    return 0;
}
