// ============================================================================
// timing_compare.cu - Comprehensive timing comparison for K-limb NTT
// ============================================================================
// Compares:
//   1. MMA optimized (TCU radix-64 with tensor cores)
//   2. Scalar radix-64 (same algorithm, no tensor cores)
//   3. Normal CUDA radix-2 (per-stage CT kernels)
// For both K=2 (50-bit prime) and K=11 (BN254 254-bit prime)
// ============================================================================

#include "ntt_arbitrary.cuh"
#include "ntt_arbitrary_radix64.cuh"
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#define CU_CHECK(call) do {                                                    \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA error " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct Bench { double us_min, us_med, us_avg; };

template <typename Fn>
static Bench time_kernel(Fn&& fn, int warmup, int iters) {
    cudaEvent_t s, e;
    CU_CHECK(cudaEventCreate(&s));
    CU_CHECK(cudaEventCreate(&e));

    for (int i = 0; i < warmup; i++) fn();
    CU_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples(iters);
    for (int i = 0; i < iters; i++) {
        CU_CHECK(cudaDeviceSynchronize());
        CU_CHECK(cudaEventRecord(s));
        fn();
        CU_CHECK(cudaEventRecord(e));
        CU_CHECK(cudaDeviceSynchronize());
        CU_CHECK(cudaEventElapsedTime(&samples[i], s, e));
    }
    std::sort(samples.begin(), samples.end());

    double sum = 0;
    for (float v : samples) sum += v;
    Bench b;
    b.us_min = samples.front()    * 1000.0;
    b.us_med = samples[iters / 2] * 1000.0;
    b.us_avg = (sum / iters)      * 1000.0;
    CU_CHECK(cudaEventDestroy(s));
    CU_CHECK(cudaEventDestroy(e));
    return b;
}

// Re-upload fresh Montgomery-form data
template <int K>
static void reupload_mont_input(ntt_arb::NTTBuffers<K>& B) {
    std::vector<ntt_arb::DScalar<K>> data_d(B.n);
    for (int i = 0; i < B.n; i++) {
        ntt_arb::BigInt<K> dx = ntt_arb::mont_mul_big<K>(B.data_host[i], B.R2, B.prime, B.np);
        data_d[i] = ntt_arb::load_scalar<K>(dx);
    }
    cudaMemcpy(B.d_data, data_d.data(), (size_t)B.n * sizeof(ntt_arb::DScalar<K>), cudaMemcpyHostToDevice);
}

template <int K>
static void run_benchmark_for_K(const std::vector<int>& log_sizes, int warmup, int iters) {
    using namespace ntt_arb;
    using FC = FieldConfig<K>;

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "K = " << K << "  (LIMB_BITS=" << LIMB_BITS 
              << ", " << K * LIMB_BITS << "-bit budget)\n";
    std::cout << "============================================================\n";

    // Print prime info
    BigInt<K> p = FC::prime();
    std::cout << "Prime p = 0x";
    bool started = false;
    for (int i = K - 1; i >= 0; i--) {
        if (started)           std::cout << std::hex << std::setw(7) << std::setfill('0') << p.limbs[i];
        else if (p.limbs[i]) { std::cout << std::hex << p.limbs[i]; started = true; }
    }
    std::cout << std::dec << std::setfill(' ') << "\n\n";

    // Table header
    std::cout << std::left  << std::setw(10) << "N"
              << std::right
              << std::setw(12) << "MMA"
              << std::setw(12) << "Radix-64"
              << std::setw(12) << "Radix-2"
              << std::setw(10) << "MMA vs"
              << std::setw(10) << "R64 vs"
              << "\n"
              << std::left  << std::setw(10) << ""
              << std::right
              << std::setw(12) << "(us)"
              << std::setw(12) << "(us)"
              << std::setw(12) << "(us)"
              << std::setw(10) << "R2"
              << std::setw(10) << "R2"
              << "\n"
              << std::string(64, '-') << "\n";

    for (int log_n : log_sizes) {
        if (log_n > FC::MAX_LOG_N) {
            std::cout << std::left << std::setw(10) << (1ULL << log_n)
                      << " (skipped: log_n > MAX_LOG_N=" << FC::MAX_LOG_N << ")\n";
            continue;
        }

        uint64_t N = 1ULL << log_n;
        NTTBuffers<K> B = setup_ntt<K>(log_n, 0xDEADBEEFULL ^ (uint64_t)log_n);
        Radix64Buffers<K> R = setup_radix64_tables<K>(B);

        // ---- MMA optimized (TCU path) ----
        reupload_mont_input<K>(B);
        Bench mma_bench = time_kernel([&]{
            if (B.d_tfm8 && B.d_hada64) {
                run_forward_ct_tcu<K>(B.d_data, B.d_tw, B.d_tfm8, B.d_hada64,
                                      B.n, log_n, B.prime_d, B.np);
            } else {
                run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
            }
        }, warmup, iters);

        // ---- Scalar Radix-64 (no tensor cores) ----
        reupload_mont_input<K>(B);
        Bench r64_bench = time_kernel([&]{
            if (R.d_omega8 && R.d_omega64) {
                run_forward_radix64_scalar<K>(B.d_data, B.d_tw, R.d_omega8, R.d_omega64,
                                              B.n, log_n, B.prime_d, B.np);
            } else {
                run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
            }
        }, warmup, iters);

        // ---- Normal CUDA radix-2 (per-stage CT kernels) ----
        reupload_mont_input<K>(B);
        Bench r2_bench = time_kernel([&]{
            int t = 256;
            int b_br = (B.n + t - 1) / t;
            bitrev_kernel_d<K><<<b_br, t>>>(B.d_data, B.n, log_n);
            int half = B.n / 2;
            int b_st = (half + t - 1) / t;
            for (int st = 0; st < log_n; st++) {
                ct_stage_kernel<K><<<b_st, t>>>(B.d_data, B.d_tw, B.n, st, B.prime_d, B.np);
            }
        }, warmup, iters);

        // Speedup calculations
        double mma_vs_r2 = r2_bench.us_med / mma_bench.us_med;
        double r64_vs_r2 = r2_bench.us_med / r64_bench.us_med;

        std::cout << std::left  << std::setw(10) << N
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << mma_bench.us_med
                  << std::setw(12) << r64_bench.us_med
                  << std::setw(12) << r2_bench.us_med
                  << std::setw(9) << mma_vs_r2 << "x"
                  << std::setw(9) << r64_vs_r2 << "x"
                  << "\n";
        std::cout.flush();

        teardown_radix64_tables<K>(R);
        teardown_ntt<K>(B);
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CU_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "============================================================\n"
              << "GPU         : " << prop.name
              << "  (sm_" << prop.major << prop.minor
              << ", " << prop.multiProcessorCount << " SMs)\n"
              << "============================================================\n";

    // Parse command line for sizes, or use defaults
    std::vector<int> log_sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) log_sizes.push_back(std::atoi(argv[i]));
    } else {
        log_sizes = {6, 8, 10, 12, 14, 16, 18};
    }

    constexpr int WARMUP = 5;
    constexpr int ITERS  = 30;

    // ========================================================================
    // K=2 benchmark (50-bit prime)
    // ========================================================================
    run_benchmark_for_K<2>(log_sizes, WARMUP, ITERS);

    // ========================================================================
    // K=11 benchmark (BN254 254-bit prime)
    // ========================================================================
    run_benchmark_for_K<11>(log_sizes, WARMUP, ITERS);

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "LEGEND:\n";
    std::cout << "  MMA:      FP64 tensor core radix-64 (fused inner stages)\n";
    std::cout << "  Radix-64: Scalar CUDA radix-64 (same algorithm, no TCU)\n";
    std::cout << "  Radix-2:  Per-stage Cooley-Tukey (one kernel per stage)\n";
    std::cout << "\n";
    std::cout << "  MMA vs R2:  Speedup of MMA over Radix-2 (>1 = MMA faster)\n";
    std::cout << "  R64 vs R2:  Speedup of Radix-64 over Radix-2 (>1 = R64 faster)\n";
    std::cout << "============================================================\n";

    return 0;
}
