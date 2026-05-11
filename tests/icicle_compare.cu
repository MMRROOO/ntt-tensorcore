// ============================================================================
// icicle_compare: APPLES-TO-APPLES head-to-head between this project's
//                 TC-MMA NTT and Ingonyama's ICICLE NTT, both running:
//
//   * same GPU                      (cudaGetDeviceProperties on slot 0)
//   * same finite field             (BabyBear, q = 15 * 2^27 + 1 = 2013265921)
//   * same N values                 (configurable below)
//   * same input data               (RNG-seeded, byte-identical)
//   * same warmup / iter counts     (10 / 100 by default)
//   * cuda-event timing per iter    (kernel-only, no H2D/D2H in the loop)
//   * forward NTT, in-domain only   (we do NOT include domain init / setup)
//
// Build:
//   cd build
//   cmake -DICICLE_ENABLED=ON -DICICLE_INSTALL_DIR=$HOME/git-repos/icicle-install/icicle ..
//   make -j icicle_compare
//
// Run:
//   ICICLE_BACKEND_INSTALL_DIR=$HOME/git-repos/icicle-install/icicle/lib/backend \
//     ./icicle_compare
// ============================================================================

#include "ntt.cuh"

// ---- ICICLE ----------------------------------------------------------------
#include "icicle/runtime.h"
#include "icicle/ntt.h"
#include "icicle/backend/ntt_config.h"
#include "icicle/fields/stark_fields/babybear.h"

// ---- stdlib / CUDA ---------------------------------------------------------
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// We deliberately do NOT `using namespace ntt;` here -- it collides with
// icicle::NTTConfig / icicle::ntt(). Qualify our project's symbols explicitly.
using icicle::Device;
using icicle::eIcicleError;
using icicle::ConfigExtension;
using icicle::Ordering;
using icicle::NTTDir;
using babybear::scalar_t;          // BabyBear field element

#define CU_CHECK(call) do {                                                    \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA error " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// Note: icicle/errors.h already defines its own ICICLE_CHECK that throws.
// Use a local name so we just std::exit on failure.
#define IC_CHECK(call) do {                                                    \
    eIcicleError e = (call);                                                   \
    if (e != eIcicleError::SUCCESS) {                                          \
        std::cerr << "ICICLE error " << (int)e                                 \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct Bench { double us_min, us_med, us_avg; };

// Measures kernel-only time using cuda events with a per-iter device-wide
// sync. The extra cudaDeviceSynchronize() guarantees correctness even when
// the timed kernel launches into its own stream pool (ICICLE does this --
// without the device sync, cudaEventRecord on the default stream can miss
// a chunk of the kernel's actual GPU work, producing artifactually low
// min times). Both kernels go through the exact same harness so the
// comparison stays apples-to-apples.
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

int main(int argc, char** argv) {
    // ---- GPU info -----------------------------------------------------------
    cudaDeviceProp prop;
    CU_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "============================================================\n"
              << "GPU         : " << prop.name
              << "  (sm_" << prop.major << prop.minor
              << ", " << prop.multiProcessorCount << " SMs)\n"
              << "Field       : BabyBear  q = 2013265921 = 15 * 2^27 + 1\n"
              << "MMA project : DEFAULT_PRIME = " << ntt::DEFAULT_PRIME << "\n"
              << "============================================================\n\n";
    if (ntt::DEFAULT_PRIME != 2013265921ULL) {
        std::cerr << "DEFAULT_PRIME does not match BabyBear; aborting.\n";
        return 1;
    }

    // ---- Initialize ICICLE backend -----------------------------------------
    IC_CHECK(icicle_load_backend_from_env_or_default());
    Device dev_cuda{"CUDA", 0};
    if (icicle_is_device_available("CUDA") != eIcicleError::SUCCESS) {
        std::cerr << "ICICLE: CUDA device not available\n";
        return 1;
    }
    IC_CHECK(icicle_set_device(dev_cuda));

    // ---- Sizes to benchmark -------------------------------------------------
    std::vector<uint64_t> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            sizes.push_back(1ULL << std::atoi(argv[i]));
        }
    } else {
        for (int log_n : {12, 14, 16, 17, 18, 19, 20}) {
            sizes.push_back(1ULL << log_n);
        }
    }
    int max_log_n = 0;
    for (auto N : sizes) {
        int l = 0; for (uint64_t t = N; t > 1; t >>= 1) l++;
        if (l > max_log_n) max_log_n = l;
    }

    // ---- One-time NTT-domain init for ICICLE (NOT in timed loop) -----------
    {
        scalar_t basic_root = scalar_t::omega(max_log_n);
        auto domain_cfg = icicle::default_ntt_init_domain_config();
        ConfigExtension ext;
        ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
        domain_cfg.ext = &ext;
        IC_CHECK(icicle::ntt_init_domain<scalar_t>(basic_root, domain_cfg));
    }

    // ---- Header ------------------------------------------------------------
    std::cout << std::left << std::setw(11) << "N"
              << std::right
              << std::setw(12) << "MMA min"
              << std::setw(12) << "MMA med"
              << std::setw(12) << "MMA avg"
              << std::setw(13) << "ICICLE min"
              << std::setw(13) << "ICICLE med"
              << std::setw(13) << "ICICLE avg"
              << std::setw(12) << "med ratio"
              << "\n"
              << std::left << std::setw(11) << ""
              << std::right
              << std::setw(12) << "(us)" << std::setw(12) << "(us)" << std::setw(12) << "(us)"
              << std::setw(13) << "(us)" << std::setw(13) << "(us)" << std::setw(13) << "(us)"
              << std::setw(12) << "ICICLE/MMA"
              << "\n"
              << std::string(98, '-') << "\n";

    constexpr int WARMUP = 10;
    constexpr int ITERS  = 100;

    for (uint64_t N : sizes) {
        // ---- Identical RNG-seeded random input shared by both ----------------
        std::mt19937 rng(0xDEADBEEFu ^ static_cast<uint32_t>(N));
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(ntt::DEFAULT_PRIME) - 1u);

        std::vector<uint64_t>  h_in_u64(N);
        std::vector<scalar_t>  h_in_sc (N);
        for (uint64_t i = 0; i < N; i++) {
            uint32_t v = dist(rng);
            h_in_u64[i] = static_cast<uint64_t>(v);
            h_in_sc [i] = scalar_t::from(v);
        }

        // ====================================================================
        //                              OUR MMA NTT
        // ====================================================================
        ntt::NTTConfig* cfg_ours = ntt::ntt_init(N, ntt::DEFAULT_PRIME);
        uint64_t* d_data_ours = nullptr;
        CU_CHECK(cudaMalloc(&d_data_ours, N * sizeof(uint64_t)));
        CU_CHECK(cudaMemcpy(d_data_ours, h_in_u64.data(),
                            N * sizeof(uint64_t), cudaMemcpyHostToDevice));

        Bench bo = time_kernel([&]{
            ntt::ntt_forward_optimized(d_data_ours, cfg_ours);
        }, WARMUP, ITERS);

        CU_CHECK(cudaFree(d_data_ours));
        ntt::ntt_cleanup(cfg_ours);

        // ====================================================================
        //                              ICICLE NTT
        // ====================================================================
        scalar_t* d_in_sc  = nullptr;
        scalar_t* d_out_sc = nullptr;
        CU_CHECK(cudaMalloc(&d_in_sc,  N * sizeof(scalar_t)));
        CU_CHECK(cudaMalloc(&d_out_sc, N * sizeof(scalar_t)));
        CU_CHECK(cudaMemcpy(d_in_sc, h_in_sc.data(),
                            N * sizeof(scalar_t), cudaMemcpyHostToDevice));

        icicle::NTTConfig<scalar_t> cfg_icicle = icicle::default_ntt_config<scalar_t>();
        cfg_icicle.are_inputs_on_device  = true;
        cfg_icicle.are_outputs_on_device = true;
        cfg_icicle.batch_size            = 1;
        // Same convention our optimized kernel uses (bit-reversed output) so
        // ICICLE isn't penalized for an extra reorder pass it doesn't need.
        cfg_icicle.ordering              = Ordering::kNR;

        Bench bi = time_kernel([&]{
            IC_CHECK(icicle::ntt(d_in_sc, static_cast<int>(N), NTTDir::kForward,
                                 cfg_icicle, d_out_sc));
        }, WARMUP, ITERS);

        CU_CHECK(cudaFree(d_in_sc));
        CU_CHECK(cudaFree(d_out_sc));

        double ratio_med = (bo.us_med > 0.0) ? (bi.us_med / bo.us_med) : 0.0;

        std::cout << std::left  << std::setw(11) << N
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << bo.us_min
                  << std::setw(12) << bo.us_med
                  << std::setw(12) << bo.us_avg
                  << std::setw(13) << bi.us_min
                  << std::setw(13) << bi.us_med
                  << std::setw(13) << bi.us_avg
                  << std::setw(11) << ratio_med << "x"
                  << "\n";
        std::cout.flush();
    }

    std::cout << "\nmed ratio = ICICLE_med / MMA_med  (>1 means our MMA kernel is faster).\n"
              << "Both kernels go through the same RNG-seeded input, same N, same timing\n"
              << "harness (cudaEvent + per-iter cudaDeviceSynchronize). Domain init for\n"
              << "ICICLE is performed once outside the timed region.\n";

    IC_CHECK(icicle::ntt_release_domain<scalar_t>());
    return 0;
}
