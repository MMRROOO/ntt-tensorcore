#include "ntt.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace ntt;

template<typename Fn>
double bench_ms(Fn&& fn, int iters = 200) {
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / iters;
}

int main() {
    const uint64_t n = 4096;
    const int N1 = 64, N2 = 64;

    std::cout << "=== RowMaj 4-step NTT correctness test (N=" << n << ") ===\n";

    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);

    std::vector<uint64_t> input(n);
    for (uint64_t i = 0; i < n; i++) input[i] = i + 1;

    // Host reference
    std::vector<uint64_t> host_result = input;
    ntt_host(host_result, config->params, false);

    // GPU reference: existing optimized path
    std::vector<uint64_t> gpu_opt(n);
    {
        uint64_t* d;
        cudaMalloc(&d, n * sizeof(uint64_t));
        cudaMemcpy(d, input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        ntt_forward_optimized(d, config);
        cudaDeviceSynchronize();
        cudaMemcpy(gpu_opt.data(), d, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d);
    }

    // GPU RowMaj 4-step: canonical -> pre-transposed -> NTT -> pre-transposed -> canonical
    std::vector<uint64_t> gpu_rowmaj(n);
    {
        uint64_t *d_canonical, *d_pre;
        cudaMalloc(&d_canonical, n * sizeof(uint64_t));
        cudaMalloc(&d_pre, n * sizeof(uint64_t));
        cudaMemcpy(d_canonical, input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);

        rowmaj_to_pretransposed(d_pre, d_canonical, N1, N2);
        ntt_forward_rowmaj(d_pre, config);
        rowmaj_from_pretransposed(d_canonical, d_pre, N1, N2);

        cudaDeviceSynchronize();
        cudaMemcpy(gpu_rowmaj.data(), d_canonical, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_canonical);
        cudaFree(d_pre);
    }

    int direct_match_opt = 0, direct_match_rowmaj = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (gpu_opt[i] == host_result[i]) direct_match_opt++;
        if (gpu_rowmaj[i] == host_result[i]) direct_match_rowmaj++;
    }
    std::cout << "Optimized vs host: " << direct_match_opt << " / " << n
              << (direct_match_opt == (int)n ? "  PASS\n" : "  FAIL\n");
    std::cout << "RowMaj    vs host: " << direct_match_rowmaj << " / " << n
              << (direct_match_rowmaj == (int)n ? "  PASS\n" : "  FAIL\n");

    if (direct_match_rowmaj < (int)n) {
        std::cout << "\nFirst 10 mismatches (RowMaj vs host):\n";
        int shown = 0;
        for (uint64_t i = 0; i < n && shown < 10; i++) {
            if (gpu_rowmaj[i] != host_result[i]) {
                std::cout << "  [" << i << "] expected " << host_result[i]
                          << ", got " << gpu_rowmaj[i] << "\n";
                shown++;
            }
        }
        ntt_cleanup(config);
        return 1;
    }

    // ----------- Performance comparison -----------
    std::cout << "\n=== Performance (median over 200 iters) ===\n";

    uint64_t *d_in_canonical, *d_in_pre, *d_work;
    cudaMalloc(&d_in_canonical, n * sizeof(uint64_t));
    cudaMalloc(&d_in_pre, n * sizeof(uint64_t));
    cudaMalloc(&d_work, n * sizeof(uint64_t));
    cudaMemcpy(d_in_canonical, input.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    rowmaj_to_pretransposed(d_in_pre, d_in_canonical, N1, N2);
    cudaDeviceSynchronize();

    // Optimized (canonical I/O)
    double t_opt = bench_ms([&]() {
        cudaMemcpyAsync(d_work, d_in_canonical, n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        ntt_forward_optimized(d_work, config);
    });

    // RowMaj kernel only (assumes data is already pre-transposed)
    double t_rowmaj_pure = bench_ms([&]() {
        cudaMemcpyAsync(d_work, d_in_pre, n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        ntt_forward_rowmaj(d_work, config);
    });

    // RowMaj including transposes (canonical I/O for fair comparison)
    double t_rowmaj_full = bench_ms([&]() {
        cudaMemcpyAsync(d_work, d_in_canonical, n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
        rowmaj_to_pretransposed(d_work, d_work, N1, N2);
        ntt_forward_rowmaj(d_work, config);
        rowmaj_from_pretransposed(d_work, d_work, N1, N2);
    });

    // Memcpy baseline (subtract from above)
    double t_memcpy = bench_ms([&]() {
        cudaMemcpyAsync(d_work, d_in_canonical, n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice);
    });

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Optimized (canonical I/O):    " << (t_opt - t_memcpy)         << " ms\n";
    std::cout << "  RowMaj pure (pre-trans I/O):  " << (t_rowmaj_pure - t_memcpy) << " ms\n";
    std::cout << "  RowMaj full (canonical I/O):  " << (t_rowmaj_full - t_memcpy) << " ms\n";
    std::cout << "  (memcpy baseline:             " << t_memcpy                   << " ms)\n";

    double speedup_pure = (t_opt - t_memcpy) / (t_rowmaj_pure - t_memcpy);
    double speedup_full = (t_opt - t_memcpy) / (t_rowmaj_full - t_memcpy);
    std::cout << "\n  Speedup (RowMaj pure / Optimized):  " << std::setprecision(3) << speedup_pure << "x\n";
    std::cout << "  Speedup (RowMaj full / Optimized):  " << speedup_full << "x\n";

    cudaFree(d_in_canonical);
    cudaFree(d_in_pre);
    cudaFree(d_work);
    ntt_cleanup(config);
    return 0;
}
