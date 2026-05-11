#include "ntt.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace ntt;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct BenchmarkResult {
    std::string name;
    uint64_t n;
    double time_us;
    double throughput_kops;  // Thousand operations per second
};

// Generate random polynomial
std::vector<uint64_t> random_polynomial(uint64_t n, uint64_t q, uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    
    std::vector<uint64_t> poly(n);
    for (uint64_t i = 0; i < n; i++) {
        poly[i] = dist(rng);
    }
    return poly;
}

// Benchmark a single NTT variant
BenchmarkResult benchmark_ntt(uint64_t n, NTTVariant variant, const std::string& name, 
                              int warmup_iters = 10, int bench_iters = 100) {
    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
    
    auto data = random_polynomial(n, config->q);
    
    uint64_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
        switch (variant) {
            case NTTVariant::BASIC:
                ntt_forward_basic(d_data, config);
                break;
            case NTTVariant::FOUR_STEP:
                ntt_forward_four_step(d_data, config);
                break;
            case NTTVariant::TLMOP:
                ntt_forward_tlmop(d_data, config);
                break;
            case NTTVariant::FULL_OPTIMIZED:
                ntt_forward_optimized(d_data, config);
                break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        switch (variant) {
            case NTTVariant::BASIC:
                ntt_forward_basic(d_data, config);
                break;
            case NTTVariant::FOUR_STEP:
                ntt_forward_four_step(d_data, config);
                break;
            case NTTVariant::TLMOP:
                ntt_forward_tlmop(d_data, config);
                break;
            case NTTVariant::FULL_OPTIMIZED:
                ntt_forward_optimized(d_data, config);
                break;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    double time_us = (elapsed_ms * 1000.0) / bench_iters;
    double throughput = 1000.0 / time_us;  // KOPS (thousands of NTTs per second)
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    ntt_cleanup(config);
    
    return {name, n, time_us, throughput};
}

// Benchmark batched NTT
BenchmarkResult benchmark_ntt_batch(uint64_t n, uint64_t batch_size, NTTVariant variant,
                                     const std::string& name, int warmup_iters = 5, 
                                     int bench_iters = 20) {
    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
    
    auto data = random_polynomial(n * batch_size, config->q);
    
    uint64_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * batch_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * batch_size * sizeof(uint64_t), 
                          cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        ntt_forward_batch(d_data, batch_size, config, variant);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        ntt_forward_batch(d_data, batch_size, config, variant);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    double time_us = (elapsed_ms * 1000.0) / bench_iters;
    double throughput = (batch_size * 1000.0) / time_us;  // KOPS per batch
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    ntt_cleanup(config);
    
    return {name + " (batch=" + std::to_string(batch_size) + ")", n, time_us, throughput};
}

void print_result(const BenchmarkResult& result) {
    std::cout << std::setw(30) << std::left << result.name
              << " N=" << std::setw(6) << result.n
              << " Time: " << std::setw(10) << std::fixed << std::setprecision(2) 
              << result.time_us << " μs"
              << " Throughput: " << std::setw(10) << std::setprecision(2) 
              << result.throughput_kops << " KOPS"
              << std::endl;
}

void print_comparison(const BenchmarkResult& base, const BenchmarkResult& optimized) {
    double speedup = base.time_us / optimized.time_us;
    std::cout << "  Speedup vs " << base.name << ": " 
              << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "=== NTT Benchmark ===" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Prime modulus: " << DEFAULT_PRIME << std::endl;
    std::cout << std::endl;
    
    std::vector<uint64_t> test_sizes = {
        1 << 12,  // 4K
        1 << 14,  // 16K
        1 << 16,  // 64K
        1 << 17,  // 128K
        1 << 18,  // 256K
        1 << 19,  // 512K
        1 << 20,  // 1M
    };
    
    std::cout << "=== Single NTT Performance ===" << std::endl;
    std::cout << std::endl;
    
    for (uint64_t n : test_sizes) {
        std::cout << "--- N = " << n << " ---" << std::endl;
        
        std::vector<BenchmarkResult> results;
        
        // Basic NTT
        results.push_back(benchmark_ntt(n, NTTVariant::BASIC, "Basic"));
        print_result(results.back());
        
        // 4-Step NTT - skip due to known issues at large N
        // if (n >= 4096 && n <= 8192) {
        //     results.push_back(benchmark_ntt(n, NTTVariant::FOUR_STEP, "4-Step"));
        //     print_result(results.back());
        //     print_comparison(results[0], results.back());
        // }
        
        // Optimized NTT
        if (n >= 4096) {
            results.push_back(benchmark_ntt(n, NTTVariant::FULL_OPTIMIZED, "Optimized"));
            print_result(results.back());
            print_comparison(results[0], results.back());
        }
        
        std::cout << std::endl;
    }
    
    // Batch benchmarks (simulating FHE workloads with multiple limbs)
    std::cout << "=== Batched NTT Performance (FHE workload simulation) ===" << std::endl;
    std::cout << std::endl;
    
    uint64_t n = 65536;  // 2^16
    std::vector<uint64_t> batch_sizes = {1, 8, 16};  // Reduced for faster benchmarks
    
    for (uint64_t batch : batch_sizes) {
        auto basic = benchmark_ntt_batch(n, batch, NTTVariant::BASIC, "Basic");
        print_result(basic);
        
        auto optimized = benchmark_ntt_batch(n, batch, NTTVariant::FULL_OPTIMIZED, "Optimized");
        print_result(optimized);
        print_comparison(basic, optimized);
        
        std::cout << std::endl;
    }
    
    // Memory bandwidth analysis
    std::cout << "=== Memory Bandwidth Analysis ===" << std::endl;
    {
        uint64_t n = 65536;
        NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
        
        auto data = random_polynomial(n, config->q);
        uint64_t* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        int iters = 100;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            ntt_forward_optimized(d_data, config);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            ntt_forward_optimized(d_data, config);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        
        // Estimate memory traffic (assuming read + write for each element)
        // Plus twiddle factor reads
        double bytes_per_ntt = n * sizeof(uint64_t) * 3;  // Rough estimate
        double bandwidth_gbps = (bytes_per_ntt * iters) / (elapsed_ms * 1e6);
        
        std::cout << "N = " << n << std::endl;
        std::cout << "Time per NTT: " << (elapsed_ms * 1000.0 / iters) << " μs" << std::endl;
        std::cout << "Estimated bandwidth: " << std::fixed << std::setprecision(1) 
                  << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "A100 peak HBM bandwidth: ~2000 GB/s" << std::endl;
        std::cout << "Bandwidth utilization: " << (bandwidth_gbps / 2000.0 * 100) << "%" << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_data));
        ntt_cleanup(config);
    }
    
    return 0;
}
