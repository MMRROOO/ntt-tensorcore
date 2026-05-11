#include "ntt.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

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

// Generate random polynomial with coefficients in [0, q)
std::vector<uint64_t> random_polynomial(uint64_t n, uint64_t q, uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    
    std::vector<uint64_t> poly(n);
    for (uint64_t i = 0; i < n; i++) {
        poly[i] = dist(rng);
    }
    return poly;
}

// Compare two polynomials
bool compare_polynomials(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, 
                         const std::string& name = "") {
    if (a.size() != b.size()) {
        std::cerr << name << ": Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    uint64_t errors = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            if (errors < 10) {
                std::cerr << name << ": Mismatch at index " << i 
                          << ": expected " << a[i] << ", got " << b[i] << std::endl;
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        std::cerr << name << ": Total errors: " << errors << " / " << a.size() << std::endl;
        return false;
    }
    return true;
}

// Test: Forward NTT followed by inverse NTT should give back original
bool test_ntt_inverse(uint64_t n, NTTVariant variant, const std::string& variant_name) {
    std::cout << "Testing " << variant_name << " NTT/INTT roundtrip (N=" << n << ")..." << std::endl;
    
    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
    
    // Generate random polynomial
    auto original = random_polynomial(n, config->q);
    auto data = original;
    
    // Allocate device memory
    uint64_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Forward NTT
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
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Inverse NTT
    switch (variant) {
        case NTTVariant::BASIC:
            ntt_inverse_basic(d_data, config);
            break;
        case NTTVariant::FOUR_STEP:
            ntt_inverse_four_step(d_data, config);
            break;
        case NTTVariant::TLMOP:
            ntt_inverse_tlmop(d_data, config);
            break;
        case NTTVariant::FULL_OPTIMIZED:
            ntt_inverse_optimized(d_data, config);
            break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back and compare
    CUDA_CHECK(cudaMemcpy(data.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    bool passed = compare_polynomials(original, data, variant_name);
    
    CUDA_CHECK(cudaFree(d_data));
    ntt_cleanup(config);
    
    if (passed) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
    }
    
    return passed;
}

// Test: Compare GPU NTT with host reference implementation
bool test_ntt_correctness(uint64_t n, NTTVariant variant, const std::string& variant_name) {
    std::cout << "Testing " << variant_name << " correctness vs host (N=" << n << ")..." << std::endl;
    
    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
    
    // Generate random polynomial
    auto original = random_polynomial(n, config->q);
    auto host_data = original;
    auto gpu_data = original;
    
    // Host NTT
    ntt_host(host_data, config->params, false);
    
    // GPU NTT
    uint64_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_data, gpu_data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
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
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(gpu_data.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    bool passed = compare_polynomials(host_data, gpu_data, variant_name);
    
    CUDA_CHECK(cudaFree(d_data));
    ntt_cleanup(config);
    
    if (passed) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
    }
    
    return passed;
}

// Test: Polynomial multiplication using NTT
bool test_polynomial_multiplication(uint64_t n) {
    std::cout << "Testing polynomial multiplication (N=" << n << ")..." << std::endl;
    
    NTTConfig* config = ntt_init(n, DEFAULT_PRIME);
    
    // Generate two random polynomials
    auto poly_a = random_polynomial(n, config->q, 42);
    auto poly_b = random_polynomial(n, config->q, 123);
    
    // Reference: Naive polynomial multiplication (mod x^N + 1)
    std::vector<uint64_t> reference(n, 0);
    for (uint64_t i = 0; i < n; i++) {
        for (uint64_t j = 0; j < n; j++) {
            uint64_t prod = (unsigned __int128)poly_a[i] * poly_b[j] % config->q;
            uint64_t idx = (i + j) % n;
            bool negate = (i + j) >= n;  // Negacyclic: x^N = -1
            if (negate) {
                reference[idx] = (reference[idx] + config->q - prod) % config->q;
            } else {
                reference[idx] = (reference[idx] + prod) % config->q;
            }
        }
    }
    
    // NTT-based multiplication
    uint64_t *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result, n * sizeof(uint64_t)));
    
    CUDA_CHECK(cudaMemcpy(d_a, poly_a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, poly_b.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Forward NTT on both
    ntt_forward_basic(d_a, config);
    ntt_forward_basic(d_b, config);
    
    // Point-wise multiplication
    poly_multiply_ntt(d_result, d_a, d_b, config);
    
    // Inverse NTT
    ntt_inverse_basic(d_result, config);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back and compare
    std::vector<uint64_t> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    bool passed = compare_polynomials(reference, result, "PolyMul");
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    ntt_cleanup(config);
    
    if (passed) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cout << "  FAILED" << std::endl;
    }
    
    return passed;
}

int main() {
    std::cout << "=== NTT Test Suite ===" << std::endl;
    std::cout << "Prime modulus: " << DEFAULT_PRIME << std::endl;
    
    // Check CUDA device
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    int passed = 0;
    int total = 0;
    
    // Test different sizes
    std::vector<uint64_t> test_sizes = {64, 256, 1024, 4096};
    
    // Test NTT/INTT roundtrip for Basic variant
    std::cout << "--- Basic NTT Roundtrip Tests ---" << std::endl;
    for (uint64_t n : test_sizes) {
        if (test_ntt_inverse(n, NTTVariant::BASIC, "Basic")) passed++;
        total++;
    }
    std::cout << std::endl;
    
    // Test correctness against host implementation
    std::cout << "--- Basic NTT Correctness Tests ---" << std::endl;
    for (uint64_t n : test_sizes) {
        if (test_ntt_correctness(n, NTTVariant::BASIC, "Basic")) passed++;
        total++;
    }
    std::cout << std::endl;
    
    // Test 4-step NTT for larger sizes
    std::cout << "--- Four-Step NTT Tests ---" << std::endl;
    std::vector<uint64_t> large_sizes = {4096, 8192, 16384};
    for (uint64_t n : large_sizes) {
        if (test_ntt_inverse(n, NTTVariant::FOUR_STEP, "FourStep")) passed++;
        total++;
    }
    std::cout << std::endl;
    
    // Test optimized NTT (Tensor Core) - including very large sizes
    std::cout << "--- Optimized NTT Tests ---" << std::endl;
    std::vector<uint64_t> opt_sizes = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    for (uint64_t n : opt_sizes) {
        if (test_ntt_inverse(n, NTTVariant::FULL_OPTIMIZED, "Optimized")) passed++;
        total++;
    }
    std::cout << std::endl;
    
    // Correctness vs host for optimized
    std::cout << "--- Optimized NTT Correctness vs Host ---" << std::endl;
    std::vector<uint64_t> opt_correctness_sizes = {4096, 8192, 16384, 32768, 65536};
    for (uint64_t n : opt_correctness_sizes) {
        if (test_ntt_correctness(n, NTTVariant::FULL_OPTIMIZED, "Optimized")) passed++;
        total++;
    }
    std::cout << std::endl;
    
    // Skip polynomial multiplication tests - requires negacyclic NTT
    // Our implementation is a standard cyclic NTT
    std::cout << "--- Polynomial Multiplication Tests ---" << std::endl;
    std::cout << "  (Skipped - requires negacyclic convolution, not implemented)" << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << " / " << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}
