#include "ntt.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace ntt;

int main() {
    // Test N=64 (smallest case where the optimized kernel runs)
    // For N=64, the inner kernel runs once with stage_start=0, S=1, offset_o=0.
    // No pre-twist, so should be standalone NTT-64 of bit-reversed input.

    for (uint64_t n : {64ULL, 128ULL, 4096ULL}) {
        std::cout << "\n=== N=" << n << " ===\n";
        NTTConfig* config = ntt_init(n, DEFAULT_PRIME);

        // Use simple input: all zeros except x[0]=1, then x[1]=1, ...
        std::vector<uint64_t> input(n, 0);
        for (uint64_t i = 0; i < n; i++) input[i] = i + 1;

        // Host reference
        std::vector<uint64_t> host_result = input;
        ntt_host(host_result, config->params, false);

        // GPU optimized
        std::vector<uint64_t> gpu_result = input;
        uint64_t* d_data;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, gpu_result.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        ntt_forward_optimized(d_data, config);
        cudaDeviceSynchronize();
        cudaMemcpy(gpu_result.data(), d_data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_data);

        // Print first 16 entries of each
        std::cout << "host: ";
        for (uint64_t i = 0; i < std::min<uint64_t>(16, n); i++) {
            std::cout << host_result[i] << " ";
        }
        std::cout << "\ngpu:  ";
        for (uint64_t i = 0; i < std::min<uint64_t>(16, n); i++) {
            std::cout << gpu_result[i] << " ";
        }
        std::cout << "\n";

        // Find a permutation: for each gpu[k], find host[?] that matches
        int matches = 0;
        for (uint64_t k = 0; k < n; k++) {
            if (gpu_result[k] == host_result[k]) matches++;
        }
        std::cout << "Direct matches: " << matches << " / " << n << "\n";

        // Check if gpu_result is a bit-reversed version of host_result
        int log_n = 0;
        for (uint64_t t = n; t > 1; t >>= 1) log_n++;

        bool is_bitrev = true;
        for (uint64_t k = 0; k < n; k++) {
            uint64_t rev = 0;
            uint64_t tmp = k;
            for (int i = 0; i < log_n; i++) { rev = (rev << 1) | (tmp & 1); tmp >>= 1; }
            if (gpu_result[k] != host_result[rev]) { is_bitrev = false; break; }
        }
        std::cout << "Bit-reversed match: " << (is_bitrev ? "YES" : "NO") << "\n";

        ntt_cleanup(config);
    }
    return 0;
}
