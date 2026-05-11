#include "ntt.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace ntt;

constexpr int INNER_SIZE = 64;

// Reference implementation of 64-point NTT using basic NTT
void reference_ntt_64(uint64_t* data, const ModularParams& params, bool inverse) {
    uint64_t omega = inverse ? params.omega_inv : params.omega;
    uint64_t q = params.q;
    uint64_t n = 64;
    
    // Compute omega_64 = omega^(n_full / 64) - but for standalone test, omega IS omega_64
    // For N=64, the twiddle table contains omega_64^k for k=0..63
    
    // Bit reversal
    std::vector<uint64_t> temp(n);
    for (int i = 0; i < n; i++) {
        int rev = 0;
        for (int j = 0; j < 6; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        temp[rev] = data[i];
    }
    std::copy(temp.begin(), temp.end(), data);
    
    // Cooley-Tukey stages
    for (int s = 0; s < 6; s++) {
        int half = 1 << s;
        int full = half << 1;
        uint64_t tw_step = n / full;
        
        for (int group = 0; group < n / full; group++) {
            for (int j = 0; j < half; j++) {
                int i = group * full + j;
                int k = i + half;
                
                // Compute twiddle factor omega_64^(j * tw_step)
                uint64_t exp = (j * tw_step) % n;
                if (inverse) exp = (n - exp) % n;
                
                // omega_64^exp = omega^(exp * (params.n / 64))
                // But for this test we need a standalone omega_64
                // Let's compute it properly
                uint64_t w = 1;
                uint64_t base = omega;
                uint64_t e = exp;
                while (e > 0) {
                    if (e & 1) w = (unsigned __int128)w * base % q;
                    base = (unsigned __int128)base * base % q;
                    e >>= 1;
                }
                
                uint64_t u = data[i];
                uint64_t v = data[k];
                uint64_t vw = (unsigned __int128)v * w % q;
                
                data[i] = (u + vw) % q;
                data[k] = (u >= vw) ? (u - vw) : (u + q - vw);
            }
        }
    }
    
    // Scale for inverse
    if (inverse) {
        uint64_t n_inv = params.n_inv;
        for (int i = 0; i < n; i++) {
            data[i] = (unsigned __int128)data[i] * n_inv % q;
        }
    }
}

int main() {
    std::cout << "=== Inner 64-point NTT Test ===\n";
    
    // Initialize for N=64
    NTTConfig* config = ntt_init(64);
    
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, config->q - 1);
    
    // Test data
    std::vector<uint64_t> original(64);
    std::vector<uint64_t> host_result(64);
    std::vector<uint64_t> device_result(64);
    
    for (int i = 0; i < 64; i++) {
        original[i] = dist(rng);
    }
    
    // Host reference NTT
    std::copy(original.begin(), original.end(), host_result.begin());
    reference_ntt_64(host_result.data(), config->params, false);
    
    // Device NTT using basic implementation
    uint64_t* d_data;
    cudaMalloc(&d_data, 64 * sizeof(uint64_t));
    cudaMemcpy(d_data, original.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    ntt_forward_basic(d_data, config, 0);
    cudaDeviceSynchronize();
    
    cudaMemcpy(device_result.data(), d_data, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Compare
    int errors = 0;
    for (int i = 0; i < 64; i++) {
        if (host_result[i] != device_result[i]) {
            if (errors < 10) {
                std::cout << "Mismatch at " << i << ": host=" << host_result[i] 
                          << ", device=" << device_result[i] << "\n";
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "Forward NTT: PASSED (host matches device)\n";
    } else {
        std::cout << "Forward NTT: FAILED (" << errors << " mismatches)\n";
    }
    
    // Test roundtrip
    cudaMemcpy(d_data, original.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    ntt_forward_basic(d_data, config, 0);
    ntt_inverse_basic(d_data, config, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(device_result.data(), d_data, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    errors = 0;
    for (int i = 0; i < 64; i++) {
        if (original[i] != device_result[i]) {
            if (errors < 10) {
                std::cout << "Roundtrip mismatch at " << i << ": expected=" << original[i] 
                          << ", got=" << device_result[i] << "\n";
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "Roundtrip: PASSED\n";
    } else {
        std::cout << "Roundtrip: FAILED (" << errors << " mismatches)\n";
    }
    
    cudaFree(d_data);
    ntt_cleanup(config);
    
    return errors > 0 ? 1 : 0;
}
