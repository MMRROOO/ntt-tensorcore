#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace ntt {

// 32-bit modular arithmetic for NTT
// Using Barrett reduction for efficiency

struct ModularParams {
    uint64_t q;           // Prime modulus (up to 32 bits)
    uint64_t q_twice;     // 2*q for lazy reduction
    uint64_t barrett_k;   // Barrett reduction parameter k = ceil(log2(q)) + 32
    uint64_t barrett_m;   // Barrett reduction multiplier m = floor(2^k / q)
    uint64_t omega;       // Primitive 2N-th root of unity
    uint64_t omega_inv;   // Inverse of omega
    uint64_t n_inv;       // Inverse of N mod q
};

// Common FHE-friendly primes (q ≡ 1 mod 2N for N up to 2^16)
// These primes allow primitive 2N-th roots of unity
constexpr uint64_t PRIME_32_1 = 0xFFFFFFFF00000001ULL; // 2^64 - 2^32 + 1 (Goldilocks)
constexpr uint64_t PRIME_30_1 = 1073479681ULL;  // 2^30 - 2^20 + 1, divisible by 2^20
constexpr uint64_t PRIME_30_2 = 1073610753ULL;  // Another 30-bit prime
constexpr uint64_t PRIME_31_1 = 2013265921ULL;  // 15 * 2^27 + 1
constexpr uint64_t PRIME_32_2 = 4293918721ULL;  // 2^32 - 2^20 + 1

// Default prime for our implementation (allows roots of unity up to 2^27)
constexpr uint64_t DEFAULT_PRIME = 2013265921ULL;  // 15 * 2^27 + 1

// Device functions for modular arithmetic

__device__ __forceinline__ 
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

__device__ __forceinline__ 
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}

__device__ __forceinline__ 
uint64_t mod_mul_naive(uint64_t a, uint64_t b, uint64_t q) {
    // Simple modular multiplication using 128-bit intermediate
    unsigned __int128 prod = (unsigned __int128)a * b;
    return (uint64_t)(prod % q);
}

__device__ __forceinline__
uint64_t mod_mul_barrett(uint64_t a, uint64_t b, uint64_t q, uint64_t barrett_m, int barrett_k) {
    // Barrett reduction: faster than naive modulo
    // q_hat = floor((a*b * barrett_m) >> barrett_k)
    // r = a*b - q_hat * q
    unsigned __int128 prod = (unsigned __int128)a * b;
    uint64_t prod_lo = (uint64_t)prod;
    uint64_t prod_hi = (uint64_t)(prod >> 64);
    
    // Approximate quotient
    unsigned __int128 tmp = (unsigned __int128)prod_hi * barrett_m + 
                            ((unsigned __int128)prod_lo * barrett_m >> 64);
    uint64_t q_hat = (uint64_t)(tmp >> (barrett_k - 64));
    
    // Compute remainder
    uint64_t r = prod_lo - q_hat * q;
    
    // Correction step (at most 2 subtractions needed)
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    
    return r;
}

// Montgomery reduction for 32-bit moduli
struct Montgomery32 {
    uint32_t q;        // Modulus
    uint32_t q_inv;    // -q^(-1) mod 2^32
    uint32_t r2;       // R^2 mod q, where R = 2^32
    
    __device__ __forceinline__
    uint32_t reduce(uint64_t a) const {
        // Montgomery reduction: compute a * R^(-1) mod q
        uint32_t a_lo = (uint32_t)a;
        uint32_t m = a_lo * q_inv;
        uint64_t t = a + (uint64_t)m * q;
        uint32_t result = (uint32_t)(t >> 32);
        return result >= q ? result - q : result;
    }
    
    __device__ __forceinline__
    uint32_t mul(uint32_t a, uint32_t b) const {
        return reduce((uint64_t)a * b);
    }
    
    __device__ __forceinline__
    uint32_t to_montgomery(uint32_t a) const {
        return reduce((uint64_t)a * r2);
    }
    
    __device__ __forceinline__
    uint32_t from_montgomery(uint32_t a) const {
        return reduce(a);
    }
};

// Host functions for setup

inline uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (unsigned __int128)result * base % mod;
        }
        base = (unsigned __int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

inline uint64_t mod_inverse(uint64_t a, uint64_t mod) {
    return mod_pow(a, mod - 2, mod);
}

// Find primitive N-th root of unity in F_q.
// Used uniformly throughout: twiddles[k] = omega^k = primitive N-th root raised to k.
// This corresponds to the standard cyclic NTT (no negacyclic twist baked into omega).
inline uint64_t find_primitive_root(uint64_t q, uint64_t n) {
    // Find a generator of the multiplicative group F_q*
    uint64_t g = 2;
    while (true) {
        bool is_primitive = true;
        uint64_t order = q - 1;
        for (uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL}) {
            if (order % p == 0) {
                if (mod_pow(g, order / p, q) == 1) {
                    is_primitive = false;
                    break;
                }
            }
        }
        if (is_primitive) break;
        g++;
    }

    // omega = g^((q-1)/N) is a primitive N-th root of unity.
    uint64_t exponent = (q - 1) / n;
    return mod_pow(g, exponent, q);
}

inline ModularParams create_params(uint64_t q, uint64_t n) {
    ModularParams params;
    params.q = q;
    params.q_twice = 2 * q;
    
    // Barrett parameters
    int k = 64;  // For 32-bit moduli
    while ((1ULL << (k - 64)) < q && k < 96) k++;
    params.barrett_k = k;
    params.barrett_m = ((unsigned __int128)1 << k) / q;
    
    // Find primitive root
    params.omega = find_primitive_root(q, n);
    params.omega_inv = mod_inverse(params.omega, q);
    params.n_inv = mod_inverse(n, q);
    
    return params;
}

// Precompute twiddle factors
inline void compute_twiddle_factors(uint64_t* twiddles, uint64_t n, uint64_t omega, uint64_t q) {
    twiddles[0] = 1;
    for (uint64_t i = 1; i < n; i++) {
        twiddles[i] = (unsigned __int128)twiddles[i-1] * omega % q;
    }
}

} // namespace ntt
