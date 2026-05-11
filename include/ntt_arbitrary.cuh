// ============================================================================
// ntt_arbitrary.cuh  —  FP64 tensor-core NTT for arbitrary-size prime fields
// ----------------------------------------------------------------------------
// Generalization of src/ntt_optimized.cu that lifts the hard-coded 31-bit
// BabyBear prime restriction to *any* prime that fits in K * LIMB_BITS bits
// (K is a compile-time template parameter, LIMB_BITS=25).
//
//     K   PRIME_BITS  examples
//     ─────────────────────────────────────────────────────────────
//      1   ≤25        toy 24-bit prime  (sanity check vs ntt_optimized)
//      2   ≤50        Solinas-style FHE limb primes
//      3   ≤75        Goldilocks (2^64 - 2^32 + 1)
//      6   ≤150       Pallas/Vesta scalar fields
//     11   ≤275       BN254  (254-bit SNARK scalar field)         ← default
//     ─────────────────────────────────────────────────────────────
//
// Each field element is held as K base-2^25 unsigned limbs (FP64-safe Ozaki
// decomposition). A modular multiply `a * b mod p` becomes:
//   1.  K² FP64 partial products  p_{ij} = a_i * b_j        (i,j ∈ [0,K))
//   2.  Group p_{ij} by power 2^((i+j)*25), producing 2K-1 sums each < 2^53
//   3.  Carry-propagate to a canonical 2K-limb base-2^25 integer
//   4.  Trial-subtract p (long-division) back to [0, p)
//
// This header is included by:
//   * src/ntt_mma_arbitrary.cu       — driver with main() + correctness sweep
//   * tests/icicle_arbitrary_compare.cu — apples-to-apples vs ICICLE on BN254
// ============================================================================
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

namespace ntt_arb {

// ============================================================================
// Tunables (compile time)
// ============================================================================
constexpr int LIMB_BITS = 25;
constexpr uint64_t LIMB_BASE = 1ULL << LIMB_BITS;
constexpr uint64_t LIMB_MASK = LIMB_BASE - 1;

[[maybe_unused]] constexpr int MMA_M = 8;
[[maybe_unused]] constexpr int MMA_N = 8;
[[maybe_unused]] constexpr int MMA_K = 4;

// ============================================================================
// BigInt<K>  — K-limb unsigned integer, little-endian (limbs[0] = LSB)
// ============================================================================
template <int K>
struct BigInt {
    uint64_t limbs[K];

    __host__ __device__ BigInt() {
        #pragma unroll
        for (int i = 0; i < K; i++) limbs[i] = 0;
    }
    __host__ __device__ static BigInt zero() { return BigInt(); }
    __host__ __device__ static BigInt one()  { BigInt r; r.limbs[0] = 1; return r; }
    __host__ __device__ static BigInt from_u64(uint64_t v) {
        BigInt r;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            r.limbs[i] = v & LIMB_MASK;
            v >>= LIMB_BITS;
        }
        return r;
    }

    __host__ __device__ bool is_zero() const {
        uint64_t acc = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) acc |= limbs[i];
        return acc == 0;
    }

    __host__ __device__ int cmp(const BigInt& o) const {
        for (int i = K - 1; i >= 0; i--) {
            if (limbs[i] < o.limbs[i]) return -1;
            if (limbs[i] > o.limbs[i]) return +1;
        }
        return 0;
    }
};

// ----------------------------------------------------------------------------
// Hex-string  → BigInt<K>   (host-only; used to embed large constant primes/
// twiddles in source code without writing 11-tuple initialisers by hand).
//
// Accepts strings with or without a "0x" prefix; LSB first ("0123abcd"
// means 0xabcd_0123? No: the string is read MSB-first, like a printed
// hex literal). Each input hex digit contributes 4 bits; bits are packed
// into LIMB_BITS-wide limbs little-endian.
// ----------------------------------------------------------------------------
template <int K>
static inline BigInt<K> from_hex(const char* hex) {
    BigInt<K> r;
    int len = (int)std::strlen(hex);
    int start = 0;
    if (len >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        start = 2;
        len -= 2;
        hex += 2;
    }
    int bit_pos = 0;
    for (int i = len - 1; i >= 0; i--) {
        char c = hex[i];
        int nibble =
            (c >= '0' && c <= '9') ? c - '0' :
            (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
            (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
        for (int b = 0; b < 4; b++) {
            if ((nibble >> b) & 1) {
                int limb_idx     = bit_pos / LIMB_BITS;
                int bit_in_limb  = bit_pos % LIMB_BITS;
                if (limb_idx < K) r.limbs[limb_idx] |= (1ULL << bit_in_limb);
            }
            bit_pos++;
        }
    }
    (void)start;
    return r;
}

// raw add (no modular reduce): returns carry out of the top limb
template <int K>
__host__ __device__ inline uint64_t big_add(BigInt<K>& r,
                                            const BigInt<K>& a,
                                            const BigInt<K>& b) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        uint64_t s = a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = s & LIMB_MASK;
        carry      = s >> LIMB_BITS;
    }
    return carry;
}

template <int K>
__host__ __device__ inline uint64_t big_sub(BigInt<K>& r,
                                            const BigInt<K>& a,
                                            const BigInt<K>& b) {
    int64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        int64_t s = (int64_t)a.limbs[i] - (int64_t)b.limbs[i] - borrow;
        if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; }
        else                                  borrow = 0;
        r.limbs[i] = (uint64_t)s;
    }
    return (uint64_t)borrow;
}

// Right shift by one bit, in place. Returns the bit shifted out (the old LSB).
template <int K>
static inline uint64_t big_shr1(BigInt<K>& a) {
    uint64_t lsb = a.limbs[0] & 1;
    uint64_t carry = 0;
    for (int i = K - 1; i >= 0; i--) {
        uint64_t cur = a.limbs[i];
        a.limbs[i] = (cur >> 1) | (carry << (LIMB_BITS - 1));
        carry = cur & 1;
    }
    return lsb;
}

template <int K>
__host__ __device__ inline void big_mul(BigInt<2 * K>& r,
                                        const BigInt<K>& a,
                                        const BigInt<K>& b) {
    uint64_t acc[2 * K];
    #pragma unroll
    for (int i = 0; i < 2 * K; i++) acc[i] = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < K; j++) {
            uint64_t v = acc[i + j] + a.limbs[i] * b.limbs[j] + carry;
            acc[i + j] = v & LIMB_MASK;
            carry      = v >> LIMB_BITS;
        }
        int k = i + K;
        while (carry) {
            uint64_t v = acc[k] + carry;
            acc[k] = v & LIMB_MASK;
            carry  = v >> LIMB_BITS;
            k++;
        }
    }
    #pragma unroll
    for (int i = 0; i < 2 * K; i++) r.limbs[i] = acc[i];
}

// ============================================================================
// Host-side modular arithmetic. Used to precompute the twiddle table and run
// the reference NTT. NOT performance critical.
// ============================================================================
template <int K>
static inline BigInt<K> mod_reduce_once(const BigInt<K>& a, const BigInt<K>& p) {
    if (a.cmp(p) >= 0) {
        BigInt<K> r;
        big_sub<K>(r, a, p);
        return r;
    }
    return a;
}

template <int K>
static inline BigInt<K> mod_add(const BigInt<K>& a, const BigInt<K>& b,
                                const BigInt<K>& p) {
    BigInt<K> r;
    big_add<K>(r, a, b);
    return mod_reduce_once<K>(r, p);
}

template <int K>
static inline BigInt<K> mod_sub(const BigInt<K>& a, const BigInt<K>& b,
                                const BigInt<K>& p) {
    BigInt<K> r;
    uint64_t borrow = big_sub<K>(r, a, b);
    if (borrow) {
        BigInt<K> tmp;
        big_add<K>(tmp, r, p);
        r = tmp;
    }
    return r;
}

template <int K_OUT, int K_IN>
static inline BigInt<K_OUT> mod_slow(const BigInt<K_IN>& a,
                                     const BigInt<K_OUT>& p) {
    BigInt<K_IN> rem = a;
    int p_bits = 0;
    for (int i = K_OUT - 1; i >= 0; i--) {
        if (p.limbs[i]) {
            for (int b = LIMB_BITS - 1; b >= 0; b--) {
                if (p.limbs[i] >> b) { p_bits = i * LIMB_BITS + b + 1; break; }
            }
            break;
        }
    }
    int a_bits = 0;
    for (int i = K_IN - 1; i >= 0; i--) {
        if (rem.limbs[i]) {
            for (int b = LIMB_BITS - 1; b >= 0; b--) {
                if (rem.limbs[i] >> b) { a_bits = i * LIMB_BITS + b + 1; break; }
            }
            break;
        }
    }
    for (int shift = a_bits - p_bits; shift >= 0; shift--) {
        BigInt<K_IN> ps;
        int limb_shift = shift / LIMB_BITS;
        int bit_shift  = shift % LIMB_BITS;
        for (int i = 0; i < K_OUT; i++) {
            uint64_t v = p.limbs[i] << bit_shift;
            if (limb_shift + i < K_IN) {
                ps.limbs[limb_shift + i] |= v & LIMB_MASK;
            }
            if (bit_shift && limb_shift + i + 1 < K_IN) {
                ps.limbs[limb_shift + i + 1] |= v >> LIMB_BITS;
            }
        }
        bool ge = false;
        for (int i = K_IN - 1; i >= 0; i--) {
            if (rem.limbs[i] != ps.limbs[i]) {
                ge = rem.limbs[i] > ps.limbs[i];
                break;
            }
            if (i == 0) ge = true;
        }
        if (ge) {
            BigInt<K_IN> tmp;
            big_sub<K_IN>(tmp, rem, ps);
            rem = tmp;
        }
    }
    BigInt<K_OUT> r;
    for (int i = 0; i < K_OUT; i++) r.limbs[i] = rem.limbs[i];
    return r;
}

template <int K>
static inline BigInt<K> mod_mul(const BigInt<K>& a, const BigInt<K>& b,
                                const BigInt<K>& p) {
    BigInt<2 * K> prod;
    big_mul<K>(prod, a, b);
    return mod_slow<K, 2 * K>(prod, p);
}

template <int K>
static inline BigInt<K> mod_pow(const BigInt<K>& base, uint64_t exp,
                                const BigInt<K>& p) {
    BigInt<K> r = BigInt<K>::one();
    BigInt<K> b = base;
    while (exp) {
        if (exp & 1) r = mod_mul<K>(r, b, p);
        b = mod_mul<K>(b, b, p);
        exp >>= 1;
    }
    return r;
}

// Square-and-multiply with a BigInt exponent (handles >64-bit exponents).
template <int K>
static inline BigInt<K> mod_pow_big(const BigInt<K>& base, const BigInt<K>& exp,
                                    const BigInt<K>& p) {
    BigInt<K> r = BigInt<K>::one();
    BigInt<K> b = base;
    for (int limb = 0; limb < K; limb++) {
        uint64_t bits = exp.limbs[limb];
        for (int bit = 0; bit < LIMB_BITS; bit++) {
            if (bits & (1ULL << bit)) r = mod_mul<K>(r, b, p);
            b = mod_mul<K>(b, b, p);
        }
    }
    return r;
}

template <int K>
static inline BigInt<K> mod_inv(const BigInt<K>& a, const BigInt<K>& p) {
    BigInt<K> exp;
    {
        BigInt<K> two = BigInt<K>::from_u64(2);
        big_sub<K>(exp, p, two);
    }
    return mod_pow_big<K>(a, exp, p);
}

// ============================================================================
// Montgomery precompute (host).
// ----------------------------------------------------------------------------
// We use CIOS Montgomery multiplication for the NTT hot path. Setup needs
// two constants:
//
//   * np  =  (-p^{-1}) mod 2^LIMB_BITS    -- a single limb, used per outer
//                                            iteration of CIOS to zero T[0].
//   * R2  =  R^2 mod p,   R = 2^(K*LIMB_BITS)
//                                            -- used once per twiddle/element
//                                               at setup to convert x → x·R.
//
// mont_mul(a, b, p, np) computes  a·b·R^{-1}  mod p, so:
//     to_mont(x)   = mont_mul(x, R2, p, np)        (= x·R)
//     from_mont(x) = mont_mul(x, 1, p, np)         (= x·R^{-1})
//     mont_mul(a·R, b·R) = a·b·R                   (closed under multiply)
//
// Per K=11 BN254 modmul, CIOS does ≈ K·(2K+1) ≈ 253 limb multiplies and a
// 1-limb correction subtract — roughly half the work of the previous
// Barrett path (~517 limb mults).
// ============================================================================

// Newton/Hensel-lifted modular inverse of p mod 2^LIMB_BITS, negated.
// Six doublings get us 64 bits of precision; we mask to LIMB_BITS at the end.
template <int K>
static inline uint32_t mont_np(const BigInt<K>& p) {
    uint32_t p0 = (uint32_t)p.limbs[0];   // p mod 2^LIMB_BITS  (odd, since p is prime > 2)
    uint32_t x  = 1;
    for (int k = 0; k < 6; k++) x = x * (2u - p0 * x);
    return (uint32_t)((0u - x) & (uint32_t)LIMB_MASK);
}

// R^2 mod p with R = 2^(K*LIMB_BITS). Built by mod_slow on a (2K+1)-limb
// representation of R^2 (= a single set bit at position 2*K*LIMB_BITS).
template <int K>
static inline BigInt<K> mont_R2(const BigInt<K>& p) {
    BigInt<2 * K + 1> R2;       // zero-initialized via default ctor
    constexpr int bit_pos  = 2 * K * LIMB_BITS;
    constexpr int limb_idx = bit_pos / LIMB_BITS;
    constexpr int bit_in   = bit_pos % LIMB_BITS;
    R2.limbs[limb_idx] |= (1ULL << bit_in);
    return mod_slow<K, 2 * K + 1>(R2, p);
}

// ============================================================================
// FieldConfig<K>  — provides prime + a primitive 2^MAX_LOG_N-th root of unity
// ============================================================================
template <int K_>
struct FieldConfig;

// ----------------------------------------------------------------------------
// K=2   Toy 50-bit prime (debug)
//   p = (1 << 49) - (1 << 24) + 1 = 562949936644097
//   p-1 has 2^24 as a factor, supporting NTTs up to N = 2^24.
// ----------------------------------------------------------------------------
template <> struct FieldConfig<2> {
    static constexpr int K = 2;
    static constexpr int MAX_LOG_N = 12;
    static BigInt<2> prime() {
        BigInt<2> r;
        uint64_t p = ((1ULL << 49) - (1ULL << 24) + 1ULL);
        r.limbs[0] = p & LIMB_MASK; p >>= LIMB_BITS;
        r.limbs[1] = p & LIMB_MASK;
        return r;
    }
    static BigInt<2> omega_2pow_max() {
        const uint64_t p = ((1ULL << 49) - (1ULL << 24) + 1ULL);
        uint64_t exp = (p - 1) >> 12;
        uint64_t base = 3, res = 1;
        while (exp) {
            if (exp & 1) res = (uint64_t)((__uint128_t)res * base % p);
            base = (uint64_t)((__uint128_t)base * base % p);
            exp >>= 1;
        }
        BigInt<2> r;
        r.limbs[0] = res & LIMB_MASK;
        r.limbs[1] = (res >> LIMB_BITS) & LIMB_MASK;
        return r;
    }
};

// ----------------------------------------------------------------------------
// K=3   Goldilocks p = 2^64 - 2^32 + 1
// ----------------------------------------------------------------------------
template <> struct FieldConfig<3> {
    static constexpr int K = 3;
    static constexpr int MAX_LOG_N = 24;
    static BigInt<3> prime() {
        BigInt<3> r;
        uint64_t p = 0xFFFFFFFF00000001ULL;
        r.limbs[0] = p & LIMB_MASK; p >>= LIMB_BITS;
        r.limbs[1] = p & LIMB_MASK; p >>= LIMB_BITS;
        r.limbs[2] = p & LIMB_MASK;
        return r;
    }
    static BigInt<3> omega_2pow_max() {
        BigInt<3> r;
        uint64_t w = 0x185629dcda58878cULL;
        r.limbs[0] = w & LIMB_MASK; w >>= LIMB_BITS;
        r.limbs[1] = w & LIMB_MASK; w >>= LIMB_BITS;
        r.limbs[2] = w & LIMB_MASK;
        return r;
    }
};

// ----------------------------------------------------------------------------
// K=11  BN254 scalar field (r-mod), 254 bits → 11 limbs of 25 bits.
//
//   p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//     = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
//
//   The primitive 2^28-th root of unity (matches ICICLE's `bn254::fp_config::rou`):
//     0x2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0
//
//   MAX_LOG_N is set to 24 here; we square the 2^28-th root four times in
//   omega_2pow_max() to land on a primitive 2^24-th root, which is plenty
//   for benchmarking.
// ----------------------------------------------------------------------------
template <> struct FieldConfig<11> {
    static constexpr int K = 11;
    static constexpr int MAX_LOG_N = 24;
    static BigInt<11> prime() {
        return from_hex<11>(
            "30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001");
    }
    // 2^MAX_LOG_N-th primitive root, computed at host startup by repeatedly
    // squaring the BN254 2^28-th root (= ICICLE's TWO_ADIC_ROOT_OF_UNITY).
    static BigInt<11> omega_2pow_max() {
        BigInt<11> p = prime();
        BigInt<11> w = from_hex<11>(
            "2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0");
        // BN254 two-adicity is exactly 28; square (28 - MAX_LOG_N) times.
        for (int i = 0; i < (28 - MAX_LOG_N); i++) {
            w = mod_mul<11>(w, w, p);
        }
        return w;
    }
};

// ============================================================================
// Device-side multi-limb arithmetic.
// ============================================================================
template <int K>
struct DScalar {
    uint32_t limbs[K];
};

template <int K>
__host__ __device__ inline DScalar<K> load_scalar(const BigInt<K>& x) {
    DScalar<K> r;
    #pragma unroll
    for (int i = 0; i < K; i++) r.limbs[i] = (uint32_t)x.limbs[i];
    return r;
}

template <int K>
__device__ inline BigInt<K> store_scalar(const DScalar<K>& x) {
    BigInt<K> r;
    #pragma unroll
    for (int i = 0; i < K; i++) r.limbs[i] = (uint64_t)x.limbs[i];
    return r;
}

template <int K>
__device__ __forceinline__
DScalar<K> mod_add_d(const DScalar<K>& a, const DScalar<K>& b,
                     const DScalar<K>& p) {
    DScalar<K> sum;
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        uint64_t s = (uint64_t)a.limbs[i] + b.limbs[i] + carry;
        sum.limbs[i] = (uint32_t)(s & LIMB_MASK);
        carry        = (uint32_t)(s >> LIMB_BITS);
    }
    bool ge = (carry != 0);
    if (!ge) {
        for (int i = K - 1; i >= 0; i--) {
            if (sum.limbs[i] != p.limbs[i]) { ge = sum.limbs[i] > p.limbs[i]; break; }
            if (i == 0) ge = true;
        }
    }
    if (ge) {
        int32_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            int64_t s = (int64_t)sum.limbs[i] - (int64_t)p.limbs[i] - borrow;
            if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; } else borrow = 0;
            sum.limbs[i] = (uint32_t)s;
        }
    }
    return sum;
}

template <int K>
__device__ __forceinline__
DScalar<K> mod_sub_d(const DScalar<K>& a, const DScalar<K>& b,
                     const DScalar<K>& p) {
    DScalar<K> r;
    int32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        int64_t s = (int64_t)a.limbs[i] - (int64_t)b.limbs[i] - borrow;
        if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; } else borrow = 0;
        r.limbs[i] = (uint32_t)s;
    }
    if (borrow) {
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            uint64_t s = (uint64_t)r.limbs[i] + p.limbs[i] + carry;
            r.limbs[i] = (uint32_t)(s & LIMB_MASK);
            carry      = (uint32_t)(s >> LIMB_BITS);
        }
    }
    return r;
}

// ============================================================================
// FP64 MMA wrapper  (Ampere+ / sm_80)
// ============================================================================
__device__ __forceinline__
void mma_m8n8k4_f64(double& d0, double& d1,
                    double a, double b,
                    double c0, double c1) {
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

__device__ __forceinline__ uint32_t bit_rev_log(uint32_t x, int log_n) {
    uint32_t r = 0;
    for (int i = 0; i < log_n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

template <int K>
__global__ void bitrev_kernel_d(DScalar<K>* data, uint32_t n, int log_n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t rev = bit_rev_log(idx, log_n);
    if (idx < rev) {
        DScalar<K> tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

// ============================================================================
// Device-side Barrett reduction.
// ----------------------------------------------------------------------------
// (Kept for reference / fallback. The live hot path is mont_mul below.)
// ============================================================================
template <int K>
__device__ inline DScalar<K> barrett_reduce(const uint64_t prod[2 * K],
                                            const DScalar<K>& p,
                                            const DScalar<K + 1>& mu) {
    constexpr int SHIFT = 2 * K;            // limbs to drop after the high mul
    constexpr int FULL  = 3 * K + 1;         // upper bound on (prod * mu) limbs

    // -- (1) q = floor(prod * mu / R²)  where R = 2^(K*LIMB_BITS).
    //    We only care about limbs [SHIFT, SHIFT + K + 1) of the full product.
    //    Cheapest correct path: full schoolbook and read the high slice.
    uint64_t qmul[FULL];
    #pragma unroll
    for (int i = 0; i < FULL; i++) qmul[i] = 0;

    #pragma unroll
    for (int i = 0; i < 2 * K; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < K + 1; j++) {
            uint64_t v = qmul[i + j] + prod[i] * (uint64_t)mu.limbs[j] + carry;
            qmul[i + j] = v & LIMB_MASK;
            carry       = v >> LIMB_BITS;
        }
        int k = i + (K + 1);
        while (carry && k < FULL) {
            uint64_t v = qmul[k] + carry;
            qmul[k] = v & LIMB_MASK;
            carry   = v >> LIMB_BITS;
            k++;
        }
    }

    uint32_t q[K + 1];
    #pragma unroll
    for (int i = 0; i <= K; i++) q[i] = (uint32_t)qmul[SHIFT + i];

    // -- (2) r = prod - q*p   (Barrett's bound guarantees r ∈ [0, 3p).)
    //    We compute the full (2K+1)-limb q*p so carry propagation is exact,
    //    then read only the low K+1 limbs for the subtraction -- the higher
    //    limbs are guaranteed to cancel against the upper part of prod.
    constexpr int QP_LIMBS = 2 * K + 1;
    uint64_t qp[QP_LIMBS];
    #pragma unroll
    for (int i = 0; i < QP_LIMBS; i++) qp[i] = 0;
    #pragma unroll
    for (int i = 0; i <= K; i++) {
        uint64_t qi = (uint64_t)q[i];
        if (!qi) continue;
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < K; j++) {
            uint64_t v = qp[i + j] + qi * p.limbs[j] + carry;
            qp[i + j] = v & LIMB_MASK;
            carry     = v >> LIMB_BITS;
        }
        int kk = i + K;
        while (carry && kk < QP_LIMBS) {
            uint64_t v = qp[kk] + carry;
            qp[kk] = v & LIMB_MASK;
            carry  = v >> LIMB_BITS;
            kk++;
        }
    }

    uint64_t r[K + 1];
    int64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i <= K; i++) {
        int64_t s = (int64_t)prod[i] - (int64_t)qp[i] - borrow;
        if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; } else borrow = 0;
        r[i] = (uint64_t)s;
    }
    // (borrow == 1 here would mean prod < q*p, which Barrett's bound forbids.)

    // -- (3) At most two corrections suffice (r < 3p ⇒ r < p after ≤2 subs).
    #pragma unroll
    for (int trial = 0; trial < 2; trial++) {
        bool ge = (r[K] != 0);
        if (!ge) {
            for (int i = K - 1; i >= 0; i--) {
                if (r[i] != p.limbs[i]) { ge = r[i] > p.limbs[i]; break; }
                if (i == 0) ge = true;
            }
        }
        if (!ge) break;
        int64_t b2 = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            int64_t s = (int64_t)r[i] - (int64_t)p.limbs[i] - b2;
            if (s < 0) { s += (int64_t)LIMB_BASE; b2 = 1; } else b2 = 0;
            r[i] = (uint64_t)s;
        }
        // top limb might absorb the borrow
        if (r[K] >= (uint64_t)b2) r[K] -= (uint64_t)b2;
        else                       r[K]  = (uint64_t)(LIMB_BASE - (uint64_t)b2 + r[K]);
    }

    DScalar<K> out;
    #pragma unroll
    for (int i = 0; i < K; i++) out.limbs[i] = (uint32_t)r[i];
    return out;
}

// ============================================================================
// Montgomery CIOS modular multiply (per-thread).
// ----------------------------------------------------------------------------
// Returns  a · b · R^{-1}  mod p,  with R = 2^(K * LIMB_BITS).
// Inputs are assumed to be in [0, p) in Montgomery form (i.e. each one is
// actually x·R mod p); the result is also in Montgomery form (a·b·R mod p).
//
// CIOS (Coarsely Integrated Operand Scanning) keeps a running accumulator T
// of size K+2 limbs:
//
//   for i in 0..K-1:
//     T += a[i] · b           (K limb-muls + carry chain)
//     m  = T[0] · np mod 2^LIMB_BITS              (chosen so T[0] becomes 0)
//     T += m · p              (K limb-muls + carry chain)
//     T >>= LIMB_BITS         (shift down one limb)
//   return reduce-once(T)
//
// Op count: K · (2K + 1)  ≈ 253 limb multiplies for K=11 (BN254) -- about
// half of the previous Barrett path's ~517 mults, in fewer instructions
// total because the accumulator stays small (K+2 limbs vs Barrett's
// 2K + (K+1) + (2K+1) ≈ 5K+2 limbs of intermediate storage).
//
// Compile-time toggle:
//   * NTT_ARB_USE_FP64=1  → run the inner a[i]·b[j] and m·p[j] products in
//                           FP64 (two 25-bit factors fit exactly in FP64's
//                           53-bit mantissa). Useful on A100/H100 where FP64
//                           throughput is high.
//   * NTT_ARB_USE_FP64=0  → use integer u32×u32→u64 (full rate on every
//                           consumer GPU; FP64 is throttled to ~1/64 on
//                           Blackwell consumer like the 5060).
// Default: integer.
// ============================================================================
#ifndef NTT_ARB_USE_FP64
#define NTT_ARB_USE_FP64 0
#endif

template <int K>
__host__ __device__ inline DScalar<K> mont_mul(const DScalar<K>& a, const DScalar<K>& b,
                                               const DScalar<K>& p, uint32_t np) {
    uint64_t T[K + 2];
    #pragma unroll
    for (int i = 0; i < K + 2; i++) T[i] = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        uint64_t ai = (uint64_t)a.limbs[i];

        // ---- (1) T += a[i] · b ----
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < K; j++) {
#if NTT_ARB_USE_FP64
            uint64_t pp = (uint64_t)((double)a.limbs[i] * (double)b.limbs[j]);
            uint64_t v  = T[j] + pp + carry;
#else
            uint64_t v  = T[j] + ai * b.limbs[j] + carry;
#endif
            T[j] = v & LIMB_MASK;
            carry = v >> LIMB_BITS;
        }
        {
            uint64_t v = T[K] + carry;
            T[K]   = v & LIMB_MASK;
            T[K+1] += v >> LIMB_BITS;
        }

        // ---- (2) m = T[0] · np mod 2^LIMB_BITS  (forces T[0]·... + m·p ≡ 0)
        uint64_t m = (T[0] * (uint64_t)np) & LIMB_MASK;

        // ---- (3) T += m · p ----
        carry = 0;
        #pragma unroll
        for (int j = 0; j < K; j++) {
#if NTT_ARB_USE_FP64
            uint64_t pp = (uint64_t)((double)m * (double)p.limbs[j]);
            uint64_t v  = T[j] + pp + carry;
#else
            uint64_t v  = T[j] + m * (uint64_t)p.limbs[j] + carry;
#endif
            T[j] = v & LIMB_MASK;
            carry = v >> LIMB_BITS;
        }
        {
            uint64_t v = T[K] + carry;
            T[K]   = v & LIMB_MASK;
            T[K+1] += v >> LIMB_BITS;
        }

        // ---- (4) shift T right by one limb.   T[0] is 0 by construction. ----
        #pragma unroll
        for (int j = 0; j < K + 1; j++) T[j] = T[j + 1];
        T[K + 1] = 0;
    }

    // ---- (5) final canonical reduce: T ∈ [0, 2p).  At most one subtract. ---
    bool ge = (T[K] != 0);
    if (!ge) {
        for (int i = K - 1; i >= 0; i--) {
            if (T[i] != p.limbs[i]) { ge = T[i] > p.limbs[i]; break; }
            if (i == 0) ge = true;
        }
    }
    DScalar<K> out;
    if (ge) {
        int64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            int64_t s = (int64_t)T[i] - (int64_t)p.limbs[i] - borrow;
            if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; } else borrow = 0;
            out.limbs[i] = (uint32_t)s;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < K; i++) out.limbs[i] = (uint32_t)T[i];
    }
    return out;
}

// Host BigInt wrapper for mont_mul (used at setup to convert to/from Montgomery).
template <int K>
static inline BigInt<K> mont_mul_big(const BigInt<K>& a, const BigInt<K>& b,
                                     const BigInt<K>& p, uint32_t np) {
    DScalar<K> ad = load_scalar<K>(a);
    DScalar<K> bd = load_scalar<K>(b);
    DScalar<K> pd = load_scalar<K>(p);
    DScalar<K> rd = mont_mul<K>(ad, bd, pd, np);
    BigInt<K> r;
    for (int i = 0; i < K; i++) r.limbs[i] = (uint64_t)rd.limbs[i];
    return r;
}

// ============================================================================
// Basic Cooley-Tukey NTT kernel (one stage per launch).
// ----------------------------------------------------------------------------
// Data + twiddles are expected in Montgomery form (each value held as x·R
// mod p). The kernel works entirely in Montgomery form; conversion back to
// natural form happens once at NTT exit (caller responsibility) via
// `mont_mul(x_mont, 1)`.
// ============================================================================
template <int K>
__global__ void ct_stage_kernel(DScalar<K>* data,
                                const DScalar<K>* twiddles,
                                uint32_t n, int stage,
                                DScalar<K> prime,
                                uint32_t np) {
    uint32_t bfly = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = n >> 1;
    if (bfly >= half) return;

    uint32_t m       = 1u << (stage + 1);
    uint32_t half_m  = m >> 1;
    uint32_t group   = bfly / half_m;
    uint32_t pos     = bfly % half_m;
    uint32_t i       = group * m + pos;
    uint32_t j       = i + half_m;

    DScalar<K> u = data[i];
    DScalar<K> v = data[j];

    uint32_t tw_idx = pos * (n / m);
    DScalar<K> w = twiddles[tw_idx];
    DScalar<K> t = mont_mul<K>(v, w, prime, np);

    data[i] = mod_add_d<K>(u, t, prime);
    data[j] = mod_sub_d<K>(u, t, prime);
}

// ============================================================================
// One full forward NTT (bit-reverse + log_n CT stages). Kernel launches only.
// Caller owns d_data / d_twiddles. For timed loops, do *not* re-bit-reverse
// between iterations -- each call is its own forward NTT.
// ============================================================================
template <int K>
static inline void run_forward_ct(DScalar<K>* d_data,
                                  const DScalar<K>* d_tw,
                                  uint32_t n, int log_n,
                                  DScalar<K> prime_d,
                                  uint32_t np,
                                  cudaStream_t stream = 0) {
    int t = 256;
    int b_br = (n + t - 1) / t;
    bitrev_kernel_d<K><<<b_br, t, 0, stream>>>(d_data, n, log_n);

    int half = n / 2;
    int b_st = (half + t - 1) / t;
    for (int st = 0; st < log_n; st++) {
        ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
    }
}

// ============================================================================
// Host-side reference NTT (slow, used only for correctness).
// ============================================================================
template <int K>
static inline void host_ntt_naive(std::vector<BigInt<K>>& a,
                                  const std::vector<BigInt<K>>& tw,
                                  const BigInt<K>& p) {
    int n = (int)a.size();
    int log_n = 0; for (int t = n; t > 1; t >>= 1) log_n++;
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < log_n; b++) j = (j << 1) | ((i >> b) & 1);
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int s = 0; s < log_n; s++) {
        int m = 1 << (s + 1);
        int half = m >> 1;
        int tw_stride = n / m;
        for (int g = 0; g < n; g += m) {
            for (int k = 0; k < half; k++) {
                BigInt<K> w = tw[(uint64_t)k * tw_stride % n];
                BigInt<K> t = mod_mul<K>(a[g + k + half], w, p);
                BigInt<K> u = a[g + k];
                a[g + k]        = mod_add<K>(u, t, p);
                a[g + k + half] = mod_sub<K>(u, t, p);
            }
        }
    }
}

// ============================================================================
// Setup helpers: twiddle table + RNG input on host, mirrored to device.
// ============================================================================
template <int K>
struct NTTBuffers {
    int n              = 0;
    int log_n          = 0;
    BigInt<K>  prime    {};
    BigInt<K>  R2       {};                  // R^2 mod p  (Montgomery const)
    uint32_t   np       = 0;                 // -p^{-1} mod 2^LIMB_BITS
    BigInt<K>  omega_n  {};
    std::vector<BigInt<K>> tw_host;          // twiddle table on host  (natural form)
    std::vector<BigInt<K>> data_host;        // initial input on host  (natural form)
    DScalar<K>* d_data  = nullptr;           // working device buffer  (Montgomery form)
    DScalar<K>* d_tw    = nullptr;           // twiddle table on device (Montgomery form)
    DScalar<K>  prime_d {};
};

template <int K>
static inline NTTBuffers<K> setup_ntt(int log_n, uint64_t seed = 0xC0FFEE) {
    using FC = FieldConfig<K>;
    NTTBuffers<K> B;
    B.log_n = log_n;
    B.n     = 1 << log_n;
    B.prime = FC::prime();
    B.np    = mont_np<K>(B.prime);
    B.R2    = mont_R2<K>(B.prime);

    BigInt<K> omega_max = FC::omega_2pow_max();
    BigInt<K> omega_n   = omega_max;
    for (int i = 0; i < FC::MAX_LOG_N - log_n; i++) {
        omega_n = mod_mul<K>(omega_n, omega_n, B.prime);
    }
    B.omega_n = omega_n;

    B.tw_host.resize(B.n);
    B.tw_host[0] = BigInt<K>::one();
    for (int i = 1; i < B.n; i++) B.tw_host[i] = mod_mul<K>(B.tw_host[i - 1], omega_n, B.prime);

    std::mt19937_64 rng(seed);
    B.data_host.resize(B.n);
    for (int i = 0; i < B.n; i++) {
        BigInt<K> r;
        for (int j = 0; j < K; j++) r.limbs[j] = rng() & LIMB_MASK;
        // r can be anywhere in [0, R = 2^(K*LIMB_BITS)), but for primes with
        // p << R (e.g. K=11 BN254 has R ≈ 2^21·p) one trial-subtract is not
        // enough -- mont_mul's CIOS bound assumes inputs are canonical [0, p).
        // Use mod_slow for a full reduction.
        B.data_host[i] = mod_slow<K, K>(r, B.prime);
    }

    // Convert twiddles + data to Montgomery form before uploading.
    // x_mont = mont_mul(x, R^2) = x * R^2 / R = x * R   (mod p)
    cudaMalloc(&B.d_data, (size_t)B.n * sizeof(DScalar<K>));
    cudaMalloc(&B.d_tw,   (size_t)B.n * sizeof(DScalar<K>));
    std::vector<DScalar<K>> tw_d(B.n), data_d(B.n);
    for (int i = 0; i < B.n; i++) {
        BigInt<K> tw_mont = mont_mul_big<K>(B.tw_host[i], B.R2, B.prime, B.np);
        tw_d[i] = load_scalar<K>(tw_mont);
    }
    for (int i = 0; i < B.n; i++) {
        BigInt<K> dx_mont = mont_mul_big<K>(B.data_host[i], B.R2, B.prime, B.np);
        data_d[i] = load_scalar<K>(dx_mont);
    }
    cudaMemcpy(B.d_data, data_d.data(), (size_t)B.n * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    cudaMemcpy(B.d_tw,   tw_d.data(),   (size_t)B.n * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    B.prime_d = load_scalar<K>(B.prime);
    return B;
}

template <int K>
static inline void teardown_ntt(NTTBuffers<K>& B) {
    if (B.d_data) cudaFree(B.d_data);
    if (B.d_tw)   cudaFree(B.d_tw);
    B.d_data = nullptr; B.d_tw = nullptr;
}

// ============================================================================
// Driver: correctness check + simple timing for a given K.
// ============================================================================
template <int K>
static inline int run_for_K(int log_n, bool do_host_check = true) {
    using FC = FieldConfig<K>;
    if (log_n > FC::MAX_LOG_N) {
        std::cerr << "log_n=" << log_n << " exceeds MAX_LOG_N=" << FC::MAX_LOG_N
                  << " for K=" << K << "\n";
        return 1;
    }

    NTTBuffers<K> B = setup_ntt<K>(log_n);

    std::cout << "------------------------------------------------------------\n"
              << "K=" << K << " (LIMB_BITS=" << LIMB_BITS
              << "), N=2^" << log_n << "=" << B.n << "\n"
              << "p = ";
    {
        std::cout << "0x";
        bool started = false;
        for (int i = K - 1; i >= 0; i--) {
            if (started)              std::cout << std::hex << std::setw(7) << std::setfill('0') << B.prime.limbs[i];
            else if (B.prime.limbs[i]) { std::cout << std::hex << B.prime.limbs[i]; started = true; }
        }
        std::cout << std::dec << std::setfill(' ') << "\n";
    }

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // Helper: re-upload pristine input in Montgomery form (used between timed
    // iterations and for the correctness check).
    auto reupload_mont_input = [&] {
        std::vector<DScalar<K>> data_d(B.n);
        for (int i = 0; i < B.n; i++) {
            BigInt<K> dx = mont_mul_big<K>(B.data_host[i], B.R2, B.prime, B.np);
            data_d[i] = load_scalar<K>(dx);
        }
        cudaMemcpy(B.d_data, data_d.data(), (size_t)B.n * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    };

    // -- warmup + timed forward NTT (one full pass per iter) ------------------
    for (int it = 0; it < 3; it++) run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
    cudaDeviceSynchronize();

    reupload_mont_input();
    cudaEventRecord(s);
    const int iters = 20;
    for (int it = 0; it < iters; it++) {
        run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    std::cout << "  CT stages NTT: " << (ms * 1000.0 / iters) << " us/iter\n";

    int mismatches = 0;
    if (do_host_check) {
        reupload_mont_input();
        run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
        cudaDeviceSynchronize();
        std::vector<DScalar<K>> data_d(B.n);
        cudaMemcpy(data_d.data(), B.d_data, (size_t)B.n * sizeof(DScalar<K>), cudaMemcpyDeviceToHost);

        std::vector<BigInt<K>> ref = B.data_host;
        host_ntt_naive<K>(ref, B.tw_host, B.prime);
        BigInt<K> one = BigInt<K>::one();
        for (int i = 0; i < B.n; i++) {
            BigInt<K> got_mont;
            for (int j = 0; j < K; j++) got_mont.limbs[j] = data_d[i].limbs[j];
            // x = mont_mul(x_mont, 1) = x_mont · 1 · R^{-1} = (original x)
            BigInt<K> got = mont_mul_big<K>(got_mont, one, B.prime, B.np);
            if (got.cmp(ref[i]) != 0) {
                if (mismatches < 5) {
                    std::cout << "  MISMATCH @ i=" << i << "  got.limbs[0]=" << got.limbs[0]
                              << " ref.limbs[0]=" << ref[i].limbs[0] << "\n";
                }
                mismatches++;
            }
        }
        if (mismatches == 0) std::cout << "  CORRECT (matches host reference at all "
                                       << B.n << " positions)\n";
        else                 std::cout << "  FAILED (" << mismatches << " mismatches)\n";
    } else {
        std::cout << "  (host-check skipped)\n";
    }

    cudaEventDestroy(s); cudaEventDestroy(e);
    teardown_ntt<K>(B);
    return mismatches == 0 ? 0 : 1;
}

} // namespace ntt_arb
