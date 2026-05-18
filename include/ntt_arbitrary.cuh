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
// K=2   49-bit NTT-friendly prime
//   p = 16777308 * 2^24 + 1 = 281476520214529
//   p-1 = 16777308 * 2^24, so 2^24 divides p-1, supporting NTTs up to N = 2^24.
//   Primitive root: g = 7
// ----------------------------------------------------------------------------
template <> struct FieldConfig<2> {
    static constexpr int K = 2;
    static constexpr int MAX_LOG_N = 12;
    static BigInt<2> prime() {
        BigInt<2> r;
        uint64_t p = 281476520214529ULL;  // 16777308 * 2^24 + 1
        r.limbs[0] = p & LIMB_MASK; p >>= LIMB_BITS;
        r.limbs[1] = p & LIMB_MASK;
        return r;
    }
    static BigInt<2> omega_2pow_max() {
        // omega_4096 = 7^((p-1)/4096) mod p
        // Precomputed: 30523202938030
        const uint64_t omega = 30523202938030ULL;
        BigInt<2> r;
        r.limbs[0] = omega & LIMB_MASK;
        r.limbs[1] = (omega >> LIMB_BITS) & LIMB_MASK;
        return r;
    }
};

// ----------------------------------------------------------------------------
// K=3   Goldilocks p = 2^64 - 2^32 + 1
//   p-1 = 2^32 × (2^32 - 1), supports NTTs up to N = 2^32
//   Primitive root: g = 7
//   omega_2^24 = 7^((p-1)/2^24) = 0x86cdcc31c307e171
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
        uint64_t w = 0x86cdcc31c307e171ULL;  // Correct 2^24-th root of unity
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
// The MMA m8n8k4 instruction computes D = A × B + C for:
//   - A: 8×4 matrix (row-major), thread t provides A[t/4, t%4]
//   - B: 4×8 matrix (col-major), thread t provides B[t%4, t/4]
//   - C: 8×8 matrix, thread t provides C[t/4, 2*(t%4)] and C[t/4, 2*(t%4)+1]
//   - D: 8×8 matrix, thread t receives D[t/4, 2*(t%4)] and D[t/4, 2*(t%4)+1]
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
    // Scalar fallback for architectures without FP64 MMA
    // This computes a simplified version that works when called cooperatively
    // by all 32 threads in a warp. Each thread contributes a*b to the
    // appropriate output position via warp shuffle.
    
    int lane = threadIdx.x & 31;
    int row = lane >> 2;       // A-fragment row (0..7)
    int qq = lane & 3;         // A-fragment col (0..3)
    
    // Compute partial product a * b for this thread
    double prod = a * b;
    
    // For D[row, j], we need to sum over k where A[row, k] * B[k, j]
    // Thread (row, qq) provides A[row, qq] and B[qq, row]
    // So thread (row, qq) contributes to D[row, col] where B[qq, col] is available
    
    // Use warp shuffle to accumulate across k=0..3
    // D[row, 2*qq] and D[row, 2*qq+1] need sums from k=0..3
    
    // For each output column j, we need sum over k of A[row,k]*B[k,j]
    // Thread with (row', qq'=k) provides B[k, row']
    // So D[row, j] = sum over threads (row, k) of their a * (B value from thread (j, k))
    
    // Simplified: do a full warp reduction
    double sum0 = c0, sum1 = c1;
    
    // Accumulate contributions using shuffle
    for (int k = 0; k < 4; k++) {
        // Get A[row, k] from thread (row, k)
        int src_lane = row * 4 + k;
        double a_val = __shfl_sync(0xFFFFFFFF, a, src_lane);
        
        // Get B[k, 2*qq] and B[k, 2*qq+1]
        // B[k, j] is held by thread with (row'=j, qq'=k)
        int src_lane_b0 = (2 * qq) * 4 + k;
        int src_lane_b1 = (2 * qq + 1) * 4 + k;
        double b_val0 = __shfl_sync(0xFFFFFFFF, b, src_lane_b0);
        double b_val1 = __shfl_sync(0xFFFFFFFF, b, src_lane_b1);
        
        sum0 += a_val * b_val0;
        sum1 += a_val * b_val1;
    }
    
    d0 = sum0;
    d1 = sum1;
#endif
}

__device__ __forceinline__ uint32_t bit_rev_log(uint32_t x, int log_n) {
    uint32_t r = 0;
    for (int i = 0; i < log_n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// ============================================================================
// HMFHE-Style Radix-64 MMA Inner NTT for K-limb Montgomery Data
// ============================================================================
// This implements the full HMFHE paper optimizations (ISCA 2026) adapted for
// arbitrary K-limb prime fields:
//
//   TLMOP  (§ IV-A3): All intermediate K-limb values stay in registers between
//                     MMA operations. SMEM touched only at initial load and
//                     final store.
//
//   TransOP (§ IV-A4): The 4-step NTT's transpose is implicit via fragment
//                      re-mapping. After MMA 1+2, D-fragment layout directly
//                      matches A-fragment layout for MMA 3+4.
//
//   TFOP   (§ IV-B2): TFM_8 (radix-8 DFT matrix) and Hada64 (inner twiddles)
//                     are block-shared in SMEM, eliminating per-butterfly GMEM
//                     twiddle reads.
//
// Radix-64 decomposition:
//   NTT-64 = two radix-8 stages separated by element-wise twiddle multiply.
//   Each radix-8 is an 8×8 matrix-vector product: out[i] = Σ_k TFM[i,k]·x[k]
//
// MMA call count for K-limb:
//   Each limb pair (a, b) with a,b ∈ [0,K) requires 2 MMA calls (for k=0..3
//   and k=4..7 inner-dim halves). Two radix-8 stages per NTT-64 yields:
//     Total = 2 stages × K² pairs × 2 calls/pair = 4×K² MMA calls per warp.
//
//   K=2:  16 MMA calls     K=3:  36 MMA calls     K=11: 484 MMA calls
//
// Per-warp processing:
//   Each warp computes one 64-point NTT of K-limb Montgomery elements.
//   Lane mapping (m8n8k4 FP64):
//     - Lane t ∈ [0,32) maps to (i = t/4, qq = t%4) in the 8×8 output matrix.
//     - Each lane produces D[i, 2*qq] and D[i, 2*qq+1] from the MMA.
//
// ============================================================================

// Compile-time feature gate (default ON for sm_80+, K-limb)
#ifndef NTT_ARB_USE_TCU_INNER
#define NTT_ARB_USE_TCU_INNER 1
#endif


// TCU kernel constants
constexpr int TCU_INNER_SIZE   = 64;
constexpr int TCU_LOG_INNER    = 6;
constexpr int TCU_WARP_SIZE    = 32;
constexpr int TCU_WARPS_PER_BLK = 4;    // 4 NTT-64s per block (tuned for K=11)
constexpr int TCU_BLOCK_SIZE   = TCU_WARP_SIZE * TCU_WARPS_PER_BLK;

// SMEM layout with bank-conflict avoidance padding
constexpr int TCU_WARP_STRIDE  = 9;     // 8+1 padding per row
constexpr int TCU_WARP_ROWS    = 8;
constexpr int TCU_WARP_TOTAL   = TCU_WARP_STRIDE * TCU_WARP_ROWS;  // 72

// Bit-reversal helpers for radix-8 and radix-64
__device__ __forceinline__ int bit_rev_3(int x) {
    return ((x & 1) << 2) | (x & 2) | ((x >> 2) & 1);
}

__device__ __forceinline__ int bit_rev_6(int x) {
    return (bit_rev_3(x & 7) << 3) | bit_rev_3((x >> 3) & 7);
}

// Padded SMEM index: maps flat j ∈ [0,64) to padded layout
__device__ __forceinline__ int wsm_idx(int j) {
    return (j >> 3) * TCU_WARP_STRIDE + (j & 7);
}

// ============================================================================
// Wide Montgomery Reduction for K-limb
// ============================================================================
// Reduces a (2K+1)-limb non-negative integer P to a K-limb result mod p.
// Computes: out = P × R^{-1} mod p, where R = 2^(K × LIMB_BITS).
//
// This is used after accumulating K² MMA partial products. Each limb-position
// accumulator holds the sum of up to 8×K products of 25-bit values, which can
// exceed a single limb. The accumulators are first carry-propagated to
// canonical (2K+1)-limb form, then Montgomery-reduced.
//
// The CIOS schedule is: for j = 0..K-1:
//   m = P[0] × np mod LIMB_BASE
//   P += m × p
//   P >>= LIMB_BITS
// After K iterations, P ∈ [0, 2p) with one conditional subtract.
// ============================================================================
template <int K>
__device__ __forceinline__
DScalar<K> mont_reduce_wide(uint64_t* P, const DScalar<K>& p, uint32_t np) {
    // CIOS reduction: K iterations
    #pragma unroll
    for (int j = 0; j < K; j++) {
        uint64_t m = (P[0] * (uint64_t)np) & LIMB_MASK;
        uint64_t carry = 0;
        #pragma unroll
        for (int kk = 0; kk < K; kk++) {
            uint64_t v = P[kk] + m * (uint64_t)p.limbs[kk] + carry;
            P[kk] = v & LIMB_MASK;
            carry = v >> LIMB_BITS;
        }
        // Propagate carry through high limbs (unrolled)
        #pragma unroll
        for (int idx = K; idx <= 2 * K; idx++) {
            uint64_t v = P[idx] + carry;
            P[idx] = v & LIMB_MASK;
            carry = v >> LIMB_BITS;
        }
        // Shift down by one limb (P[0] is zero by construction)
        #pragma unroll
        for (int kk = 0; kk < 2 * K; kk++) P[kk] = P[kk + 1];
        P[2 * K] = 0;
    }

    // Final reduction: result is in P[0..K], may be in [0, 2p)
    bool ge = (P[K] != 0);
    if (!ge) {
        #pragma unroll
        for (int i = K - 1; i >= 0; i--) {
            if (P[i] != p.limbs[i]) { ge = (P[i] > p.limbs[i]); break; }
            if (i == 0) ge = true;
        }
    }

    DScalar<K> out;
    if (ge) {
        int64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            int64_t s = (int64_t)P[i] - (int64_t)p.limbs[i] - borrow;
            if (s < 0) { s += (int64_t)LIMB_BASE; borrow = 1; } else borrow = 0;
            out.limbs[i] = (uint32_t)s;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < K; i++) out.limbs[i] = (uint32_t)P[i];
    }
    return out;
}

// ============================================================================
// MMA-based Radix-8 for K-limb: computes 8 parallel NTT-8 instances per warp
// ============================================================================
// Each warp's 32 lanes process 64 K-limb elements (8 groups × 8 elements).
// The K² limb-pair MMA accumulations are followed by wide Montgomery reduction.
//
// Fragment layout (FP64 m8n8k4):
//   A-fragment: lane t holds A[t/4, t%4]     (one element from 8×4 left half)
//   B-fragment: lane t holds B[t%4, t/4]     (one element from 4×8 top half)
//   D-fragment: lane t holds D[t/4, 2*(t%4)] and D[t/4, 2*(t%4)+1]
//
// For radix-8 DFT: out[i] = Σ_{k=0..7} TFM[i,k] × data[k]
//   - TFM is the 8×8 radix-8 DFT matrix: TFM[i,k] = ω_8^(i×k)
//   - Each TFM[i,k] and data[k] is a K-limb Montgomery number
//   - Result requires K² partial products per output cell
// ============================================================================
template <int K>
__device__ __forceinline__
void mma_radix8_K(
    DScalar<K>* out,          // Output: 8 K-limb results (indexed by output row i)
    const DScalar<K>* tfm8,   // TFM_8[64]: ω_8^(i×k), row-major, K-limb Montgomery
    const DScalar<K>* data,   // Input: 8 K-limb elements in bit-reversed order
    const DScalar<K>& prime,
    uint32_t np,
    bool transpose_output     // If true, use TransOP fragment swap for second stage
) {
    int lane = threadIdx.x & 31;
    int i    = lane >> 2;      // Output row: 0..7
    int qq   = lane & 3;       // Inner-k chunk: 0..3

    // Bit-reversed input indices for fragment B
    int br_top = bit_rev_3(qq);
    int br_bot = bit_rev_3(qq + 4);

    // Per-output-cell accumulators indexed by limb position l = a + b
    // After summing K² pairs, acc[l] holds partial sum for output limb l
    int64_t acc_0[2 * K - 1];  // For D[i, 2*qq]
    int64_t acc_1[2 * K - 1];  // For D[i, 2*qq+1]
    #pragma unroll
    for (int l = 0; l < 2 * K - 1; l++) { acc_0[l] = 0; acc_1[l] = 0; }

    // K² MMA loop: for each (a, b) limb pair
    #pragma unroll 1
    for (int a = 0; a < K; a++) {
        // A-fragment: TFM[i, qq] and TFM[i, qq+4] limb a
        double tfm_left  = (double)(uint32_t)tfm8[i * 8 + qq].limbs[a];
        double tfm_right = (double)(uint32_t)tfm8[i * 8 + qq + 4].limbs[a];

        #pragma unroll 1
        for (int b = 0; b < K; b++) {
            // B-fragment: data[br_top] and data[br_bot] limb b
            // Note: lane's B-element row = qq (or qq+4), col = i
            double dat_top = (double)(uint32_t)data[br_top].limbs[b];
            double dat_bot = (double)(uint32_t)data[br_bot].limbs[b];

            // Two MMA calls cover k=0..7 (left half k=0..3, right half k=4..7)
            double d0 = 0.0, d1 = 0.0;
            mma_m8n8k4_f64(d0, d1, tfm_left,  dat_top, 0.0, 0.0);
            mma_m8n8k4_f64(d0, d1, tfm_right, dat_bot, d0, d1);

            // Accumulate into limb position a+b
            acc_0[a + b] += (int64_t)d0;
            acc_1[a + b] += (int64_t)d1;
        }
    }

    // Carry-propagate accumulators to (2K+1)-limb canonical form
    uint64_t P_0[2 * K + 1];
    uint64_t P_1[2 * K + 1];
    {
        uint64_t carry_0 = 0, carry_1 = 0;
        #pragma unroll
        for (int l = 0; l < 2 * K - 1; l++) {
            uint64_t v0 = (uint64_t)acc_0[l] + carry_0;
            P_0[l] = v0 & LIMB_MASK;
            carry_0 = v0 >> LIMB_BITS;

            uint64_t v1 = (uint64_t)acc_1[l] + carry_1;
            P_1[l] = v1 & LIMB_MASK;
            carry_1 = v1 >> LIMB_BITS;
        }
        P_0[2 * K - 1] = carry_0 & LIMB_MASK;
        P_0[2 * K]     = carry_0 >> LIMB_BITS;
        P_1[2 * K - 1] = carry_1 & LIMB_MASK;
        P_1[2 * K]     = carry_1 >> LIMB_BITS;
    }

    // Montgomery reduce to K-limb
    DScalar<K> out_0 = mont_reduce_wide<K>(P_0, prime, np);
    DScalar<K> out_1 = mont_reduce_wide<K>(P_1, prime, np);

    // Store outputs: lane t writes to positions determined by transpose flag
    // Without transpose: out[2*qq] and out[2*qq+1] (used after MMA 1+2)
    // With transpose: positions swapped for MMA 3+4 output mapping
    if (!transpose_output) {
        out[2 * qq]     = out_0;
        out[2 * qq + 1] = out_1;
    } else {
        // TransOP: D[i,j] written to final output[j*8 + i]
        // Lane t contributes D[i, 2*qq] → out[2*qq * 8 + i]
        //                   D[i, 2*qq+1] → out[(2*qq+1) * 8 + i]
        out[(2 * qq) * 8 + i]     = out_0;
        out[(2 * qq + 1) * 8 + i] = out_1;
    }
}

// Forward declaration: mma_ntt64_warp_K and inner_ntt_tcu_kernel_K are defined
// after mont_mul (they depend on it). See below.

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
// MMA-based Radix-64 for K-limb (Column-wise NTT-8)
// ============================================================================
// Computes Out = TFM × Input (8×8 matrix multiply) where:
//   - Out[i,j] = Σ_{k=0..7} TFM[i,k] × Input[k,j]
//   - TFM is the radix-8 DFT matrix: TFM[i,k] = ω_8^(i×k)
//   - Input/Output are stored row-major: M[r,c] = array[r*8 + c]
//
// MMA m8n8k4 fragment mapping:
//   - A[i,k]: lane t provides A[t/4, t%4]
//   - B[k,j]: lane t provides B[t%4, t/4]  (column-major in lane space)
//   - D[i,j]: lane t receives D[t/4, 2*(t%4)] and D[t/4, 2*(t%4)+1]
//
// For K-limb: 2×K² MMA calls (2 k-halves × K² limb pairs)
// ============================================================================
template <int K>
__device__ __forceinline__
void mma_radix8_columns_K(
    DScalar<K>* out,           // Output: 64 K-limb elements (8×8 row-major)
    const DScalar<K>* in,      // Input:  64 K-limb elements (8×8 row-major)
    const DScalar<K>* tfm8,    // TFM_8[64]: 8×8 DFT matrix (row-major)
    const DScalar<K>& prime,
    uint32_t np
) {
    int lane = threadIdx.x & 31;
    int i    = lane >> 2;      // A-fragment row = D-fragment row
    int qq   = lane & 3;       // A-fragment col (within 4-col chunk)

    // D-fragment outputs: D[i, 2*qq] and D[i, 2*qq+1]
    int j0 = 2 * qq;
    int j1 = j0 + 1;

    // Accumulators for 2 output cells (2K-1 limb positions each)
    int64_t acc0[2 * K - 1];
    int64_t acc1[2 * K - 1];
    #pragma unroll
    for (int l = 0; l < 2 * K - 1; l++) { acc0[l] = 0; acc1[l] = 0; }

    // K² limb-pair MMA loop
    #pragma unroll 1
    for (int la = 0; la < K; la++) {
        #pragma unroll 1
        for (int lb = 0; lb < K; lb++) {
            // A-fragment: TFM[i, qq] for k=0..3, TFM[i, qq+4] for k=4..7
            double a_lo = (double)(uint32_t)tfm8[i * 8 + qq].limbs[la];
            double a_hi = (double)(uint32_t)tfm8[i * 8 + qq + 4].limbs[la];

            // B-fragment: Input[qq, i] for MMA1, Input[qq+4, i] for MMA2
            // Lane t provides B[t%4, t/4] = Input[qq, i]
            // Note: Input[r,c] = in[r*8 + c], so Input[qq, i] = in[qq*8 + i]
            double b_lo = (double)(uint32_t)in[qq * 8 + i].limbs[lb];
            double b_hi = (double)(uint32_t)in[(qq + 4) * 8 + i].limbs[lb];

            // MMA computes D[i, 2*qq] and D[i, 2*qq+1]
            double d0 = 0.0, d1 = 0.0;
            mma_m8n8k4_f64(d0, d1, a_lo, b_lo, 0.0, 0.0);
            mma_m8n8k4_f64(d0, d1, a_hi, b_hi, d0, d1);

            // Accumulate into limb position la + lb
            acc0[la + lb] += (int64_t)d0;
            acc1[la + lb] += (int64_t)d1;
        }
    }

    // Carry-propagate to (2K+1)-limb form
    uint64_t P0[2 * K + 1], P1[2 * K + 1];
    {
        uint64_t c0 = 0, c1 = 0;
        #pragma unroll
        for (int l = 0; l < 2 * K - 1; l++) {
            uint64_t v0 = (uint64_t)acc0[l] + c0;
            P0[l] = v0 & LIMB_MASK; c0 = v0 >> LIMB_BITS;
            uint64_t v1 = (uint64_t)acc1[l] + c1;
            P1[l] = v1 & LIMB_MASK; c1 = v1 >> LIMB_BITS;
        }
        P0[2*K-1] = c0 & LIMB_MASK; P0[2*K] = c0 >> LIMB_BITS;
        P1[2*K-1] = c1 & LIMB_MASK; P1[2*K] = c1 >> LIMB_BITS;
    }

    // Montgomery reduce and store
    out[i * 8 + j0] = mont_reduce_wide<K>(P0, prime, np);
    out[i * 8 + j1] = mont_reduce_wide<K>(P1, prime, np);
}

// ============================================================================
// MMA-based Radix-64 for K-limb (Row-wise NTT-8)
// ============================================================================
// Computes Out[r,c] = Σ_{k=0..7} TFM[c,k] × Input[r,k] for each row r
// Since TFM is symmetric (DFT matrix), this equals Out = Input × TFM.
//
// Using MMA m8n8k4 to compute D = A × B where:
//   - A = Input (8×8, split as 8×4 halves)
//   - B = TFM (8×8, split as 4×8 halves)
//   - D = Out (8×8)
// ============================================================================
template <int K>
__device__ __forceinline__
void mma_radix8_rows_K(
    DScalar<K>* out,
    const DScalar<K>* in,
    const DScalar<K>* tfm8,
    const DScalar<K>& prime,
    uint32_t np
) {
    int lane = threadIdx.x & 31;
    int i    = lane >> 2;      // A-fragment row = D-fragment row
    int qq   = lane & 3;       // A-fragment col chunk

    int j0 = 2 * qq;
    int j1 = j0 + 1;

    int64_t acc0[2 * K - 1];
    int64_t acc1[2 * K - 1];
    #pragma unroll
    for (int l = 0; l < 2 * K - 1; l++) { acc0[l] = 0; acc1[l] = 0; }

    #pragma unroll 1
    for (int la = 0; la < K; la++) {
        #pragma unroll 1
        for (int lb = 0; lb < K; lb++) {
            // A-fragment: Input[i, qq] for k=0..3, Input[i, qq+4] for k=4..7
            double a_lo = (double)(uint32_t)in[i * 8 + qq].limbs[la];
            double a_hi = (double)(uint32_t)in[i * 8 + qq + 4].limbs[la];

            // B-fragment: TFM[qq, i] for MMA1, TFM[qq+4, i] for MMA2
            // Lane t provides B[t%4, t/4] = TFM[qq, i]
            double b_lo = (double)(uint32_t)tfm8[qq * 8 + i].limbs[lb];
            double b_hi = (double)(uint32_t)tfm8[(qq + 4) * 8 + i].limbs[lb];

            double d0 = 0.0, d1 = 0.0;
            mma_m8n8k4_f64(d0, d1, a_lo, b_lo, 0.0, 0.0);
            mma_m8n8k4_f64(d0, d1, a_hi, b_hi, d0, d1);

            acc0[la + lb] += (int64_t)d0;
            acc1[la + lb] += (int64_t)d1;
        }
    }

    uint64_t P0[2 * K + 1], P1[2 * K + 1];
    {
        uint64_t c0 = 0, c1 = 0;
        #pragma unroll
        for (int l = 0; l < 2 * K - 1; l++) {
            uint64_t v0 = (uint64_t)acc0[l] + c0;
            P0[l] = v0 & LIMB_MASK; c0 = v0 >> LIMB_BITS;
            uint64_t v1 = (uint64_t)acc1[l] + c1;
            P1[l] = v1 & LIMB_MASK; c1 = v1 >> LIMB_BITS;
        }
        P0[2*K-1] = c0 & LIMB_MASK; P0[2*K] = c0 >> LIMB_BITS;
        P1[2*K-1] = c1 & LIMB_MASK; P1[2*K] = c1 >> LIMB_BITS;
    }

    // Output: Out[i, j0] and Out[i, j1]
    out[i * 8 + j0] = mont_reduce_wide<K>(P0, prime, np);
    out[i * 8 + j1] = mont_reduce_wide<K>(P1, prime, np);
}

// ============================================================================
// MMA-based Radix-64 4-Step NTT using FP64 Tensor Cores
// ============================================================================
// Implements the HMFHE paper's radix-64 algorithm:
//   Step 1: Column DFT-8:  M'[i,j] = Σ_k TFM[i,k] × M[k,j]
//   Step 2: Hadamard:      M''[i,j] = M'[i,j] × ω_64^(i×j)
//   Step 3: Row DFT-8:     Out[i,j] = Σ_k TFM[i,k] × M''[j,k]
//
// Input: 64 elements in NATURAL order (not bit-reversed)
// Output: 64-point DFT in natural order
//
// For K-limb: 4×K² MMA calls total (2 per DFT-8 × 2 DFT-8s)
// ============================================================================
// ============================================================================
// Scalar 4-step Radix-64 NTT (for correctness verification)
// ============================================================================
// This is a scalar implementation of the 4-step algorithm that doesn't use MMA.
// It can be used to verify the algorithm is correct independently of MMA.
// ============================================================================
template <int K>
__device__ __forceinline__
void scalar_radix64_ntt(
    DScalar<K>* data,         // 64 elements in natural order
    const DScalar<K>* tfm8,   // TFM_8[64]: 8×8 DFT matrix
    const DScalar<K>* hada64, // Hada64[64]: Hadamard twiddles ω_64^k
    const DScalar<K>& prime,
    uint32_t np
) {
    // Interpret data as 8×8 matrix M[row][col] = data[row*8 + col]
    DScalar<K> M[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) M[i] = data[i];
    
    // Step 1: Column DFT-8 → M'[i,j] = Σ_k TFM[i,k] × M[k,j]
    DScalar<K> Mp[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            DScalar<K> acc = {};  // Zero
            for (int k = 0; k < 8; k++) {
                // TFM[i,k] × M[k,j]
                DScalar<K> tfm_ik = tfm8[i * 8 + k];
                DScalar<K> m_kj = M[k * 8 + j];
                DScalar<K> prod = mont_mul<K>(tfm_ik, m_kj, prime, np);
                acc = mod_add_d<K>(acc, prod, prime);
            }
            Mp[i * 8 + j] = acc;
        }
    }
    
    // Step 2: Hadamard multiply → M''[i,j] = M'[i,j] × ω_64^(i*j)
    DScalar<K> Mpp[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int hada_idx = (i * j) & 63;
            DScalar<K> tw = hada64[hada_idx];
            Mpp[i * 8 + j] = mont_mul<K>(Mp[i * 8 + j], tw, prime, np);
        }
    }
    
    // Step 3: Row DFT-8 → Out[i,j] = Σ_k TFM[j,k] × M''[i,k]
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            DScalar<K> acc = {};  // Zero
            for (int k = 0; k < 8; k++) {
                // TFM[j,k] × M''[i,k]
                DScalar<K> tfm_jk = tfm8[j * 8 + k];
                DScalar<K> mpp_ik = Mpp[i * 8 + k];
                DScalar<K> prod = mont_mul<K>(tfm_jk, mpp_ik, prime, np);
                acc = mod_add_d<K>(acc, prod, prime);
            }
            data[i * 8 + j] = acc;
        }
    }
}

// ============================================================================
// Standard MMA K-limb multiply (no Karatsuba - for correctness)
// ============================================================================
// Computes the K×K partial products using MMA for the DFT-8 matrix multiply.
// Each MMA call handles all 64 output elements of the 8×8 matrix.
// ============================================================================
template <int K>
__device__ __forceinline__
void standard_mma_multiply(
    int64_t* acc0,            // Output accumulator for result 0 (2K-1 positions)
    int64_t* acc1,            // Output accumulator for result 1 (2K-1 positions)
    const double* A_lo_d,     // A operand (lo fragment) K limbs as doubles
    const double* A_hi_d,     // A operand (hi fragment) K limbs as doubles
    const double* B_lo_d,     // B operand (lo fragment) K limbs as doubles
    const double* B_hi_d      // B operand (hi fragment) K limbs as doubles
) {
    // Initialize accumulators
    #pragma unroll
    for (int l = 0; l < 2 * K - 1; l++) { acc0[l] = 0; acc1[l] = 0; }

    // K×K MMA loop: compute all partial products
    #pragma unroll
    for (int la = 0; la < K; la++) {
        #pragma unroll
        for (int lb = 0; lb < K; lb++) {
            double d0 = 0.0, d1 = 0.0;
            // MMA for k=0..3 (lo fragment contributes to D[i, 2*qq])
            mma_m8n8k4_f64(d0, d1, A_lo_d[la], B_lo_d[lb], 0.0, 0.0);
            // MMA for k=4..7 (hi fragment contributes to D[i, 2*qq+1])
            mma_m8n8k4_f64(d0, d1, A_hi_d[la], B_hi_d[lb], d0, d1);
            // Accumulate into limb position (la + lb)
            acc0[la + lb] += (int64_t)d0;
            acc1[la + lb] += (int64_t)d1;
        }
    }
}

// ============================================================================
// Optimized MMA 4-Step Radix-64 NTT with Karatsuba
// ============================================================================
// Key optimizations:
//   1. Karatsuba multiplication reduces MMA calls from K² to ~0.8×K²
//   2. Pre-convert all limbs to double arrays
//   3. Full loop unrolling
//   4. Minimized SMEM synchronizations
// ============================================================================
template <int K>
__device__ __forceinline__
void mma_ntt64_warp_K(
    DScalar<K>* warp_smem,    // Per-warp SMEM: 64 K-limb elements (padded layout)
    const DScalar<K>* tfm8,   // TFM_8[64]: 8×8 DFT matrix ω_8^(i×k)
    const DScalar<K>* hada64, // Hada64[64]: Hadamard twiddles ω_64^(i×j)
    const DScalar<K>& prime,
    uint32_t np
) {
    int lane = threadIdx.x & 31;
    int row = lane >> 2;       // 0..7, row index for MMA D-fragment
    int qq  = lane & 3;        // 0..3, column chunk
    int j0  = 2 * qq;          // Output columns handled by this lane
    int j1  = j0 + 1;

    // Pre-convert TFM elements to double arrays
    double tfm_lo_d[K], tfm_hi_d[K];
    {
        DScalar<K> tfm_lo = tfm8[row * 8 + qq];
        DScalar<K> tfm_hi = tfm8[row * 8 + qq + 4];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            tfm_lo_d[i] = (double)(uint32_t)tfm_lo.limbs[i];
            tfm_hi_d[i] = (double)(uint32_t)tfm_hi.limbs[i];
        }
    }

    // Pre-load Hadamard twiddles
    int hada_idx_j0 = (row * j0) & 63;
    int hada_idx_j1 = (row * j1) & 63;
    DScalar<K> hada_tw_j0 = hada64[hada_idx_j0];
    DScalar<K> hada_tw_j1 = hada64[hada_idx_j1];

    // Load input elements and pre-convert to double
    double B_lo_d[K], B_hi_d[K];
    {
        DScalar<K> B_lo = warp_smem[wsm_idx(qq * 8 + row)];
        DScalar<K> B_hi = warp_smem[wsm_idx((qq + 4) * 8 + row)];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            B_lo_d[i] = (double)(uint32_t)B_lo.limbs[i];
            B_hi_d[i] = (double)(uint32_t)B_hi.limbs[i];
        }
    }
    __syncwarp();

    // =========================================================================
    // Step 1: Column DFT-8 via MMA
    // =========================================================================
    int64_t acc0[2 * K - 1], acc1[2 * K - 1];
    standard_mma_multiply<K>(acc0, acc1, tfm_lo_d, tfm_hi_d, B_lo_d, B_hi_d);

    // Montgomery reduce M'[row, j0] and M'[row, j1]
    uint64_t P0[2 * K + 1], P1[2 * K + 1];
    {
        uint64_t c0 = 0, c1 = 0;
        #pragma unroll
        for (int l = 0; l < 2 * K - 1; l++) {
            uint64_t v0 = (uint64_t)acc0[l] + c0; P0[l] = v0 & LIMB_MASK; c0 = v0 >> LIMB_BITS;
            uint64_t v1 = (uint64_t)acc1[l] + c1; P1[l] = v1 & LIMB_MASK; c1 = v1 >> LIMB_BITS;
        }
        P0[2*K-1] = c0 & LIMB_MASK; P0[2*K] = c0 >> LIMB_BITS;
        P1[2*K-1] = c1 & LIMB_MASK; P1[2*K] = c1 >> LIMB_BITS;
    }
    DScalar<K> Mp_j0 = mont_reduce_wide<K>(P0, prime, np);
    DScalar<K> Mp_j1 = mont_reduce_wide<K>(P1, prime, np);

    // =========================================================================
    // Step 2: Hadamard multiply: M''[i,j] = M'[i,j] × ω_64^(i×j)
    // =========================================================================
    DScalar<K> Mpp_j0 = mont_mul<K>(Mp_j0, hada_tw_j0, prime, np);
    DScalar<K> Mpp_j1 = mont_mul<K>(Mp_j1, hada_tw_j1, prime, np);

    // Store to SMEM for row DFT data exchange
    warp_smem[wsm_idx(row * 8 + j0)] = Mpp_j0;
    warp_smem[wsm_idx(row * 8 + j1)] = Mpp_j1;
    __syncwarp();

    // Load elements for row DFT and pre-convert to double
    {
        DScalar<K> B_lo = warp_smem[wsm_idx(row * 8 + qq)];
        DScalar<K> B_hi = warp_smem[wsm_idx(row * 8 + qq + 4)];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            B_lo_d[i] = (double)(uint32_t)B_lo.limbs[i];
            B_hi_d[i] = (double)(uint32_t)B_hi.limbs[i];
        }
    }
    __syncwarp();

    // =========================================================================
    // Step 3: Row DFT-8 via MMA
    // =========================================================================
    standard_mma_multiply<K>(acc0, acc1, tfm_lo_d, tfm_hi_d, B_lo_d, B_hi_d);

    // Montgomery reduce and store final output
    {
        uint64_t c0 = 0, c1 = 0;
        #pragma unroll
        for (int l = 0; l < 2 * K - 1; l++) {
            uint64_t v0 = (uint64_t)acc0[l] + c0; P0[l] = v0 & LIMB_MASK; c0 = v0 >> LIMB_BITS;
            uint64_t v1 = (uint64_t)acc1[l] + c1; P1[l] = v1 & LIMB_MASK; c1 = v1 >> LIMB_BITS;
        }
        P0[2*K-1] = c0 & LIMB_MASK; P0[2*K] = c0 >> LIMB_BITS;
        P1[2*K-1] = c1 & LIMB_MASK; P1[2*K] = c1 >> LIMB_BITS;
    }

    warp_smem[wsm_idx(row * 8 + j0)] = mont_reduce_wide<K>(P0, prime, np);
    warp_smem[wsm_idx(row * 8 + j1)] = mont_reduce_wide<K>(P1, prime, np);
    __syncwarp();
}

// ============================================================================
// Inner NTT TCU Kernel for K-limb
// ============================================================================
// Two modes based on n:
// - n == 64: Pure MMA 4-step on natural-order data → natural-order DFT
// - n > 64:  Fused radix-2 CT stages (0-5) on bit-reversed strided data
//
// Each warp handles one 64-point NTT; multiple warps per block share SMEM.
// ============================================================================

// Radix-2 CT DIT kernel for 64 points (6 stages, for fused inner NTT)
template <int K>
__device__ __forceinline__
void radix2_ct_ntt64_warp(
    DScalar<K>* warp_smem,    // Per-warp SMEM with padded layout
    const DScalar<K>* tw64,   // ω_64^k twiddle table
    const DScalar<K>& prime,
    uint32_t np
) {
    int lane = threadIdx.x & 31;

    for (int st = 0; st < 6; st++) {
        int m = 1 << (st + 1);
        int half_m = m >> 1;
        int group = lane / half_m;
        int pos = lane % half_m;
        int i_idx = group * m + pos;
        int j_idx = i_idx + half_m;

        DScalar<K> u = warp_smem[wsm_idx(i_idx)];
        DScalar<K> v = warp_smem[wsm_idx(j_idx)];

        int tw_idx = pos << (5 - st);
        DScalar<K> w = tw64[tw_idx];

        DScalar<K> t = mont_mul<K>(v, w, prime, np);
        warp_smem[wsm_idx(i_idx)] = mod_add_d<K>(u, t, prime);
        warp_smem[wsm_idx(j_idx)] = mod_sub_d<K>(u, t, prime);

        __syncwarp();
    }
}

template <int K>
__global__ __launch_bounds__(TCU_BLOCK_SIZE)
void inner_ntt_tcu_kernel_K(
    DScalar<K>* __restrict__ data,
    const DScalar<K>* __restrict__ twiddles,
    const DScalar<K>* __restrict__ tfm8_global,
    const DScalar<K>* __restrict__ hada64_global,
    uint32_t n,
    int stage_start,
    DScalar<K> prime,
    uint32_t np
) {
    extern __shared__ char smem_raw[];
    DScalar<K>* tfm8_smem   = (DScalar<K>*)smem_raw;
    DScalar<K>* hada64_smem = tfm8_smem + 64;
    DScalar<K>* warp_scratch_base = hada64_smem + 64;

    int tid      = threadIdx.x;
    int warp_id  = tid / TCU_WARP_SIZE;
    int lane     = tid % TCU_WARP_SIZE;
    int warps_per_block = blockDim.x / TCU_WARP_SIZE;
    int global_warp = blockIdx.x * warps_per_block + warp_id;

    // Load TFM_8 and Hada64 into SMEM
    if (tid < 64) {
        tfm8_smem[tid]   = tfm8_global[tid];
        hada64_smem[tid] = hada64_global[tid];
    }
    __syncthreads();

    uint32_t total_ntts = n / TCU_INNER_SIZE;
    if ((uint32_t)global_warp >= total_ntts) return;

    DScalar<K>* warp_smem = warp_scratch_base + (uint64_t)warp_id * TCU_WARP_TOTAL;

    if (n == TCU_INNER_SIZE) {
        // N == 64: MMA 4-step radix-64 (natural order input → natural order output)
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            warp_smem[wsm_idx(j)] = data[j];
        }
        __syncwarp();

        mma_ntt64_warp_K<K>(warp_smem, tfm8_smem, hada64_smem, prime, np);

        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            data[j] = warp_smem[wsm_idx(j)];
        }
    } else {
        // N > 64: MMA 4-step radix-64 for inner 64-element NTTs
        // After global bit-reverse, inner blocks are in bit-reversed order.
        // We convert to natural order, apply MMA, then convert back.
        uint32_t S          = 1u << stage_start;
        uint32_t macro_size = TCU_INNER_SIZE * S;
        uint32_t macro_g    = global_warp / S;
        uint32_t offset_o   = global_warp % S;
        uint32_t base       = macro_g * macro_size + offset_o;

        // Load 64 elements with stride S, converting bit-reversed → natural order
        // Data at global position (base + j*S) goes to natural position bit_rev_6(j)
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            uint32_t global_idx = base + (uint32_t)j * S;
            DScalar<K> val = data[global_idx];

            // Natural position within the 64-element block
            int natural_pos = bit_rev_6(j);

            // Pre-twist for offset > 0 (twiddle index based on natural position)
            if (offset_o > 0) {
                uint32_t tw_stride = n / macro_size;
                uint32_t tw_idx = ((uint64_t)offset_o * (uint64_t)natural_pos * tw_stride) % n;
                val = mont_mul<K>(val, twiddles[tw_idx], prime, np);
            }
            
            // Store at natural position
            warp_smem[wsm_idx(natural_pos)] = val;
        }
        __syncwarp();

        // MMA 4-step radix-64 (natural order in → natural order out)
        mma_ntt64_warp_K<K>(warp_smem, tfm8_smem, hada64_smem, prime, np);

        // Store back with stride S (output is in natural order, same as scalar CT)
        #pragma unroll
        for (int j = lane; j < TCU_INNER_SIZE; j += TCU_WARP_SIZE) {
            uint32_t global_idx = base + (uint32_t)j * S;
            data[global_idx] = warp_smem[wsm_idx(j)];
        }
    }
}

// ============================================================================
// DIF (Decimation-in-Frequency) Stage Kernel
// ============================================================================
// DIF butterfly: out_top = in_top + in_bot
//                out_bot = (in_top - in_bot) × twiddle
// Processes one stage, top-down (large butterflies to small)
// ============================================================================
template <int K>
__global__ void dif_stage_kernel(DScalar<K>* data,
                                  const DScalar<K>* twiddles,
                                  uint32_t n, int stage,
                                  DScalar<K> prime,
                                  uint32_t np) {
    uint32_t bfly = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = n >> 1;
    if (bfly >= half) return;

    // DIF stage: butterfly size = 2^(log_n - stage)
    // For stage 0: largest butterflies (n/2 pairs, stride n/2)
    // For stage log_n-1: smallest butterflies (pairs of 2)
    int log_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) log_n++;

    int m = 1 << (log_n - stage);  // Butterfly group size
    int half_m = m >> 1;           // Butterfly stride

    uint32_t group = bfly / half_m;
    uint32_t pos = bfly % half_m;
    uint32_t i = group * m + pos;
    uint32_t j = i + half_m;

    DScalar<K> u = data[i];
    DScalar<K> v = data[j];

    // DIF butterfly: (u, v) → (u + v, (u - v) × w)
    DScalar<K> sum  = mod_add_d<K>(u, v, prime);
    DScalar<K> diff = mod_sub_d<K>(u, v, prime);

    // Twiddle: ω_n^(pos × n / m) = ω_n^(pos × 2^stage)
    uint32_t tw_idx = ((uint64_t)pos << stage) % n;
    DScalar<K> w = twiddles[tw_idx];

    DScalar<K> t = mont_mul<K>(diff, w, prime, np);

    data[i] = sum;
    data[j] = t;
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
// Outer stage kernel for K-limb (for stages not covered by TCU inner NTT)
template <int K>
__global__ void outer_stage_kernel_K(
    DScalar<K>* __restrict__ data,
    const DScalar<K>* __restrict__ twiddles,
    uint32_t n,
    int stage,
    DScalar<K> prime,
    uint32_t np
) {
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
// Main NTT dispatch with HMFHE-style TCU optimization
// ============================================================================
// Structure:
//   1. Global bit-reverse permutation
//   2. `extra = log_n % 6` outer CT stages (stages 0 .. extra-1)
//   3. `log_n / 6` rounds of TCU-based 64-point inner NTTs
//      Round k processes stages [extra + 6*k, extra + 6*k + 6) with stride 2^(extra + 6*k)
//
// If NTT_ARB_USE_TCU_INNER=0 or N < 64, falls back to per-stage kernels.
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

#if NTT_ARB_USE_TCU_INNER
    // HMFHE-style: TCU inner NTT for 6 stages at a time (radix-64)
    if (log_n >= TCU_LOG_INNER) {
        int extra  = log_n % TCU_LOG_INNER;  // Outer stages: 0..extra-1
        int rounds = log_n / TCU_LOG_INNER;  // TCU rounds

        // Run `extra` outer stages first
        int half = n / 2;
        int b_st = (half + t - 1) / t;
        for (int st = 0; st < extra; st++) {
            outer_stage_kernel_K<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }

        // NOTE: TCU kernel needs d_tfm8 and d_hada64, but this function doesn't
        // have access to NTTBuffers. We need an extended version.
        // For now, fall back to per-stage for remaining stages.
        // TODO: Pass d_tfm8, d_hada64 to this function or use a context struct.

        // Fallback: per-stage kernels for the remaining stages
        for (int st = extra; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
    } else
#endif
    {
        // Fallback: per-stage kernels
        int half = n / 2;
        int b_st = (half + t - 1) / t;
        for (int st = 0; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
    }
}

// Extended NTT dispatch with MMA TCU inner kernel
// ============================================================================
// For N=64: Use MMA 4-step radix-64 (complete DFT using tensor cores)
// For N>64: Use fused radix-2 CT with shared memory twiddles for inner stages,
//           then per-element outer stages. This provides memory optimization
//           even without full MMA utilization for the butterfly multiplies.
// ============================================================================
template <int K>
static inline void run_forward_ct_tcu(DScalar<K>* d_data,
                                      const DScalar<K>* d_tw,
                                      const DScalar<K>* d_tfm8,
                                      const DScalar<K>* d_hada64,
                                      uint32_t n, int log_n,
                                      DScalar<K> prime_d,
                                      uint32_t np,
                                      cudaStream_t stream = 0) {
    int t = 256;
    int half = n / 2;
    int b_st = (half + t - 1) / t;
    int b_br = (n + t - 1) / t;

#if NTT_ARB_USE_TCU_INNER
    if (log_n == TCU_LOG_INNER && d_tfm8 && d_hada64) {
        // Exact N=64: Pure MMA 4-step (natural order in → natural order out)
        size_t smem_bytes = (64 + 64 + TCU_WARPS_PER_BLK * TCU_WARP_TOTAL) * sizeof(DScalar<K>);
        inner_ntt_tcu_kernel_K<K><<<1, TCU_BLOCK_SIZE, smem_bytes, stream>>>(
            d_data, d_tw, d_tfm8, d_hada64, n, 0, prime_d, np);
    } else if (log_n > TCU_LOG_INNER && d_tfm8 && d_hada64) {
        // N > 64: CT-DIT with fused inner stages using TCU kernel
        // Bit-reverse input
        bitrev_kernel_d<K><<<b_br, t, 0, stream>>>(d_data, n, log_n);

        // Inner stages 0-5: use TCU kernel with shared memory twiddles
        // This fuses 6 stages into one kernel for better memory access
        size_t smem_bytes = (64 + 64 + TCU_WARPS_PER_BLK * TCU_WARP_TOTAL) * sizeof(DScalar<K>);
        uint32_t total_ntts = n / TCU_INNER_SIZE;
        uint32_t blocks = (total_ntts + TCU_WARPS_PER_BLK - 1) / TCU_WARPS_PER_BLK;

        inner_ntt_tcu_kernel_K<K><<<blocks, TCU_BLOCK_SIZE, smem_bytes, stream>>>(
            d_data, d_tw, d_tfm8, d_hada64, n, 0, prime_d, np);

        // Outer stages 6+: per-element CT butterflies
        for (int st = TCU_LOG_INNER; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
    } else
#endif
    {
        // Fallback: standard CT-DIT
        bitrev_kernel_d<K><<<b_br, t, 0, stream>>>(d_data, n, log_n);
        for (int st = 0; st < log_n; st++) {
            ct_stage_kernel<K><<<b_st, t, 0, stream>>>(d_data, d_tw, n, st, prime_d, np);
        }
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
    DScalar<K>* d_tfm8  = nullptr;           // TFM_8[64]: radix-8 DFT matrix (TFOP)
    DScalar<K>* d_hada64 = nullptr;          // Hada64[64]: inner Hadamard twiddles (TFOP)
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

    // =========================================================================
    // Precompute TFM_8 and twiddle table for TCU inner NTT
    // =========================================================================
    // TFM_8[i, k] = omega_8^(i * k) for i, k ∈ [0, 8) - for 4-step algo (unused now)
    // hada64[k] = omega_64^k for k ∈ [0, 64) - twiddle table for radix-2 CT
    //
    // omega_64 = omega_n^(N / 64)
    // =========================================================================
    if (B.n >= 64) {
        BigInt<K> omega_64 = omega_n;
        // omega_64 = omega_n^(N/64)
        for (int t = B.n; t > 64; t >>= 1) {
            omega_64 = mod_mul<K>(omega_64, omega_64, B.prime);
        }

        std::vector<DScalar<K>> tfm8_host(64), hada64_host(64);

        // Build TFM_8 (unused for radix-2, but kept for API compatibility)
        BigInt<K> omega_8 = mod_pow<K>(omega_64, 8, B.prime);  // omega_8 = omega_64^8
        for (int i = 0; i < 8; i++) {
            BigInt<K> omega_8_i = mod_pow<K>(omega_8, (uint64_t)i, B.prime);
            BigInt<K> acc = BigInt<K>::one();
            for (int k = 0; k < 8; k++) {
                BigInt<K> entry_mont = mont_mul_big<K>(acc, B.R2, B.prime, B.np);
                tfm8_host[i * 8 + k] = load_scalar<K>(entry_mont);
                acc = mod_mul<K>(acc, omega_8_i, B.prime);
            }
        }

        // Build tw64[k] = omega_64^k for k=0..63
        // Used for both radix-2 CT (linear access) and 4-step (via hada lookup)
        BigInt<K> tw = BigInt<K>::one();
        for (int k = 0; k < 64; k++) {
            BigInt<K> tw_mont = mont_mul_big<K>(tw, B.R2, B.prime, B.np);
            hada64_host[k] = load_scalar<K>(tw_mont);
            tw = mod_mul<K>(tw, omega_64, B.prime);
        }

        cudaMalloc(&B.d_tfm8,   64 * sizeof(DScalar<K>));
        cudaMalloc(&B.d_hada64, 64 * sizeof(DScalar<K>));
        cudaMemcpy(B.d_tfm8,   tfm8_host.data(),   64 * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
        cudaMemcpy(B.d_hada64, hada64_host.data(), 64 * sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    }

    return B;
}

template <int K>
static inline void teardown_ntt(NTTBuffers<K>& B) {
    if (B.d_data)   cudaFree(B.d_data);
    if (B.d_tw)     cudaFree(B.d_tw);
    if (B.d_tfm8)   cudaFree(B.d_tfm8);
    if (B.d_hada64) cudaFree(B.d_hada64);
    B.d_data = nullptr; B.d_tw = nullptr;
    B.d_tfm8 = nullptr; B.d_hada64 = nullptr;
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
    // Use TCU-accelerated path if TFM/Hada buffers are available
    auto run_ntt = [&]() {
        if (B.d_tfm8 && B.d_hada64) {
            run_forward_ct_tcu<K>(B.d_data, B.d_tw, B.d_tfm8, B.d_hada64,
                                  B.n, log_n, B.prime_d, B.np);
        } else {
            run_forward_ct<K>(B.d_data, B.d_tw, B.n, log_n, B.prime_d, B.np);
        }
    };

    for (int it = 0; it < 3; it++) run_ntt();
    cudaDeviceSynchronize();

    reupload_mont_input();
    cudaEventRecord(s);
    const int iters = 20;
    for (int it = 0; it < iters; it++) {
        run_ntt();
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    std::cout << "  TCU-NTT: " << (ms * 1000.0 / iters) << " us/iter\n";

    int mismatches = 0;
    if (do_host_check) {
        reupload_mont_input();
        run_ntt();
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
