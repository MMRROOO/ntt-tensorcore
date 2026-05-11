// ============================================================================
// ntt_mma_arbitrary.cu  —  driver for the K-limb FP64-MMA NTT.
//
// All the kernels and templated helpers live in include/ntt_arbitrary.cuh so
// they can be reused by tests/icicle_arbitrary_compare.cu without duplicating
// the code. This file just provides main(): correctness sweep + simple
// per-K timing.
//
// Usage:
//   ./ntt_arbitrary                # default sweep across K=2, K=3, K=11
//   ./ntt_arbitrary 3 12           # K=3 Goldilocks, N=2^12
//   ./ntt_arbitrary 11 14          # K=11 BN254, N=2^14
//   ./ntt_arbitrary 11 16 nocheck  # skip slow host reference at large N
// ============================================================================
#include "ntt_arbitrary.cuh"

// Debug helper: verify that mont_mul(x, R2) followed by mont_mul(., 1) gets us
// back to x.  Catches mont_R2/mont_np/mont_mul issues without running the NTT.
template <int K>
static int self_test_mont(const char* tag) {
    using namespace ntt_arb;
    BigInt<K> p  = FieldConfig<K>::prime();
    BigInt<K> R2 = mont_R2<K>(p);
    uint32_t  np = mont_np<K>(p);

    auto print = [](const char* lbl, const BigInt<K>& x) {
        std::cout << "  " << lbl << " = limbs[";
        for (int i = 0; i < K; i++) {
            if (i) std::cout << ",";
            std::cout << x.limbs[i];
        }
        std::cout << "]\n";
    };

    BigInt<K> one   = BigInt<K>::one();
    BigInt<K> two   = BigInt<K>::from_u64(2);
    BigInt<K> three = BigInt<K>::from_u64(3);

    auto check = [&](const BigInt<K>& x, const char* name) {
        BigInt<K> xm  = mont_mul_big<K>(x, R2,  p, np);  // x · R
        BigInt<K> got = mont_mul_big<K>(xm, one, p, np); // (x · R) · R^{-1}
        if (got.cmp(x) != 0) {
            std::cout << "[" << tag << "] FAIL for " << name << "\n";
            print("expected", x);
            print("got     ", got);
            print("xm      ", xm);
            return 1;
        }
        return 0;
    };
    auto check_mult = [&](const BigInt<K>& a, const BigInt<K>& b, const char* name) {
        BigInt<K> am  = mont_mul_big<K>(a,  R2, p, np);   // a · R
        BigInt<K> bm  = mont_mul_big<K>(b,  R2, p, np);   // b · R
        BigInt<K> cm  = mont_mul_big<K>(am, bm, p, np);   // (a · R) · (b · R) / R = a·b·R  (expected)
        BigInt<K> got = mont_mul_big<K>(cm, one, p, np);  // a · b (mod p)
        BigInt<K> ref = mod_mul<K>(a, b, p);
        if (got.cmp(ref) != 0) {
            std::cout << "[" << tag << "] FAIL mult " << name << "\n";
            print("a       ", a);
            print("b       ", b);
            print("am      ", am);
            print("bm      ", bm);
            print("cm      ", cm);
            print("got     ", got);
            print("ref     ", ref);
            return 1;
        }
        return 0;
    };

    int errs = 0;
    errs += check(one,   "1");
    errs += check(two,   "2");
    errs += check(three, "3");
    errs += check_mult(two,   three, "2*3");
    errs += check_mult(three, three, "3*3");
    {
        BigInt<K> big_a = BigInt<K>::from_u64(0x123456789abcdefULL);
        BigInt<K> big_b = BigInt<K>::from_u64(0xfedcba987654321ULL);
        errs += check_mult(big_a, big_b, "big*big");
    }
    // Fully-populated K-limb random values - exercises every limb position.
    {
        std::mt19937_64 rng(42);
        for (int trial = 0; trial < 8; trial++) {
            BigInt<K> ra, rb;
            for (int j = 0; j < K; j++) {
                ra.limbs[j] = rng() & ntt_arb::LIMB_MASK;
                rb.limbs[j] = rng() & ntt_arb::LIMB_MASK;
            }
            ra = ntt_arb::mod_reduce_once<K>(ra, p);
            rb = ntt_arb::mod_reduce_once<K>(rb, p);
            char name[32]; std::snprintf(name, sizeof(name), "rand[%d]", trial);
            errs += check_mult(ra, rb, name);
        }
    }
    if (!errs) std::cout << "[" << tag << "] mont round-trip + mult OK\n";
    return errs;
}

// Host-side Montgomery NTT (mirror of ct_stage_kernel). If this DIFFERS from
// host_ntt_naive (which works in natural form), the bug is in our Montgomery
// algorithm/setup, not in the device kernel.
template <int K>
static void host_ntt_montgomery(std::vector<ntt_arb::BigInt<K>>& a,
                                const std::vector<ntt_arb::BigInt<K>>& tw_natural,
                                const ntt_arb::BigInt<K>& p,
                                const ntt_arb::BigInt<K>& R2,
                                uint32_t np) {
    using namespace ntt_arb;
    int n = (int)a.size();
    int log_n = 0; for (int t = n; t > 1; t >>= 1) log_n++;

    // Convert input + twiddles to Montgomery.
    std::vector<BigInt<K>> tw(n);
    for (int i = 0; i < n; i++) tw[i] = mont_mul_big<K>(tw_natural[i], R2, p, np);
    for (int i = 0; i < n; i++) a[i]  = mont_mul_big<K>(a[i],          R2, p, np);

    // Bit reverse.
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < log_n; b++) j = (j << 1) | ((i >> b) & 1);
        if (i < j) std::swap(a[i], a[j]);
    }

    // CT stages, in Montgomery form.
    for (int s = 0; s < log_n; s++) {
        int m = 1 << (s + 1);
        int half = m >> 1;
        int tw_stride = n / m;
        for (int g = 0; g < n; g += m) {
            for (int k = 0; k < half; k++) {
                BigInt<K> w = tw[(uint64_t)k * tw_stride % n];
                BigInt<K> t = mont_mul_big<K>(a[g + k + half], w, p, np);
                BigInt<K> u = a[g + k];
                a[g + k]        = mod_add<K>(u, t, p);
                a[g + k + half] = mod_sub<K>(u, t, p);
            }
        }
    }

    // Convert back to natural form.
    BigInt<K> one = BigInt<K>::one();
    for (int i = 0; i < n; i++) a[i] = mont_mul_big<K>(a[i], one, p, np);
}

// Tiny kernel that runs mont_mul on (a, b) for thread 0 and writes result.
template <int K>
__global__ void mont_mul_probe(const ntt_arb::DScalar<K>* a,
                               const ntt_arb::DScalar<K>* b,
                               ntt_arb::DScalar<K>* out,
                               ntt_arb::DScalar<K> p, uint32_t np) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[0] = ntt_arb::mont_mul<K>(a[0], b[0], p, np);
}

// Run host-mont_mul AND device-mont_mul on the same inputs and diff.
template <int K>
static void dev_mont_self_test(const char* tag) {
    using namespace ntt_arb;
    BigInt<K> p  = FieldConfig<K>::prime();
    BigInt<K> R2 = mont_R2<K>(p);
    uint32_t  np = mont_np<K>(p);
    BigInt<K> a  = BigInt<K>::from_u64(0x123456789abcdefULL);
    BigInt<K> b  = BigInt<K>::from_u64(0xfedcba987654321ULL);
    BigInt<K> am = mont_mul_big<K>(a, R2, p, np);
    BigInt<K> bm = mont_mul_big<K>(b, R2, p, np);
    DScalar<K> ad = load_scalar<K>(am);
    DScalar<K> bd = load_scalar<K>(bm);
    DScalar<K> pd = load_scalar<K>(p);

    DScalar<K> *d_a, *d_b, *d_out;
    cudaMalloc(&d_a,   sizeof(DScalar<K>));
    cudaMalloc(&d_b,   sizeof(DScalar<K>));
    cudaMalloc(&d_out, sizeof(DScalar<K>));
    cudaMemcpy(d_a, &ad, sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &bd, sizeof(DScalar<K>), cudaMemcpyHostToDevice);
    mont_mul_probe<K><<<1,32>>>(d_a, d_b, d_out, pd, np);
    cudaDeviceSynchronize();
    DScalar<K> rd_dev;
    cudaMemcpy(&rd_dev, d_out, sizeof(DScalar<K>), cudaMemcpyDeviceToHost);

    DScalar<K> rd_host = mont_mul<K>(ad, bd, pd, np);

    bool same = true;
    for (int i = 0; i < K; i++) if (rd_dev.limbs[i] != rd_host.limbs[i]) same = false;
    if (same) {
        std::cout << "[" << tag << "] device mont_mul == host mont_mul OK\n";
    } else {
        std::cout << "[" << tag << "] device mont_mul MISMATCH\n";
        std::cout << "  host: ";
        for (int i = 0; i < K; i++) std::cout << rd_host.limbs[i] << " ";
        std::cout << "\n  dev : ";
        for (int i = 0; i < K; i++) std::cout << rd_dev.limbs[i]  << " ";
        std::cout << "\n";
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "============================================================\n"
              << "GPU: " << prop.name << "  (sm_" << prop.major << prop.minor
              << ", " << prop.multiProcessorCount << " SMs)\n"
              << "LIMB_BITS=" << ntt_arb::LIMB_BITS
              << "  -- supports primes up to K*" << ntt_arb::LIMB_BITS
              << " bits per build.\n"
              << "============================================================\n";

    // Quick self-test of Montgomery round-trip per K before running NTT.
    self_test_mont<2>("K=2");
    self_test_mont<3>("K=3");
    self_test_mont<11>("K=11");

    dev_mont_self_test<2>("K=2");
    dev_mont_self_test<3>("K=3");
    dev_mont_self_test<11>("K=11");

    // Host Montgomery NTT vs natural NTT - this tells us if the Montgomery
    // algorithm itself is wrong (independent of GPU).
    auto host_mont_vs_natural = [](auto K_tag, int log_n, const char* tag) {
        constexpr int K = decltype(K_tag)::value;
        using namespace ntt_arb;
        using FC = FieldConfig<K>;
        int n = 1 << log_n;
        BigInt<K> p  = FC::prime();
        BigInt<K> R2 = mont_R2<K>(p);
        uint32_t  np = mont_np<K>(p);

        BigInt<K> omega_n = FC::omega_2pow_max();
        for (int i = 0; i < FC::MAX_LOG_N - log_n; i++)
            omega_n = mod_mul<K>(omega_n, omega_n, p);

        std::vector<BigInt<K>> tw(n);
        tw[0] = BigInt<K>::one();
        for (int i = 1; i < n; i++) tw[i] = mod_mul<K>(tw[i - 1], omega_n, p);

        std::mt19937_64 rng(0xC0FFEE);
        std::vector<BigInt<K>> data(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < K; j++) data[i].limbs[j] = rng() & LIMB_MASK;
            data[i] = mod_slow<K, K>(data[i], p);  // need full reduction, not just one subtract
        }

        std::vector<BigInt<K>> ref  = data;
        std::vector<BigInt<K>> mont = data;
        host_ntt_naive<K>(ref, tw, p);
        host_ntt_montgomery<K>(mont, tw, p, R2, np);

        int diffs = 0;
        for (int i = 0; i < n; i++) if (ref[i].cmp(mont[i]) != 0) diffs++;
        std::cout << "[" << tag << " log_n=" << log_n << "] host-mont vs host-natural: "
                  << diffs << " mismatches" << (diffs ? "" : "  OK") << "\n";
    };
    host_mont_vs_natural(std::integral_constant<int, 2>{},  6, "K=2");
    host_mont_vs_natural(std::integral_constant<int, 3>{},  6, "K=3");
    host_mont_vs_natural(std::integral_constant<int, 11>{}, 5, "K=11");

    if (argc >= 3) {
        int k = std::atoi(argv[1]);
        int ln = std::atoi(argv[2]);
        bool check = true;
        if (argc >= 4 && std::string(argv[3]) == "nocheck") check = false;
        if      (k == 2)  return ntt_arb::run_for_K<2>(ln, check);
        else if (k == 3)  return ntt_arb::run_for_K<3>(ln, check);
        else if (k == 11) return ntt_arb::run_for_K<11>(ln, check);
        else {
            std::cerr << "K=" << k << " is not specialized in this build.\n"
                      << "Add a FieldConfig<" << k << "> and rerun.\n";
            return 2;
        }
    }

    int rc = 0;
    rc |= ntt_arb::run_for_K<2>(8);
    rc |= ntt_arb::run_for_K<2>(10);
    rc |= ntt_arb::run_for_K<3>(8);
    rc |= ntt_arb::run_for_K<3>(10);
    rc |= ntt_arb::run_for_K<3>(12);
    // BN254 (K=11): do correctness at a small size, then a timing-only larger N.
    rc |= ntt_arb::run_for_K<11>(8);
    rc |= ntt_arb::run_for_K<11>(10);
    rc |= ntt_arb::run_for_K<11>(12, /*check=*/false);
    return rc;
}
