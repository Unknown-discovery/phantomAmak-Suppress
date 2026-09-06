#include "dct_kernels.hpp"
#include "binary.hpp"

#include <algorithm>

#ifndef AMAK_ENABLE_AVX2
#define AMAK_ENABLE_AVX2 1
#endif

// Per-function targeting, NOT a translation-unit-wide -mavx2 requirement.
// Unsupported architectures/compilers build the reference functions only.
#if AMAK_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define AMAK_HAS_AVX2_KERNELS 1
#include <immintrin.h>
#define AMAK_AVX2_TARGET __attribute__((target("avx2")))
#else
#define AMAK_HAS_AVX2_KERNELS 0
#endif

namespace amak {
namespace {

void forward_scalar_f64(const double* basis, std::size_t n,
                        const double* input, double* output) {
    for (std::size_t k = 0; k < n; ++k) {
        double sum = 0;
        for (std::size_t j = 0; j < n; ++j) sum += basis[k * n + j] * input[j];
        output[k] = sum;
    }
}

void forward_scalar_f32(const double* basis, std::size_t n,
                        const float* input, double* output) {
    for (std::size_t k = 0; k < n; ++k) {
        double sum = 0;
        for (std::size_t j = 0; j < n; ++j)
            sum += basis[k * n + j] * static_cast<double>(input[j]);
        output[k] = sum;
    }
}

void inverse_scalar_f64(const double* basis, std::size_t n,
                        const double* input, double* output) {
    for (std::size_t j = 0; j < n; ++j) {
        double sum = 0;
        for (std::size_t k = 0; k < n; ++k) sum += basis[k * n + j] * input[k];
        output[j] = sum;
    }
}

#if AMAK_HAS_AVX2_KERNELS
AMAK_AVX2_TARGET inline double horizontal_sum(__m256d lanes) {
    const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(lanes),
                                   _mm256_extractf128_pd(lanes, 1));
    return _mm_cvtsd_f64(pair) + _mm_cvtsd_f64(_mm_unpackhi_pd(pair, pair));
}

AMAK_AVX2_TARGET void forward_avx2_f64(const double* basis, std::size_t n,
                                      const double* input, double* output) {
    for (std::size_t k = 0; k < n; ++k) {
        const auto* row = basis + k * n;
        __m256d a = _mm256_setzero_pd(), b = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            a = _mm256_add_pd(a, _mm256_mul_pd(_mm256_loadu_pd(row + j), _mm256_loadu_pd(input + j)));
            b = _mm256_add_pd(b, _mm256_mul_pd(_mm256_loadu_pd(row + j + 4), _mm256_loadu_pd(input + j + 4)));
        }
        a = _mm256_add_pd(a, b);
        for (; j + 4 <= n; j += 4)
            a = _mm256_add_pd(a, _mm256_mul_pd(_mm256_loadu_pd(row + j), _mm256_loadu_pd(input + j)));
        double sum = horizontal_sum(a);
        for (; j < n; ++j) sum += row[j] * input[j]; // no over-read on edge tiles
        output[k] = sum;
    }
}

AMAK_AVX2_TARGET void forward_avx2_f32(const double* basis, std::size_t n,
                                      const float* input, double* output) {
    for (std::size_t k = 0; k < n; ++k) {
        const auto* row = basis + k * n;
        __m256d a = _mm256_setzero_pd(), b = _mm256_setzero_pd();
        std::size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(input + j));
            const __m256d x1 = _mm256_cvtps_pd(_mm_loadu_ps(input + j + 4));
            a = _mm256_add_pd(a, _mm256_mul_pd(_mm256_loadu_pd(row + j), x0));
            b = _mm256_add_pd(b, _mm256_mul_pd(_mm256_loadu_pd(row + j + 4), x1));
        }
        a = _mm256_add_pd(a, b);
        for (; j + 4 <= n; j += 4) {
            const __m256d x = _mm256_cvtps_pd(_mm_loadu_ps(input + j));
            a = _mm256_add_pd(a, _mm256_mul_pd(_mm256_loadu_pd(row + j), x));
        }
        double sum = horizontal_sum(a);
        for (; j < n; ++j) sum += row[j] * static_cast<double>(input[j]);
        output[k] = sum;
    }
}

AMAK_AVX2_TARGET void inverse_avx2_f64(const double* basis, std::size_t n,
                                      const double* input, double* output) {
    // D^T x as outer-product accumulation: contiguous basis/output loads, with
    // no transposed basis allocation. Unroll four frequencies, preserving their
    // addition order independently in every output lane. No FMA is required.
    std::fill(output, output + n, 0.0);
    std::size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        const auto* r0 = basis + k * n;
        const auto* r1 = r0 + n;
        const auto* r2 = r1 + n;
        const auto* r3 = r2 + n;
        const __m256d x0 = _mm256_set1_pd(input[k]);
        const __m256d x1 = _mm256_set1_pd(input[k + 1]);
        const __m256d x2 = _mm256_set1_pd(input[k + 2]);
        const __m256d x3 = _mm256_set1_pd(input[k + 3]);
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            __m256d sum = _mm256_loadu_pd(output + j);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(r0 + j), x0));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(r1 + j), x1));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(r2 + j), x2));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(r3 + j), x3));
            _mm256_storeu_pd(output + j, sum);
        }
        for (; j < n; ++j) {
            output[j] += r0[j] * input[k];
            output[j] += r1[j] * input[k + 1];
            output[j] += r2[j] * input[k + 2];
            output[j] += r3[j] * input[k + 3];
        }
    }
    for (; k < n; ++k) {
        const auto* row = basis + k * n;
        const __m256d x = _mm256_set1_pd(input[k]);
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4)
            _mm256_storeu_pd(output + j, _mm256_add_pd(_mm256_loadu_pd(output + j),
                _mm256_mul_pd(_mm256_loadu_pd(row + j), x)));
        for (; j < n; ++j) output[j] += row[j] * input[k];
    }
}
#endif

const detail::DctKernels scalar_kernels{
    DctBackend::scalar, forward_scalar_f64, forward_scalar_f32, inverse_scalar_f64};
#if AMAK_HAS_AVX2_KERNELS
const detail::DctKernels avx2_kernels{
    DctBackend::avx2, forward_avx2_f64, forward_avx2_f32, inverse_avx2_f64};
#endif

} // namespace

const char* dct_backend_name(DctBackend backend) noexcept {
    switch (backend) {
    case DctBackend::automatic: return "auto";
    case DctBackend::scalar: return "scalar";
    case DctBackend::avx2: return "avx2";
    }
    return "unknown";
}

bool avx2_dct_compiled() noexcept { return AMAK_HAS_AVX2_KERNELS != 0; }

bool avx2_dct_available() noexcept {
#if AMAK_HAS_AVX2_KERNELS
    // GCC/Clang account for OSXSAVE/XCR0 when reporting usable AVX features.
    // A thread-safe local static keeps probing out of subsequent hot paths.
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
    }();
    return available;
#else
    return false;
#endif
}

DctBackend resolve_dct_backend(DctBackend requested) {
    switch (requested) {
    case DctBackend::automatic:
        return avx2_dct_available() ? DctBackend::avx2 : DctBackend::scalar;
    case DctBackend::scalar:
        return DctBackend::scalar;
    case DctBackend::avx2:
        detail::require(avx2_dct_available(), "AVX2 DCT backend is unavailable on this CPU/OS or in this build");
        return DctBackend::avx2;
    }
    throw std::runtime_error("amak: invalid DCT backend");
}

namespace detail {
const DctKernels& dct_kernels(DctBackend requested) {
    const auto selected = resolve_dct_backend(requested);
#if AMAK_HAS_AVX2_KERNELS
    if (selected == DctBackend::avx2) return avx2_kernels;
#else
    (void)selected;
#endif
    return scalar_kernels;
}
} // namespace detail

} // namespace amak
