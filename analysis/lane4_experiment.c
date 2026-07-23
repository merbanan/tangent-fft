#define _POSIX_C_SOURCE 200809L

#include "fft.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#define LANE4_PI 3.141592653589793238462643383279502884f

enum { SAMPLE_COUNT = 31 };

typedef struct {
    size_t n;
    size_t m;
    unsigned levels;
    uint32_t *bit_reverse;
    uint32_t *digit_reverse;
    fft_complex *root;
    __m256 *finish_factor;
    __m256 *work;
} lane4_plan;

static volatile float cycle_sink;

static size_t round_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static unsigned integer_log2(size_t n)
{
    unsigned result = 0;
    while (n > 1) {
        n >>= 1;
        ++result;
    }
    return result;
}

static uint32_t reverse_bits(uint32_t value, unsigned count)
{
    uint32_t result = 0;
    while (count-- != 0) {
        result = (result << 1) | (value & 1U);
        value >>= 1;
    }
    return result;
}

static uint32_t reverse_mixed_radix(uint32_t value, unsigned levels)
{
    unsigned radices[16];
    unsigned digits[16] = {0};
    unsigned count = 0;
    uint32_t result = 0;
    uint32_t multiplier = 1;
    int i;

    if ((levels & 1U) != 0) {
        radices[count++] = 2;
    }
    while (count < (levels + 1U) / 2U) {
        radices[count++] = 4;
    }
    for (i = 0; i < (int)count; ++i) {
        digits[i] = value % radices[i];
        value /= radices[i];
    }
    for (i = (int)count - 1; i >= 0; --i) {
        result += digits[i] * multiplier;
        multiplier *= radices[i];
    }
    return result;
}

static __m256 pack_four_factors(size_t q, size_t n)
{
    float values[8];
    unsigned r;
    for (r = 0; r < 4; ++r) {
        const float angle =
            -2.0f * LANE4_PI * (float)(r * q) / (float)n;
        values[2 * r] = cosf(angle);
        values[2 * r + 1] = sinf(angle);
    }
    return _mm256_loadu_ps(values);
}

static lane4_plan *lane4_plan_create(size_t n)
{
    lane4_plan *plan;
    size_t i;

    if (n < 16 || (n & (n - 1)) != 0) {
        return NULL;
    }

    plan = (lane4_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->m = n / 4;
    plan->levels = integer_log2(plan->m);
    plan->bit_reverse =
        (uint32_t *)malloc(plan->m * sizeof(*plan->bit_reverse));
    plan->digit_reverse =
        (uint32_t *)malloc(plan->m * sizeof(*plan->digit_reverse));
    plan->root = (fft_complex *)aligned_alloc(
        32, round_up(plan->m * sizeof(*plan->root), 32));
    plan->finish_factor = (__m256 *)aligned_alloc(
        32, round_up(plan->m * sizeof(*plan->finish_factor), 32));
    plan->work = (__m256 *)aligned_alloc(
        32, round_up(plan->m * sizeof(*plan->work), 32));
    if (plan->bit_reverse == NULL || plan->digit_reverse == NULL ||
        plan->root == NULL ||
        plan->finish_factor == NULL || plan->work == NULL) {
        free(plan->bit_reverse);
        free(plan->digit_reverse);
        free(plan->root);
        free(plan->finish_factor);
        free(plan->work);
        free(plan);
        return NULL;
    }

    for (i = 0; i < plan->m; ++i) {
        const float angle =
            -2.0f * LANE4_PI * (float)i / (float)plan->m;
        plan->bit_reverse[i] =
            reverse_bits((uint32_t)i, plan->levels);
        plan->digit_reverse[i] =
            reverse_mixed_radix((uint32_t)i, plan->levels);
        plan->root[i].re = cosf(angle);
        plan->root[i].im = sinf(angle);
        plan->finish_factor[i] = pack_four_factors(i, n);
    }
    return plan;
}

static void lane4_plan_destroy(lane4_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->bit_reverse);
    free(plan->digit_reverse);
    free(plan->root);
    free(plan->finish_factor);
    free(plan->work);
    free(plan);
}

static inline __m256 multiply_common(__m256 value, float re, float im)
{
    const __m256 real = _mm256_set1_ps(re);
    const __m256 imag = _mm256_set1_ps(im);
    const __m256 swapped = _mm256_permute_ps(value, 0xb1);
    return _mm256_fmaddsub_ps(value, real,
                              _mm256_mul_ps(swapped, imag));
}

static inline __m256 multiply_lanes(__m256 value, __m256 factor)
{
    const __m256 real = _mm256_moveldup_ps(factor);
    const __m256 imag = _mm256_movehdup_ps(factor);
    const __m256 swapped = _mm256_permute_ps(value, 0xb1);
    return _mm256_fmaddsub_ps(value, real,
                              _mm256_mul_ps(swapped, imag));
}

static inline __m256 multiply_minus_i(__m256 value)
{
    const __m256 sign_odd = _mm256_castsi256_ps(_mm256_set_epi32(
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0));
    return _mm256_xor_ps(_mm256_permute_ps(value, 0xb1), sign_odd);
}

static inline void radix4_vectors(__m256 a,
                                  __m256 b,
                                  __m256 c,
                                  __m256 d,
                                  __m256 *output0,
                                  __m256 *output1,
                                  __m256 *output2,
                                  __m256 *output3)
{
    const __m256 ac_sum = _mm256_add_ps(a, c);
    const __m256 ac_difference = _mm256_sub_ps(a, c);
    const __m256 bd_sum = _mm256_add_ps(b, d);
    const __m256 rotated = multiply_minus_i(_mm256_sub_ps(b, d));
    *output0 = _mm256_add_ps(ac_sum, bd_sum);
    *output1 = _mm256_add_ps(ac_difference, rotated);
    *output2 = _mm256_sub_ps(ac_sum, bd_sum);
    *output3 = _mm256_sub_ps(ac_difference, rotated);
}

/*
 * Four complex values occupy the four 64-bit lanes. This computes their
 * natural-order forward FFT4 and returns [y0,y1,y2,y3].
 */
static inline __m256 fft4_across_complex_lanes(__m256 value)
{
    const __m256 sign_odd = _mm256_castsi256_ps(_mm256_set_epi32(
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0,
        (int)UINT32_C(0x80000000), 0));
    const __m256 half_swapped =
        _mm256_permute2f128_ps(value, value, 0x01);
    const __m256 ac = _mm256_add_ps(value, half_swapped);
    const __m256 bd = _mm256_sub_ps(value, half_swapped);
    const __m256 ac_adjacent = _mm256_castpd_ps(
        _mm256_permute4x64_pd(_mm256_castps_pd(ac), 0xb1));
    const __m256 minus_i_bd = _mm256_xor_ps(
        _mm256_permute_ps(bd, 0xb1), sign_odd);
    const __m256 minus_i_bd_adjacent = _mm256_castpd_ps(
        _mm256_permute4x64_pd(
            _mm256_castps_pd(minus_i_bd), 0xb1));
    const __m256 output0 = _mm256_add_ps(ac, ac_adjacent);
    const __m256 output2 = _mm256_sub_ps(ac, ac_adjacent);
    const __m256 output1 = _mm256_add_ps(bd, minus_i_bd_adjacent);
    const __m256 output3 = _mm256_sub_ps(bd, minus_i_bd_adjacent);
    const __m256d low = _mm256_unpacklo_pd(
        _mm256_castps_pd(output0), _mm256_castps_pd(output1));
    const __m256d high = _mm256_unpacklo_pd(
        _mm256_castps_pd(output2), _mm256_castps_pd(output3));
    return _mm256_castpd_ps(_mm256_permute2f128_pd(low, high, 0x20));
}

static inline void transpose_store4(fft_complex *output,
                                    size_t m,
                                    size_t q,
                                    __m256 row0,
                                    __m256 row1,
                                    __m256 row2,
                                    __m256 row3)
{
    const __m256d a = _mm256_unpacklo_pd(
        _mm256_castps_pd(row0), _mm256_castps_pd(row1));
    const __m256d b = _mm256_unpackhi_pd(
        _mm256_castps_pd(row0), _mm256_castps_pd(row1));
    const __m256d c = _mm256_unpacklo_pd(
        _mm256_castps_pd(row2), _mm256_castps_pd(row3));
    const __m256d d = _mm256_unpackhi_pd(
        _mm256_castps_pd(row2), _mm256_castps_pd(row3));
    const __m256 output0 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(a, c, 0x20));
    const __m256 output2 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(a, c, 0x31));
    const __m256 output1 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x20));
    const __m256 output3 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x31));

    _mm256_storeu_ps((float *)(output + q), output0);
    _mm256_storeu_ps((float *)(output + m + q), output1);
    _mm256_storeu_ps((float *)(output + 2 * m + q), output2);
    _mm256_storeu_ps((float *)(output + 3 * m + q), output3);
}

__attribute__((noinline))
static void lane4_execute(const lane4_plan *plan, fft_complex *data)
{
    const size_t m = plan->m;
    size_t i;
    size_t length;

    /*
     * One YMM is four adjacent input samples, hence four residue-class FFTs
     * run in parallel without an AoS/SoA conversion.
     */
    for (i = 0; i < m; ++i) {
        plan->work[plan->bit_reverse[i]] =
            _mm256_loadu_ps((const float *)(data + 4 * i));
    }

    for (length = 2; length <= m; length <<= 1) {
        const size_t half = length / 2;
        const size_t root_stride = m / length;
        size_t k;

        /* k outermost broadcasts each root once for all independent blocks. */
        for (k = 0; k < half; ++k) {
            const fft_complex factor = plan->root[k * root_stride];
            size_t start;
            for (start = 0; start < m; start += length) {
                const size_t even_index = start + k;
                const size_t odd_index = even_index + half;
                const __m256 even = plan->work[even_index];
                __m256 odd = plan->work[odd_index];
                if (k != 0) {
                    odd = multiply_common(odd, factor.re, factor.im);
                }
                plan->work[even_index] = _mm256_add_ps(even, odd);
                plan->work[odd_index] = _mm256_sub_ps(even, odd);
            }
        }
    }

    for (i = 0; i < m; i += 4) {
        __m256 row0 = plan->work[i];
        __m256 row1 = plan->work[i + 1];
        __m256 row2 = plan->work[i + 2];
        __m256 row3 = plan->work[i + 3];
        if (i != 0) {
            row0 = multiply_lanes(row0, plan->finish_factor[i]);
        }
        row1 = multiply_lanes(row1, plan->finish_factor[i + 1]);
        row2 = multiply_lanes(row2, plan->finish_factor[i + 2]);
        row3 = multiply_lanes(row3, plan->finish_factor[i + 3]);
        row0 = fft4_across_complex_lanes(row0);
        row1 = fft4_across_complex_lanes(row1);
        row2 = fft4_across_complex_lanes(row2);
        row3 = fft4_across_complex_lanes(row3);
        transpose_store4(data, m, i, row0, row1, row2, row3);
    }
    _mm256_zeroupper();
}

__attribute__((noinline))
static void lane4_radix4_execute(const lane4_plan *plan, fft_complex *data)
{
    const size_t m = plan->m;
    size_t i;
    size_t length;

    if ((plan->levels & 1U) != 0) {
        /*
         * Fuse the mixed-radix input permutation, FFT2s, and first radix-4
         * stage into an FFT8 codelet. Eight input vectors and eight output
         * vectors fit in the AVX2 register file, eliminating a work-array
         * round trip for every odd-log2 inner transform.
         */
        const float diagonal = 0.7071067811865475244f;
        for (i = 0; i < m; i += 8) {
            const __m256 x0 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i]));
            const __m256 x1 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 1]));
            const __m256 x2 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 2]));
            const __m256 x3 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 3]));
            const __m256 x4 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 4]));
            const __m256 x5 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 5]));
            const __m256 x6 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 6]));
            const __m256 x7 = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 7]));
            const __m256 p0 = _mm256_add_ps(x0, x1);
            const __m256 p1 = _mm256_sub_ps(x0, x1);
            const __m256 p2 = _mm256_add_ps(x2, x3);
            __m256 p3 = _mm256_sub_ps(x2, x3);
            const __m256 p4 = _mm256_add_ps(x4, x5);
            __m256 p5 = _mm256_sub_ps(x4, x5);
            const __m256 p6 = _mm256_add_ps(x6, x7);
            __m256 p7 = _mm256_sub_ps(x6, x7);
            __m256 output0;
            __m256 output1;
            __m256 output2;
            __m256 output3;

            radix4_vectors(p0, p2, p4, p6,
                           &output0, &output1, &output2, &output3);
            plan->work[i] = output0;
            plan->work[i + 2] = output1;
            plan->work[i + 4] = output2;
            plan->work[i + 6] = output3;

            p3 = multiply_common(p3, diagonal, -diagonal);
            p5 = multiply_minus_i(p5);
            p7 = multiply_common(p7, -diagonal, -diagonal);
            radix4_vectors(p1, p3, p5, p7,
                           &output0, &output1, &output2, &output3);
            plan->work[i + 1] = output0;
            plan->work[i + 3] = output1;
            plan->work[i + 5] = output2;
            plan->work[i + 7] = output3;
        }
        length = 8;
    } else {
        /* The even-level case similarly fuses permutation with a base FFT4. */
        for (i = 0; i < m; i += 4) {
            const __m256 a = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i]));
            const __m256 b = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 1]));
            const __m256 c = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 2]));
            const __m256 d = _mm256_loadu_ps((const float *)(
                data + 4 * plan->digit_reverse[i + 3]));
            radix4_vectors(a, b, c, d,
                           &plan->work[i],
                           &plan->work[i + 1],
                           &plan->work[i + 2],
                           &plan->work[i + 3]);
        }
        length = 4;
    }

    while (length < m) {
        const size_t previous = length;
        const size_t root_stride = m / (4 * previous);
        size_t start;
        length *= 4;

        for (start = 0; start < m; start += length) {
            size_t k;
            for (k = 0; k < previous; ++k) {
                __m256 a = plan->work[start + k];
                __m256 b = plan->work[start + previous + k];
                __m256 c = plan->work[start + 2 * previous + k];
                __m256 d = plan->work[start + 3 * previous + k];
                if (k != 0) {
                    const size_t index = k * root_stride;
                    const fft_complex w1 = plan->root[index];
                    const fft_complex w2 = plan->root[2 * index];
                    const fft_complex w3 = plan->root[3 * index];
                    b = multiply_common(b, w1.re, w1.im);
                    c = multiply_common(c, w2.re, w2.im);
                    d = multiply_common(d, w3.re, w3.im);
                }

                radix4_vectors(
                    a, b, c, d,
                    &plan->work[start + k],
                    &plan->work[start + previous + k],
                    &plan->work[start + 2 * previous + k],
                    &plan->work[start + 3 * previous + k]);
            }
        }
    }

    for (i = 0; i < m; i += 4) {
        __m256 row0 = plan->work[i];
        __m256 row1 = plan->work[i + 1];
        __m256 row2 = plan->work[i + 2];
        __m256 row3 = plan->work[i + 3];
        if (i != 0) {
            row0 = multiply_lanes(row0, plan->finish_factor[i]);
        }
        row1 = multiply_lanes(row1, plan->finish_factor[i + 1]);
        row2 = multiply_lanes(row2, plan->finish_factor[i + 2]);
        row3 = multiply_lanes(row3, plan->finish_factor[i + 3]);
        row0 = fft4_across_complex_lanes(row0);
        row1 = fft4_across_complex_lanes(row1);
        row2 = fft4_across_complex_lanes(row2);
        row3 = fft4_across_complex_lanes(row3);
        transpose_store4(data, m, i, row0, row1, row2, row3);
    }
    _mm256_zeroupper();
}

static int compare_u64(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t read_tsc_start(void)
{
    _mm_lfence();
    return __rdtsc();
}

static uint64_t read_tsc_end(void)
{
    unsigned auxiliary;
    const uint64_t value = __rdtscp(&auxiliary);
    _mm_lfence();
    return value;
}

static double relative_max_error(const fft_complex *actual,
                                 const fft_complex *expected,
                                 size_t n)
{
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    size_t i;
    for (i = 0; i < n; ++i) {
        const double error = hypot(
            (double)actual[i].re - expected[i].re,
            (double)actual[i].im - expected[i].im);
        const double reference =
            hypot((double)expected[i].re, (double)expected[i].im);
        maximum_error = fmax(maximum_error, error);
        maximum_reference = fmax(maximum_reference, reference);
    }
    return maximum_error / fmax(1.0, maximum_reference);
}

static double benchmark_lane4(const lane4_plan *plan, fft_complex *data)
{
    size_t iterations = 1048576 / plan->n;
    size_t warm_iterations = 67108864 / plan->n;
    uint64_t samples[SAMPLE_COUNT];
    size_t sample;

    if (iterations < 128) {
        iterations = 128;
    }
    if (warm_iterations < 8192) {
        warm_iterations = 8192;
    }
    memset(data, 0, plan->n * sizeof(*data));
    for (sample = 0; sample < warm_iterations; ++sample) {
        lane4_execute(plan, data);
    }
    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        const uint64_t start = read_tsc_start();
        for (iteration = 0; iteration < iterations; ++iteration) {
            lane4_execute(plan, data);
        }
        samples[sample] = read_tsc_end() - start;
    }
    qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
    cycle_sink += data[(plan->n * 7) / 11].re;
    return (double)samples[SAMPLE_COUNT / 2] / (double)iterations;
}

static double benchmark_lane4_radix4(const lane4_plan *plan,
                                     fft_complex *data)
{
    size_t iterations = 1048576 / plan->n;
    size_t warm_iterations = 67108864 / plan->n;
    uint64_t samples[SAMPLE_COUNT];
    size_t sample;

    if (iterations < 128) {
        iterations = 128;
    }
    if (warm_iterations < 8192) {
        warm_iterations = 8192;
    }
    memset(data, 0, plan->n * sizeof(*data));
    for (sample = 0; sample < warm_iterations; ++sample) {
        lane4_radix4_execute(plan, data);
    }
    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        const uint64_t start = read_tsc_start();
        for (iteration = 0; iteration < iterations; ++iteration) {
            lane4_radix4_execute(plan, data);
        }
        samples[sample] = read_tsc_end() - start;
    }
    qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
    cycle_sink += data[(plan->n * 7) / 11].re;
    return (double)samples[SAMPLE_COUNT / 2] / (double)iterations;
}

static double benchmark_existing(fft_plan *plan,
                                 fft_algorithm algorithm,
                                 fft_complex *data)
{
    const size_t n = fft_plan_size(plan);
    size_t iterations = 1048576 / n;
    size_t warm_iterations = 67108864 / n;
    uint64_t samples[SAMPLE_COUNT];
    size_t sample;

    if (iterations < 128) {
        iterations = 128;
    }
    if (warm_iterations < 8192) {
        warm_iterations = 8192;
    }
    memset(data, 0, n * sizeof(*data));
    for (sample = 0; sample < warm_iterations; ++sample) {
        (void)fft_execute(plan, algorithm, data);
    }
    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        const uint64_t start = read_tsc_start();
        for (iteration = 0; iteration < iterations; ++iteration) {
            (void)fft_execute(plan, algorithm, data);
        }
        samples[sample] = read_tsc_end() - start;
    }
    qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
    cycle_sink += data[(n * 7) / 11].re;
    return (double)samples[SAMPLE_COUNT / 2] / (double)iterations;
}

int main(void)
{
    unsigned power;

    puts("Lane-factorized FFT experiment (cycles/execute, median)");
    printf("%8s %14s %14s %14s %14s %14s %12s\n",
           "N", "lane4-radix2", "prototype-r4", "lane4-prod",
           "tangent-asm", "ffmpeg-avtx", "rel.error");

    for (power = 4; power <= 13; ++power) {
        const size_t n = (size_t)1 << power;
        lane4_plan *lane_plan = lane4_plan_create(n);
        fft_plan *reference_plan = fft_plan_create(n);
        fft_complex *input =
            (fft_complex *)aligned_alloc(32, n * sizeof(*input));
        fft_complex *reference =
            (fft_complex *)aligned_alloc(32, n * sizeof(*reference));
        fft_complex *work =
            (fft_complex *)aligned_alloc(32, n * sizeof(*work));
        double error;
        double lane_cycles;
        double lane_radix4_cycles;
        double production_cycles;
        double tangent_cycles;
        double ffmpeg_cycles;
        size_t i;

        if (lane_plan == NULL || reference_plan == NULL || input == NULL ||
            reference == NULL || work == NULL) {
            fprintf(stderr, "allocation failed for N=%zu\n", n);
            lane4_plan_destroy(lane_plan);
            fft_plan_destroy(reference_plan);
            free(input);
            free(reference);
            free(work);
            return EXIT_FAILURE;
        }
        for (i = 0; i < n; ++i) {
            input[i].re = sinf(0.071f * (float)i) +
                          cosf(0.013f * (float)i);
            input[i].im = cosf(0.037f * (float)i) -
                          sinf(0.019f * (float)i);
        }
        memcpy(reference, input, n * sizeof(*input));
        memcpy(work, input, n * sizeof(*input));
        (void)fft_execute(reference_plan, FFT_RADIX2, reference);
        lane4_execute(lane_plan, work);
        error = relative_max_error(work, reference, n);
        memcpy(work, input, n * sizeof(*input));
        lane4_radix4_execute(lane_plan, work);
        error = fmax(error, relative_max_error(work, reference, n));

        lane_cycles = benchmark_lane4(lane_plan, work);
        lane_radix4_cycles = benchmark_lane4_radix4(lane_plan, work);
        production_cycles = benchmark_existing(
            reference_plan, FFT_LANE4_RADIX4, work);
        tangent_cycles = benchmark_existing(
            reference_plan, FFT_TANGENT_X86_ASM, work);
        ffmpeg_cycles = benchmark_existing(
            reference_plan, FFT_FFMPEG, work);
        printf("%8zu %14.3f %14.3f %14.3f %14.3f %14.3f %12.3e\n",
               n, lane_cycles, lane_radix4_cycles, production_cycles,
               tangent_cycles, ffmpeg_cycles, error);

        lane4_plan_destroy(lane_plan);
        fft_plan_destroy(reference_plan);
        free(input);
        free(reference);
        free(work);
    }
    return cycle_sink == 12345.0f ? EXIT_FAILURE : EXIT_SUCCESS;
}
