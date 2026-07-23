#include "lane4_fft.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define LANE4_PI 3.141592653589793238462643383279502884f

struct lane4_fft_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    fft_complex *root;
    __m256 *finish_factor;
    __m256 *work;
};

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

/*
 * Reverse the digits for a DIT factorization whose first codelet is a fused
 * radix-2/radix-4 FFT8 for odd log2 sizes or FFT4 for even log2 sizes,
 * followed by radix-4 stages.
 */
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

static __m256 pack_finish_factor(size_t frequency, size_t n)
{
    float values[8];
    unsigned lane;
    for (lane = 0; lane < 4; ++lane) {
        const float angle =
            -2.0f * LANE4_PI * (float)(lane * frequency) / (float)n;
        values[2 * lane] = cosf(angle);
        values[2 * lane + 1] = sinf(angle);
    }
    return _mm256_loadu_ps(values);
}

lane4_fft_plan *lane4_fft_plan_create(size_t n)
{
    lane4_fft_plan *plan;
    size_t i;

    if (n < 16 || (n & (n - 1)) != 0) {
        return NULL;
    }
    plan = (lane4_fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->inner_size = n / 4;
    plan->inner_levels = integer_log2(plan->inner_size);
    plan->permutation =
        (uint32_t *)malloc(plan->inner_size * sizeof(*plan->permutation));
    plan->root = (fft_complex *)aligned_alloc(
        32, round_up(plan->inner_size * sizeof(*plan->root), 32));
    plan->finish_factor = (__m256 *)aligned_alloc(
        32, round_up(plan->inner_size * sizeof(*plan->finish_factor), 32));
    plan->work = (__m256 *)aligned_alloc(
        32, round_up(plan->inner_size * sizeof(*plan->work), 32));
    if (plan->permutation == NULL || plan->root == NULL ||
        plan->finish_factor == NULL || plan->work == NULL) {
        lane4_fft_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        const float angle =
            -2.0f * LANE4_PI * (float)i / (float)plan->inner_size;
        plan->permutation[i] =
            reverse_mixed_radix((uint32_t)i, plan->inner_levels);
        plan->root[i].re = cosf(angle);
        plan->root[i].im = sinf(angle);
        plan->finish_factor[i] = pack_finish_factor(i, n);
    }
    return plan;
}

void lane4_fft_plan_destroy(lane4_fft_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->permutation);
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
    return _mm256_fmaddsub_ps(
        value, real, _mm256_mul_ps(swapped, imag));
}

static inline __m256 multiply_lanes(__m256 value, __m256 factor)
{
    const __m256 real = _mm256_moveldup_ps(factor);
    const __m256 imag = _mm256_movehdup_ps(factor);
    const __m256 swapped = _mm256_permute_ps(value, 0xb1);
    return _mm256_fmaddsub_ps(
        value, real, _mm256_mul_ps(swapped, imag));
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
 * Transpose four q rows into four r columns, then perform one vector FFT4
 * down those columns. This is the same operation as four horizontal FFT4s
 * followed by a transpose, but it replaces four shuffle-heavy lane FFTs with
 * one ordinary full-width butterfly.
 */
static inline void finish_store4(fft_complex *output,
                                 size_t inner_size,
                                 size_t frequency,
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
    const __m256 column0 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(a, c, 0x20));
    const __m256 column2 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(a, c, 0x31));
    const __m256 column1 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x20));
    const __m256 column3 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x31));
    __m256 output0;
    __m256 output1;
    __m256 output2;
    __m256 output3;

    radix4_vectors(column0, column1, column2, column3,
                   &output0, &output1, &output2, &output3);

    _mm256_storeu_ps((float *)(output + frequency), output0);
    _mm256_storeu_ps(
        (float *)(output + inner_size + frequency), output1);
    _mm256_storeu_ps(
        (float *)(output + 2 * inner_size + frequency), output2);
    _mm256_storeu_ps(
        (float *)(output + 3 * inner_size + frequency), output3);
}

static void fused_base_fft8(lane4_fft_plan *plan, const fft_complex *data)
{
    const float diagonal = 0.7071067811865475244f;
    size_t i;

    for (i = 0; i < plan->inner_size; i += 8) {
        const __m256 x0 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i]));
        const __m256 x1 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 1]));
        const __m256 x2 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 2]));
        const __m256 x3 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 3]));
        const __m256 x4 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 4]));
        const __m256 x5 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 5]));
        const __m256 x6 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 6]));
        const __m256 x7 = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 7]));
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
}

static void fused_base_fft4(lane4_fft_plan *plan, const fft_complex *data)
{
    size_t i;
    for (i = 0; i < plan->inner_size; i += 4) {
        const __m256 a = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i]));
        const __m256 b = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 1]));
        const __m256 c = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 2]));
        const __m256 d = _mm256_loadu_ps((const float *)(
            data + 4 * plan->permutation[i + 3]));
        radix4_vectors(a, b, c, d,
                       &plan->work[i],
                       &plan->work[i + 1],
                       &plan->work[i + 2],
                       &plan->work[i + 3]);
    }
}

/*
 * At N=16 the complete lane factorization is one vector FFT4 followed by
 * four cross-lane FFT4s. Keeping those rows in registers avoids a work-array
 * store/load round trip that dominates such a small transform.
 */
static void lane4_fft16(lane4_fft_plan *plan, fft_complex *data)
{
    const __m256 a = _mm256_loadu_ps((const float *)(data + 0));
    const __m256 b = _mm256_loadu_ps((const float *)(data + 4));
    const __m256 c = _mm256_loadu_ps((const float *)(data + 8));
    const __m256 d = _mm256_loadu_ps((const float *)(data + 12));
    __m256 row0;
    __m256 row1;
    __m256 row2;
    __m256 row3;

    radix4_vectors(a, b, c, d, &row0, &row1, &row2, &row3);
    row1 = multiply_lanes(row1, plan->finish_factor[1]);
    row2 = multiply_lanes(row2, plan->finish_factor[2]);
    row3 = multiply_lanes(row3, plan->finish_factor[3]);
    finish_store4(data, 4, 0, row0, row1, row2, row3);
}

/*
 * N=32 is a fused radix-2/radix-4 vector FFT8. Its eight output rows fit in
 * the sixteen-register AVX2 file, so feed them directly into the cross-lane
 * codelets rather than materializing the intermediate vector transform.
 */
static void lane4_fft32(lane4_fft_plan *plan, fft_complex *data)
{
    const float diagonal = 0.7071067811865475244f;
    const __m256 x0 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[0]));
    const __m256 x1 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[1]));
    const __m256 x2 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[2]));
    const __m256 x3 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[3]));
    const __m256 x4 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[4]));
    const __m256 x5 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[5]));
    const __m256 x6 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[6]));
    const __m256 x7 = _mm256_loadu_ps((const float *)(
        data + 4 * plan->permutation[7]));
    const __m256 p0 = _mm256_add_ps(x0, x1);
    const __m256 p1 = _mm256_sub_ps(x0, x1);
    const __m256 p2 = _mm256_add_ps(x2, x3);
    __m256 p3 = _mm256_sub_ps(x2, x3);
    const __m256 p4 = _mm256_add_ps(x4, x5);
    __m256 p5 = _mm256_sub_ps(x4, x5);
    const __m256 p6 = _mm256_add_ps(x6, x7);
    __m256 p7 = _mm256_sub_ps(x6, x7);
    __m256 even0;
    __m256 even1;
    __m256 even2;
    __m256 even3;
    __m256 odd0;
    __m256 odd1;
    __m256 odd2;
    __m256 odd3;

    radix4_vectors(p0, p2, p4, p6,
                   &even0, &even1, &even2, &even3);
    p3 = multiply_common(p3, diagonal, -diagonal);
    p5 = multiply_minus_i(p5);
    p7 = multiply_common(p7, -diagonal, -diagonal);
    radix4_vectors(p1, p3, p5, p7, &odd0, &odd1, &odd2, &odd3);

    odd0 = multiply_lanes(odd0, plan->finish_factor[1]);
    even1 = multiply_lanes(even1, plan->finish_factor[2]);
    odd1 = multiply_lanes(odd1, plan->finish_factor[3]);
    finish_store4(data, 8, 0, even0, odd0, even1, odd1);

    even2 = multiply_lanes(even2, plan->finish_factor[4]);
    odd2 = multiply_lanes(odd2, plan->finish_factor[5]);
    even3 = multiply_lanes(even3, plan->finish_factor[6]);
    odd3 = multiply_lanes(odd3, plan->finish_factor[7]);
    finish_store4(data, 8, 4, even2, odd2, even3, odd3);
}

int lane4_fft_execute(lane4_fft_plan *plan, fft_complex *data)
{
    size_t length;
    size_t frequency;

    if (plan == NULL || data == NULL) {
        return -1;
    }
    if (plan->n == 16) {
        lane4_fft16(plan, data);
        _mm256_zeroupper();
        return 0;
    }
    if (plan->n == 32) {
        lane4_fft32(plan, data);
        _mm256_zeroupper();
        return 0;
    }
    if ((plan->inner_levels & 1U) != 0) {
        fused_base_fft8(plan, data);
        length = 8;
    } else {
        fused_base_fft4(plan, data);
        length = 4;
    }

    while (length < plan->inner_size) {
        const size_t previous = length;
        const size_t root_stride =
            plan->inner_size / (4 * previous);
        size_t start;
        length *= 4;

        for (start = 0; start < plan->inner_size; start += length) {
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

    for (frequency = 0;
         frequency < plan->inner_size;
         frequency += 4) {
        __m256 row0 = plan->work[frequency];
        __m256 row1 = plan->work[frequency + 1];
        __m256 row2 = plan->work[frequency + 2];
        __m256 row3 = plan->work[frequency + 3];
        if (frequency != 0) {
            row0 = multiply_lanes(
                row0, plan->finish_factor[frequency]);
        }
        row1 = multiply_lanes(
            row1, plan->finish_factor[frequency + 1]);
        row2 = multiply_lanes(
            row2, plan->finish_factor[frequency + 2]);
        row3 = multiply_lanes(
            row3, plan->finish_factor[frequency + 3]);
        finish_store4(data, plan->inner_size, frequency,
                      row0, row1, row2, row3);
    }
    _mm256_zeroupper();
    return 0;
}
