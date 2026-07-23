#if defined(LANE4_BUILD_AVX)
#define lane4_fft_plan lane4_avx_fft_plan
#define lane4_fft_plan_create lane4_avx_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx_fft_plan_destroy
#define lane4_fft_execute lane4_avx_fft_execute
#define LANE4_USE_FMA 0
#elif defined(LANE4_BUILD_AVX_FMA)
#define lane4_fft_plan lane4_avx_fma_fft_plan
#define lane4_fft_plan_create lane4_avx_fma_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx_fma_fft_plan_destroy
#define lane4_fft_execute lane4_avx_fma_fft_execute
#define LANE4_USE_FMA 1
#elif defined(LANE4_BUILD_AVX2)
#define lane4_fft_plan lane4_avx2_fft_plan
#define lane4_fft_plan_create lane4_avx2_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx2_fft_plan_destroy
#define lane4_fft_execute lane4_avx2_fft_execute
#define LANE4_USE_FMA 0
#else
#define LANE4_USE_FMA 1
#endif

#include "lane4_fft.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define LANE4_PI 3.141592653589793238462643383279502884f

typedef struct {
    fft_complex w1;
    fft_complex w2;
    fft_complex w3;
} lane4_twiddle_triplet;

struct lane4_fft_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    lane4_twiddle_triplet *twiddle;
    size_t twiddle_start[16];
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

static __m256 pack_finish_column(size_t first_frequency,
                                 unsigned lane,
                                 size_t n)
{
    float values[8];
    unsigned i;
    for (i = 0; i < 4; ++i) {
        const float angle =
            -2.0f * LANE4_PI *
            (float)(lane * (first_frequency + i)) / (float)n;
        values[2 * i] = cosf(angle);
        values[2 * i + 1] = sinf(angle);
    }
    return _mm256_loadu_ps(values);
}

lane4_fft_plan *lane4_fft_plan_create(size_t n)
{
    lane4_fft_plan *plan;
    size_t twiddle_count = 0;
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
        (uint32_t *)malloc(
            plan->inner_size * sizeof(*plan->permutation));
    {
        size_t length =
            (plan->inner_levels & 1U) != 0 ? 8 : 4;
        unsigned stage = 0;
        while (length < plan->inner_size) {
            plan->twiddle_start[stage++] = twiddle_count;
            twiddle_count += length - 1;
            length *= 4;
        }
    }
    if (twiddle_count != 0) {
        plan->twiddle = (lane4_twiddle_triplet *)aligned_alloc(
            32, round_up(twiddle_count * sizeof(*plan->twiddle), 32));
    }
    plan->finish_factor = (__m256 *)aligned_alloc(
        32, round_up(6 * (plan->inner_size / 4) *
                     sizeof(*plan->finish_factor), 32));
    plan->work = (__m256 *)aligned_alloc(
        32, round_up(plan->inner_size * sizeof(*plan->work), 32));
    if (plan->permutation == NULL ||
        (twiddle_count != 0 && plan->twiddle == NULL) ||
        plan->finish_factor == NULL || plan->work == NULL) {
        lane4_fft_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        plan->permutation[i] = 4 * reverse_mixed_radix(
            (uint32_t)i, plan->inner_levels);
    }
    {
        size_t length =
            (plan->inner_levels & 1U) != 0 ? 8 : 4;
        unsigned stage = 0;
        while (length < plan->inner_size) {
            const size_t root_stride =
                plan->inner_size / (4 * length);
            lane4_twiddle_triplet *table =
                plan->twiddle + plan->twiddle_start[stage++];
            size_t k;
            for (k = 1; k < length; ++k) {
                const float angle =
                    -2.0f * LANE4_PI *
                    (float)(k * root_stride) /
                    (float)plan->inner_size;
                table[k - 1].w1.re = cosf(angle);
                table[k - 1].w1.im = sinf(angle);
                table[k - 1].w2.re = cosf(2.0f * angle);
                table[k - 1].w2.im = sinf(2.0f * angle);
                table[k - 1].w3.re = cosf(3.0f * angle);
                table[k - 1].w3.im = sinf(3.0f * angle);
            }
            length *= 4;
        }
    }
    for (i = 0; i < plan->inner_size; i += 4) {
        unsigned lane;
        for (lane = 1; lane < 4; ++lane) {
            const __m256 factor = pack_finish_column(i, lane, n);
            plan->finish_factor[6 * (i / 4) + 2 * (lane - 1)] =
                _mm256_moveldup_ps(factor);
            plan->finish_factor[6 * (i / 4) + 2 * (lane - 1) + 1] =
                _mm256_movehdup_ps(factor);
        }
    }
    return plan;
}

void lane4_fft_plan_destroy(lane4_fft_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->permutation);
    free(plan->twiddle);
    free(plan->finish_factor);
    free(plan->work);
    free(plan);
}

static inline __m256 multiply_common(__m256 value, float re, float im)
{
    const __m256 real = _mm256_set1_ps(re);
    const __m256 imag = _mm256_set1_ps(im);
    const __m256 swapped = _mm256_permute_ps(value, 0xb1);
#if LANE4_USE_FMA
    return _mm256_fmaddsub_ps(
        value, real, _mm256_mul_ps(swapped, imag));
#else
    return _mm256_addsub_ps(
        _mm256_mul_ps(value, real),
        _mm256_mul_ps(swapped, imag));
#endif
}

static inline __m256 multiply_lanes_split(__m256 value,
                                          __m256 real,
                                          __m256 imag)
{
    const __m256 swapped = _mm256_permute_ps(value, 0xb1);
#if LANE4_USE_FMA
    return _mm256_fmaddsub_ps(
        value, real, _mm256_mul_ps(swapped, imag));
#else
    return _mm256_addsub_ps(
        _mm256_mul_ps(value, real),
        _mm256_mul_ps(swapped, imag));
#endif
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

static inline void stage_butterfly(
    __m256 *a_pointer,
    __m256 *b_pointer,
    __m256 *c_pointer,
    __m256 *d_pointer,
    const lane4_twiddle_triplet *factor)
{
    __m256 a = *a_pointer;
    __m256 b = multiply_common(
        *b_pointer, factor->w1.re, factor->w1.im);
    __m256 c = multiply_common(
        *c_pointer, factor->w2.re, factor->w2.im);
    __m256 d = multiply_common(
        *d_pointer, factor->w3.re, factor->w3.im);
    radix4_vectors(a, b, c, d,
                   a_pointer, b_pointer, c_pointer, d_pointer);
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
                                 const __m256 *factor,
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
    __m256 column2 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(a, c, 0x31));
    __m256 column1 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x20));
    __m256 column3 =
        _mm256_castpd_ps(_mm256_permute2f128_pd(b, d, 0x31));
    __m256 output0;
    __m256 output1;
    __m256 output2;
    __m256 output3;

    column1 = multiply_lanes_split(column1, factor[0], factor[1]);
    column2 = multiply_lanes_split(column2, factor[2], factor[3]);
    column3 = multiply_lanes_split(column3, factor[4], factor[5]);
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

static inline void load_fft8_leaf(
    const lane4_fft_plan *plan,
    const fft_complex *data,
    size_t index,
    __m256 *output0,
    __m256 *output1,
    __m256 *output2,
    __m256 *output3,
    __m256 *output4,
    __m256 *output5,
    __m256 *output6,
    __m256 *output7)
{
    const float diagonal = 0.7071067811865475244f;
    const __m256 x0 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index]));
    const __m256 x1 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 1]));
    const __m256 x2 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 2]));
    const __m256 x3 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 3]));
    const __m256 x4 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 4]));
    const __m256 x5 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 5]));
    const __m256 x6 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 6]));
    const __m256 x7 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 7]));
    const __m256 p0 = _mm256_add_ps(x0, x1);
    const __m256 p1 = _mm256_sub_ps(x0, x1);
    const __m256 p2 = _mm256_add_ps(x2, x3);
    __m256 p3 = _mm256_sub_ps(x2, x3);
    const __m256 p4 = _mm256_add_ps(x4, x5);
    __m256 p5 = _mm256_sub_ps(x4, x5);
    const __m256 p6 = _mm256_add_ps(x6, x7);
    __m256 p7 = _mm256_sub_ps(x6, x7);

    radix4_vectors(
        p0, p2, p4, p6, output0, output2, output4, output6);
    p3 = multiply_common(p3, diagonal, -diagonal);
    p5 = multiply_minus_i(p5);
    p7 = multiply_common(p7, -diagonal, -diagonal);
    radix4_vectors(
        p1, p3, p5, p7, output1, output3, output5, output7);
}

static inline void stage_butterfly_last(
    __m256 *a_pointer,
    __m256 *b_pointer,
    __m256 *c_pointer,
    __m256 *d_pointer,
    __m256 d,
    const lane4_twiddle_triplet *factor)
{
    __m256 a = *a_pointer;
    __m256 b = multiply_common(
        *b_pointer, factor->w1.re, factor->w1.im);
    __m256 c = multiply_common(
        *c_pointer, factor->w2.re, factor->w2.im);
    d = multiply_common(d, factor->w3.re, factor->w3.im);
    radix4_vectors(
        a, b, c, d, a_pointer, b_pointer, c_pointer, d_pointer);
}

static __attribute__((noinline)) void fused_base_fft32(
    lane4_fft_plan *plan,
    const fft_complex *data)
{
    const lane4_twiddle_triplet *twiddle =
        plan->twiddle + plan->twiddle_start[0];
    size_t start;

    for (start = 0; start < plan->inner_size; start += 32) {
        __m256 d0;
        __m256 d1;
        __m256 d2;
        __m256 d3;
        __m256 d4;
        __m256 d5;
        __m256 d6;
        __m256 d7;

        load_fft8_leaf(
            plan, data, start,
            &plan->work[start],
            &plan->work[start + 1],
            &plan->work[start + 2],
            &plan->work[start + 3],
            &plan->work[start + 4],
            &plan->work[start + 5],
            &plan->work[start + 6],
            &plan->work[start + 7]);
        load_fft8_leaf(
            plan, data, start + 8,
            &plan->work[start + 8],
            &plan->work[start + 9],
            &plan->work[start + 10],
            &plan->work[start + 11],
            &plan->work[start + 12],
            &plan->work[start + 13],
            &plan->work[start + 14],
            &plan->work[start + 15]);
        load_fft8_leaf(
            plan, data, start + 16,
            &plan->work[start + 16],
            &plan->work[start + 17],
            &plan->work[start + 18],
            &plan->work[start + 19],
            &plan->work[start + 20],
            &plan->work[start + 21],
            &plan->work[start + 22],
            &plan->work[start + 23]);
        load_fft8_leaf(
            plan, data, start + 24,
            &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

        radix4_vectors(
            plan->work[start],
            plan->work[start + 8],
            plan->work[start + 16],
            d0,
            &plan->work[start],
            &plan->work[start + 8],
            &plan->work[start + 16],
            &plan->work[start + 24]);
        stage_butterfly_last(
            &plan->work[start + 1],
            &plan->work[start + 9],
            &plan->work[start + 17],
            &plan->work[start + 25],
            d1, &twiddle[0]);
        stage_butterfly_last(
            &plan->work[start + 2],
            &plan->work[start + 10],
            &plan->work[start + 18],
            &plan->work[start + 26],
            d2, &twiddle[1]);
        stage_butterfly_last(
            &plan->work[start + 3],
            &plan->work[start + 11],
            &plan->work[start + 19],
            &plan->work[start + 27],
            d3, &twiddle[2]);
        stage_butterfly_last(
            &plan->work[start + 4],
            &plan->work[start + 12],
            &plan->work[start + 20],
            &plan->work[start + 28],
            d4, &twiddle[3]);
        stage_butterfly_last(
            &plan->work[start + 5],
            &plan->work[start + 13],
            &plan->work[start + 21],
            &plan->work[start + 29],
            d5, &twiddle[4]);
        stage_butterfly_last(
            &plan->work[start + 6],
            &plan->work[start + 14],
            &plan->work[start + 22],
            &plan->work[start + 30],
            d6, &twiddle[5]);
        stage_butterfly_last(
            &plan->work[start + 7],
            &plan->work[start + 15],
            &plan->work[start + 23],
            &plan->work[start + 31],
            d7, &twiddle[6]);
    }
}

static void fused_base_fft4(lane4_fft_plan *plan, const fft_complex *data)
{
    size_t i;
    for (i = 0; i < plan->inner_size; i += 4) {
        const __m256 a = _mm256_loadu_ps((const float *)(
            data + plan->permutation[i]));
        const __m256 b = _mm256_loadu_ps((const float *)(
            data + plan->permutation[i + 1]));
        const __m256 c = _mm256_loadu_ps((const float *)(
            data + plan->permutation[i + 2]));
        const __m256 d = _mm256_loadu_ps((const float *)(
            data + plan->permutation[i + 3]));
        radix4_vectors(a, b, c, d,
                       &plan->work[i],
                       &plan->work[i + 1],
                       &plan->work[i + 2],
                       &plan->work[i + 3]);
    }
}

static inline void load_fft4_leaf(const lane4_fft_plan *plan,
                                  const fft_complex *data,
                                  size_t index,
                                  __m256 *output0,
                                  __m256 *output1,
                                  __m256 *output2,
                                  __m256 *output3)
{
    const __m256 a = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index]));
    const __m256 b = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 1]));
    const __m256 c = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 2]));
    const __m256 d = _mm256_loadu_ps((const float *)(
        data + plan->permutation[index + 3]));
    radix4_vectors(
        a, b, c, d, output0, output1, output2, output3);
}

/*
 * Fuse four vector FFT4 leaves with their first radix-4 parent. The generated
 * code carries almost all rows in registers and uses only a few short-lived
 * stack spills instead of materializing and reloading all sixteen work rows.
 */
static void fused_base_fft16(lane4_fft_plan *plan,
                             const fft_complex *data)
{
    const lane4_twiddle_triplet *twiddle =
        plan->twiddle + plan->twiddle_start[0];
    size_t start;

    for (start = 0; start < plan->inner_size; start += 16) {
        __m256 a0;
        __m256 a1;
        __m256 a2;
        __m256 a3;
        __m256 b0;
        __m256 b1;
        __m256 b2;
        __m256 b3;
        __m256 c0;
        __m256 c1;
        __m256 c2;
        __m256 c3;
        __m256 d0;
        __m256 d1;
        __m256 d2;
        __m256 d3;

        load_fft4_leaf(
            plan, data, start,
            &a0, &a1, &a2, &a3);
        load_fft4_leaf(
            plan, data, start + 4,
            &b0, &b1, &b2, &b3);
        load_fft4_leaf(
            plan, data, start + 8,
            &c0, &c1, &c2, &c3);
        load_fft4_leaf(
            plan, data, start + 12,
            &d0, &d1, &d2, &d3);

        radix4_vectors(
            a0, b0, c0, d0,
            &plan->work[start],
            &plan->work[start + 4],
            &plan->work[start + 8],
            &plan->work[start + 12]);
        b1 = multiply_common(b1, twiddle[0].w1.re, twiddle[0].w1.im);
        c1 = multiply_common(c1, twiddle[0].w2.re, twiddle[0].w2.im);
        d1 = multiply_common(d1, twiddle[0].w3.re, twiddle[0].w3.im);
        radix4_vectors(
            a1, b1, c1, d1,
            &plan->work[start + 1],
            &plan->work[start + 5],
            &plan->work[start + 9],
            &plan->work[start + 13]);
        b2 = multiply_common(b2, twiddle[1].w1.re, twiddle[1].w1.im);
        c2 = multiply_common(c2, twiddle[1].w2.re, twiddle[1].w2.im);
        d2 = multiply_common(d2, twiddle[1].w3.re, twiddle[1].w3.im);
        radix4_vectors(
            a2, b2, c2, d2,
            &plan->work[start + 2],
            &plan->work[start + 6],
            &plan->work[start + 10],
            &plan->work[start + 14]);
        b3 = multiply_common(b3, twiddle[2].w1.re, twiddle[2].w1.im);
        c3 = multiply_common(c3, twiddle[2].w2.re, twiddle[2].w2.im);
        d3 = multiply_common(d3, twiddle[2].w3.re, twiddle[2].w3.im);
        radix4_vectors(
            a3, b3, c3, d3,
            &plan->work[start + 3],
            &plan->work[start + 7],
            &plan->work[start + 11],
            &plan->work[start + 15]);
    }
}

static __attribute__((noinline)) void lane4_fft64(
    lane4_fft_plan *plan,
    fft_complex *data)
{
    const lane4_twiddle_triplet *twiddle =
        plan->twiddle + plan->twiddle_start[0];
    __m256 a0;
    __m256 a1;
    __m256 a2;
    __m256 a3;
    __m256 b0;
    __m256 b1;
    __m256 b2;
    __m256 b3;
    __m256 c0;
    __m256 c1;
    __m256 c2;
    __m256 c3;
    __m256 d0;
    __m256 d1;
    __m256 d2;
    __m256 d3;
    __m256 q0;
    __m256 q1;
    __m256 q2;
    __m256 q3;
    __m256 q4;
    __m256 q5;
    __m256 q6;
    __m256 q7;

    load_fft4_leaf(plan, data, 0, &a0, &a1, &a2, &a3);
    load_fft4_leaf(plan, data, 4, &b0, &b1, &b2, &b3);
    load_fft4_leaf(plan, data, 8, &c0, &c1, &c2, &c3);
    load_fft4_leaf(plan, data, 12, &d0, &d1, &d2, &d3);

    radix4_vectors(
        a0, b0, c0, d0,
        &q0, &q4, &plan->work[8], &plan->work[12]);

    b1 = multiply_common(b1, twiddle[0].w1.re, twiddle[0].w1.im);
    c1 = multiply_common(c1, twiddle[0].w2.re, twiddle[0].w2.im);
    d1 = multiply_common(d1, twiddle[0].w3.re, twiddle[0].w3.im);
    radix4_vectors(
        a1, b1, c1, d1,
        &q1, &q5, &plan->work[9], &plan->work[13]);

    b2 = multiply_common(b2, twiddle[1].w1.re, twiddle[1].w1.im);
    c2 = multiply_common(c2, twiddle[1].w2.re, twiddle[1].w2.im);
    d2 = multiply_common(d2, twiddle[1].w3.re, twiddle[1].w3.im);
    radix4_vectors(
        a2, b2, c2, d2,
        &q2, &q6, &plan->work[10], &plan->work[14]);

    b3 = multiply_common(b3, twiddle[2].w1.re, twiddle[2].w1.im);
    c3 = multiply_common(c3, twiddle[2].w2.re, twiddle[2].w2.im);
    d3 = multiply_common(d3, twiddle[2].w3.re, twiddle[2].w3.im);
    radix4_vectors(
        a3, b3, c3, d3,
        &q3, &q7, &plan->work[11], &plan->work[15]);

    finish_store4(
        data, 16, 0, plan->finish_factor,
        q0, q1, q2, q3);
    finish_store4(
        data, 16, 4, plan->finish_factor + 6,
        q4, q5, q6, q7);
    finish_store4(
        data, 16, 8, plan->finish_factor + 12,
        plan->work[8], plan->work[9],
        plan->work[10], plan->work[11]);
    finish_store4(
        data, 16, 12, plan->finish_factor + 18,
        plan->work[12], plan->work[13],
        plan->work[14], plan->work[15]);
}

static inline void stage_butterfly_last_keep0(
    __m256 *a_pointer,
    __m256 *b_pointer,
    __m256 *c_pointer,
    __m256 *d_pointer,
    __m256 d,
    const lane4_twiddle_triplet *factor,
    __m256 *output0)
{
    __m256 a = *a_pointer;
    __m256 b = multiply_common(
        *b_pointer, factor->w1.re, factor->w1.im);
    __m256 c = multiply_common(
        *c_pointer, factor->w2.re, factor->w2.im);
    d = multiply_common(d, factor->w3.re, factor->w3.im);
    radix4_vectors(
        a, b, c, d, output0, b_pointer, c_pointer, d_pointer);
}

static __attribute__((noinline)) void lane4_fft128(
    lane4_fft_plan *plan,
    fft_complex *data)
{
    const lane4_twiddle_triplet *twiddle =
        plan->twiddle + plan->twiddle_start[0];
    __m256 d0;
    __m256 d1;
    __m256 d2;
    __m256 d3;
    __m256 d4;
    __m256 d5;
    __m256 d6;
    __m256 d7;
    __m256 q0;
    __m256 q1;
    __m256 q2;
    __m256 q3;
    __m256 q4;
    __m256 q5;
    __m256 q6;
    __m256 q7;
    size_t frequency;

    load_fft8_leaf(
        plan, data, 0,
        &plan->work[0], &plan->work[1],
        &plan->work[2], &plan->work[3],
        &plan->work[4], &plan->work[5],
        &plan->work[6], &plan->work[7]);
    load_fft8_leaf(
        plan, data, 8,
        &plan->work[8], &plan->work[9],
        &plan->work[10], &plan->work[11],
        &plan->work[12], &plan->work[13],
        &plan->work[14], &plan->work[15]);
    load_fft8_leaf(
        plan, data, 16,
        &plan->work[16], &plan->work[17],
        &plan->work[18], &plan->work[19],
        &plan->work[20], &plan->work[21],
        &plan->work[22], &plan->work[23]);
    load_fft8_leaf(
        plan, data, 24,
        &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

    radix4_vectors(
        plan->work[0], plan->work[8], plan->work[16], d0,
        &q0, &plan->work[8], &plan->work[16], &plan->work[24]);
    stage_butterfly_last_keep0(
        &plan->work[1], &plan->work[9],
        &plan->work[17], &plan->work[25],
        d1, &twiddle[0], &q1);
    stage_butterfly_last_keep0(
        &plan->work[2], &plan->work[10],
        &plan->work[18], &plan->work[26],
        d2, &twiddle[1], &q2);
    stage_butterfly_last_keep0(
        &plan->work[3], &plan->work[11],
        &plan->work[19], &plan->work[27],
        d3, &twiddle[2], &q3);
    stage_butterfly_last_keep0(
        &plan->work[4], &plan->work[12],
        &plan->work[20], &plan->work[28],
        d4, &twiddle[3], &q4);
    stage_butterfly_last_keep0(
        &plan->work[5], &plan->work[13],
        &plan->work[21], &plan->work[29],
        d5, &twiddle[4], &q5);
    stage_butterfly_last_keep0(
        &plan->work[6], &plan->work[14],
        &plan->work[22], &plan->work[30],
        d6, &twiddle[5], &q6);
    stage_butterfly_last_keep0(
        &plan->work[7], &plan->work[15],
        &plan->work[23], &plan->work[31],
        d7, &twiddle[6], &q7);

    finish_store4(
        data, 32, 0, plan->finish_factor,
        q0, q1, q2, q3);
    finish_store4(
        data, 32, 4, plan->finish_factor + 6,
        q4, q5, q6, q7);
    for (frequency = 8; frequency < 32; frequency += 4) {
        finish_store4(
            data, 32, frequency,
            plan->finish_factor + 6 * (frequency / 4),
            plan->work[frequency],
            plan->work[frequency + 1],
            plan->work[frequency + 2],
            plan->work[frequency + 3]);
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
    finish_store4(data, 4, 0, plan->finish_factor,
                  row0, row1, row2, row3);
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
        data + plan->permutation[0]));
    const __m256 x1 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[1]));
    const __m256 x2 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[2]));
    const __m256 x3 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[3]));
    const __m256 x4 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[4]));
    const __m256 x5 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[5]));
    const __m256 x6 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[6]));
    const __m256 x7 = _mm256_loadu_ps((const float *)(
        data + plan->permutation[7]));
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

    finish_store4(data, 8, 0, plan->finish_factor,
                  even0, odd0, even1, odd1);

    finish_store4(data, 8, 4, plan->finish_factor + 6,
                  even2, odd2, even3, odd3);
}

int __attribute__((aligned(64))) lane4_fft_execute(
    lane4_fft_plan *plan,
    fft_complex *data)
{
    size_t length;
    size_t frequency;
    unsigned stage = 0;

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
    if (plan->n == 64) {
        lane4_fft64(plan, data);
        _mm256_zeroupper();
        return 0;
    }
    if (plan->n == 128) {
        lane4_fft128(plan, data);
        _mm256_zeroupper();
        return 0;
    }
    if ((plan->inner_levels & 1U) != 0) {
        fused_base_fft32(plan, data);
        length = 32;
        stage = 1;
    } else if (plan->inner_size >= 16) {
        fused_base_fft16(plan, data);
        length = 16;
        stage = 1;
    } else {
        fused_base_fft4(plan, data);
        length = 4;
    }

    while (length < plan->inner_size) {
        const size_t previous = length;
        const lane4_twiddle_triplet *stage_twiddle =
            plan->twiddle + plan->twiddle_start[stage];
        size_t start;
        ++stage;
        length *= 4;

        for (start = 0; start < plan->inner_size; start += length) {
            __m256 *a_pointer = plan->work + start;
            __m256 *b_pointer = a_pointer + previous;
            __m256 *c_pointer = b_pointer + previous;
            __m256 *d_pointer = c_pointer + previous;
            size_t k;
            {
                __m256 a = a_pointer[0];
                __m256 b = b_pointer[0];
                __m256 c = c_pointer[0];
                __m256 d = d_pointer[0];
                radix4_vectors(
                    a, b, c, d,
                    &a_pointer[0],
                    &b_pointer[0],
                    &c_pointer[0],
                    &d_pointer[0]);
            }
            for (k = 1; k + 1 < previous; k += 2) {
                stage_butterfly(
                    &a_pointer[k], &b_pointer[k],
                    &c_pointer[k], &d_pointer[k],
                    &stage_twiddle[k - 1]);
                stage_butterfly(
                    &a_pointer[k + 1], &b_pointer[k + 1],
                    &c_pointer[k + 1], &d_pointer[k + 1],
                    &stage_twiddle[k]);
            }
            if (k < previous) {
                stage_butterfly(
                    &a_pointer[k], &b_pointer[k],
                    &c_pointer[k], &d_pointer[k],
                    &stage_twiddle[k - 1]);
            }
        }
    }

    for (frequency = 0;
         frequency < plan->inner_size;
         frequency += 4) {
        finish_store4(
            data, plan->inner_size, frequency,
            plan->finish_factor + 6 * (frequency / 4),
            plan->work[frequency],
            plan->work[frequency + 1],
            plan->work[frequency + 2],
            plan->work[frequency + 3]);
    }
    _mm256_zeroupper();
    return 0;
}
