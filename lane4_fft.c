#if defined(LANE4_BUILD_AVX)
#define lane4_fft_plan lane4_avx_fft_plan
#define lane4_fft_plan_create lane4_avx_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx_fft_plan_destroy
#elif defined(LANE4_BUILD_AVX_FMA)
#define lane4_fft_plan lane4_avx_fma_fft_plan
#define lane4_fft_plan_create lane4_avx_fma_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx_fma_fft_plan_destroy
#elif defined(LANE4_BUILD_AVX2)
#define lane4_fft_plan lane4_avx2_fft_plan
#define lane4_fft_plan_create lane4_avx2_fft_plan_create
#define lane4_fft_plan_destroy lane4_avx2_fft_plan_destroy
#endif

#include "lane4_fft.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define LANE4_PI 3.141592653589793238462643383279502884f

/*
 * Execution is entirely handwritten assembly.  These scalar types describe
 * its aligned memory ABI without exposing compiler vector types or intrinsics.
 */
typedef struct {
    float value[8];
} lane4_avx_row;

typedef struct {
    lane4_avx_row w1_re;
    lane4_avx_row w1_im;
    lane4_avx_row w2_re;
    lane4_avx_row w2_im;
    lane4_avx_row w3_re;
    lane4_avx_row w3_im;
} lane4_avx_twiddle;

struct lane4_fft_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    lane4_avx_twiddle *twiddle;
    lane4_avx_row *finish_factor;
    lane4_avx_row *work;
};

#if HAVE_TANGENT_X86_ASM
_Static_assert(offsetof(struct lane4_fft_plan, n) == 0,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, inner_size) == 8,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, inner_levels) == 16,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, permutation) == 24,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, twiddle) == 32,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, finish_factor) == 40,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(struct lane4_fft_plan, work) == 48,
               "update lane4 assembly plan offsets");
#endif

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
 * Reverse the digits for a DIT factorization whose first codelet is FFT8 for
 * odd log2 sizes or FFT4 for even log2 sizes, followed by radix-4 stages.
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

static void replicate(float value, lane4_avx_row *row)
{
    unsigned lane;

    for (lane = 0; lane < 8; ++lane) {
        row->value[lane] = value;
    }
}

static void make_twiddle(lane4_avx_twiddle *output,
                         size_t k,
                         size_t root_stride,
                         size_t inner_size)
{
    unsigned multiple;

    for (multiple = 1; multiple <= 3; ++multiple) {
        const float angle =
            -2.0f * LANE4_PI *
            (float)(multiple * k * root_stride) / (float)inner_size;
        lane4_avx_row *real_row =
            (lane4_avx_row *)output + 2 * (multiple - 1);
        lane4_avx_row *imag_row = real_row + 1;

        replicate(cosf(angle), real_row);
        replicate(sinf(angle), imag_row);
    }
}

static void make_finish_group(lane4_avx_row *output,
                              size_t first_frequency,
                              size_t n)
{
    unsigned lane;

    for (lane = 1; lane < 4; ++lane) {
        lane4_avx_row *real_row = output + 2 * (lane - 1);
        lane4_avx_row *imag_row = real_row + 1;
        unsigned frequency;

        for (frequency = 0; frequency < 4; ++frequency) {
            const float angle =
                -2.0f * LANE4_PI *
                (float)(lane * (first_frequency + frequency)) / (float)n;
            const float real = cosf(angle);
            const float imag = sinf(angle);

            real_row->value[2 * frequency] = real;
            real_row->value[2 * frequency + 1] = real;
            imag_row->value[2 * frequency] = imag;
            imag_row->value[2 * frequency + 1] = imag;
        }
    }
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
    plan->permutation = (uint32_t *)malloc(
        plan->inner_size * sizeof(*plan->permutation));

    {
        size_t length =
            (plan->inner_levels & 1U) != 0 ? 8 : 4;

        while (length < plan->inner_size) {
            twiddle_count += length - 1;
            length *= 4;
        }
    }
    if (twiddle_count != 0) {
        plan->twiddle = (lane4_avx_twiddle *)aligned_alloc(
            64, round_up(twiddle_count * sizeof(*plan->twiddle), 64));
    }
    plan->finish_factor = (lane4_avx_row *)aligned_alloc(
        64, round_up(6 * (plan->inner_size / 4) *
                     sizeof(*plan->finish_factor), 64));
    plan->work = (lane4_avx_row *)aligned_alloc(
        64, round_up(plan->inner_size * sizeof(*plan->work), 64));

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
        size_t output = 0;
        size_t length =
            (plan->inner_levels & 1U) != 0 ? 8 : 4;

        while (length < plan->inner_size) {
            const size_t root_stride =
                plan->inner_size / (4 * length);
            size_t k;

            for (k = 1; k < length; ++k) {
                make_twiddle(
                    &plan->twiddle[output++],
                    k, root_stride, plan->inner_size);
            }
            length *= 4;
        }
    }
    for (i = 0; i < plan->inner_size; i += 4) {
        make_finish_group(
            plan->finish_factor + 6 * (i / 4), i, n);
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
