#include "lane4_portable_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define LANE4_PI 3.141592653589793238462643383279502884f

#if HAVE_TANGENT_X86_ASM
_Static_assert(offsetof(lane4_portable_plan, inner_size) == 8,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, inner_levels) == 16,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, mixed_permutation) == 32,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, replicated_root) == 48,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, finish_re) == 56,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, finish_im) == 64,
               "update lane4 assembly plan offsets");
_Static_assert(offsetof(lane4_portable_plan, work) == 72,
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

static uint32_t reverse_bits(uint32_t value, unsigned bits)
{
    uint32_t result = 0;
    unsigned bit;

    for (bit = 0; bit < bits; ++bit) {
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

lane4_portable_plan *lane4_portable_plan_create(size_t n)
{
    lane4_portable_plan *plan;
    size_t i;

    if (n < 16 || (n & (n - 1)) != 0) {
        return NULL;
    }

    plan = (lane4_portable_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->inner_size = n / 4;
    plan->inner_levels = integer_log2(plan->inner_size);
    plan->permutation = (uint32_t *)malloc(
        plan->inner_size * sizeof(*plan->permutation));
    plan->mixed_permutation = (uint32_t *)malloc(
        plan->inner_size * sizeof(*plan->mixed_permutation));
    plan->root = (fft_complex *)aligned_alloc(
        64,
        round_up(plan->inner_size * sizeof(*plan->root), 64));
    plan->replicated_root =
        (lane4_replicated_root *)aligned_alloc(
            64, round_up(plan->inner_size *
                         sizeof(*plan->replicated_root), 64));
    plan->finish_re = (float *)aligned_alloc(
        64, round_up(3 * plan->inner_size * sizeof(float), 64));
    plan->finish_im = (float *)aligned_alloc(
        64, round_up(3 * plan->inner_size * sizeof(float), 64));
    plan->work = (lane4_portable_row *)aligned_alloc(
        64,
        round_up(plan->inner_size * sizeof(*plan->work), 64));

    if (plan->permutation == NULL ||
        plan->mixed_permutation == NULL ||
        plan->root == NULL ||
        plan->replicated_root == NULL ||
        plan->finish_re == NULL || plan->finish_im == NULL ||
        plan->work == NULL) {
        lane4_portable_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        plan->permutation[i] =
            4 * reverse_bits((uint32_t)i, plan->inner_levels);
        plan->mixed_permutation[i] =
            4 * reverse_mixed_radix(
                (uint32_t)i, plan->inner_levels);
    }
    for (i = 0; i < plan->inner_size; ++i) {
        const float angle =
            -2.0f * LANE4_PI * (float)i / (float)plan->inner_size;
        plan->root[i].re = cosf(angle);
        plan->root[i].im = sinf(angle);
    }
    {
        size_t output = 0;
        size_t length =
            (plan->inner_levels & 1U) != 0 ? 8 : 4;

        /*
         * The SSE stage consumes three coefficients per nonzero radix-4
         * frequency.  Store that exact stream once so execution needs no
         * multiply or indexed root addressing in the butterfly loop.
         */
        while (length < plan->inner_size) {
            const size_t root_stride =
                plan->inner_size / (4 * length);
            size_t k;

            for (k = 1; k < length; ++k) {
                unsigned multiple;
                for (multiple = 1; multiple <= 3; ++multiple) {
                    const fft_complex value =
                        plan->root[multiple * k * root_stride];
                    unsigned lane;
                    for (lane = 0; lane < 4; ++lane) {
                        plan->replicated_root[output].re[lane] =
                            value.re;
                        plan->replicated_root[output].im[lane] =
                            value.im;
                    }
                    ++output;
                }
            }
            length *= 4;
        }
    }
    {
        unsigned lane;
        for (lane = 1; lane < 4; ++lane) {
            float *factor_re =
                plan->finish_re + (lane - 1) * plan->inner_size;
            float *factor_im =
                plan->finish_im + (lane - 1) * plan->inner_size;
            for (i = 0; i < plan->inner_size; ++i) {
                const float angle =
                    -2.0f * LANE4_PI * (float)(lane * i) / (float)n;
                factor_re[i] = cosf(angle);
                factor_im[i] = sinf(angle);
            }
        }
    }

    return plan;
}

void lane4_portable_plan_destroy(lane4_portable_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->permutation);
    free(plan->mixed_permutation);
    free(plan->root);
    free(plan->replicated_root);
    free(plan->finish_re);
    free(plan->finish_im);
    free(plan->work);
    free(plan);
}

static fft_complex multiply(fft_complex value, float re, float im)
{
    const fft_complex result = {
        value.re * re - value.im * im,
        value.re * im + value.im * re
    };
    return result;
}

static void finish_scalar(lane4_portable_plan *plan,
                          fft_complex *data)
{
    const size_t inner_size = plan->inner_size;
    size_t frequency;

    for (frequency = 0; frequency < inner_size; ++frequency) {
        fft_complex a = {
            plan->work[frequency].re[0],
            plan->work[frequency].im[0]
        };
        fft_complex b = {
            plan->work[frequency].re[1],
            plan->work[frequency].im[1]
        };
        fft_complex c = {
            plan->work[frequency].re[2],
            plan->work[frequency].im[2]
        };
        fft_complex d = {
            plan->work[frequency].re[3],
            plan->work[frequency].im[3]
        };
        fft_complex ac_sum;
        fft_complex ac_difference;
        fft_complex bd_sum;
        fft_complex rotated;

        b = multiply(
            b,
            plan->finish_re[frequency],
            plan->finish_im[frequency]);
        c = multiply(
            c,
            plan->finish_re[inner_size + frequency],
            plan->finish_im[inner_size + frequency]);
        d = multiply(
            d,
            plan->finish_re[2 * inner_size + frequency],
            plan->finish_im[2 * inner_size + frequency]);

        ac_sum.re = a.re + c.re;
        ac_sum.im = a.im + c.im;
        ac_difference.re = a.re - c.re;
        ac_difference.im = a.im - c.im;
        bd_sum.re = b.re + d.re;
        bd_sum.im = b.im + d.im;
        rotated.re = b.im - d.im;
        rotated.im = d.re - b.re;

        data[frequency].re = ac_sum.re + bd_sum.re;
        data[frequency].im = ac_sum.im + bd_sum.im;
        data[inner_size + frequency].re =
            ac_difference.re + rotated.re;
        data[inner_size + frequency].im =
            ac_difference.im + rotated.im;
        data[2 * inner_size + frequency].re = ac_sum.re - bd_sum.re;
        data[2 * inner_size + frequency].im = ac_sum.im - bd_sum.im;
        data[3 * inner_size + frequency].re =
            ac_difference.re - rotated.re;
        data[3 * inner_size + frequency].im =
            ac_difference.im - rotated.im;
    }
}

int lane4_c_execute(lane4_portable_plan *plan, fft_complex *data)
{
    size_t i;
    size_t length;

    if (plan == NULL || data == NULL) {
        return -1;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        const fft_complex *input = data + plan->permutation[i];
        unsigned lane;
        for (lane = 0; lane < 4; ++lane) {
            plan->work[i].re[lane] = input[lane].re;
            plan->work[i].im[lane] = input[lane].im;
        }
    }

    for (length = 2; length <= plan->inner_size; length *= 2) {
        const size_t half = length / 2;
        const size_t root_stride = plan->inner_size / length;
        size_t start;

        for (start = 0; start < plan->inner_size; start += length) {
            size_t k;
            for (k = 0; k < half; ++k) {
                lane4_portable_row *a = plan->work + start + k;
                lane4_portable_row *b = a + half;
                const fft_complex factor = plan->root[k * root_stride];
                unsigned lane;

                for (lane = 0; lane < 4; ++lane) {
                    const float b_re =
                        b->re[lane] * factor.re -
                        b->im[lane] * factor.im;
                    const float b_im =
                        b->re[lane] * factor.im +
                        b->im[lane] * factor.re;
                    const float a_re = a->re[lane];
                    const float a_im = a->im[lane];
                    a->re[lane] = a_re + b_re;
                    a->im[lane] = a_im + b_im;
                    b->re[lane] = a_re - b_re;
                    b->im[lane] = a_im - b_im;
                }
            }
        }
    }

    finish_scalar(plan, data);
    return 0;
}
