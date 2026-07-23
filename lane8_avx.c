#include "lane8_avx_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define LANE8_PI 3.141592653589793238462643383279502884f

#if HAVE_TANGENT_X86_ASM
_Static_assert(offsetof(lane8_avx_plan, n) == 0,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, inner_size) == 8,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, inner_levels) == 16,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, permutation) == 24,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, replicated_root) == 32,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, finish_re) == 40,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, finish_im) == 48,
               "update lane8 assembly plan offsets");
_Static_assert(offsetof(lane8_avx_plan, work) == 56,
               "update lane8 assembly plan offsets");
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
 * Reverse the digits of a base-(2,4,4,...) DIT factorization.  Odd level
 * counts >= 3 fuse the initial radix-2 and radix-4 into an FFT8 codelet.
 */
static uint32_t reverse_mixed_radix(uint32_t value, unsigned levels)
{
    unsigned radices[16];
    unsigned digits[16] = {0};
    const unsigned count = (levels + 1U) / 2U;
    uint32_t result = 0;
    uint32_t multiplier = 1;
    unsigned i;

    i = 0;
    if ((levels & 1U) != 0) {
        radices[i++] = 2;
    }
    while (i < count) {
        radices[i++] = 4;
    }
    for (i = 0; i < count; ++i) {
        digits[i] = value % radices[i];
        value /= radices[i];
    }
    for (i = count; i-- > 0;) {
        result += digits[i] * multiplier;
        multiplier *= radices[i];
    }
    return result;
}

lane8_avx_plan *lane8_avx_plan_create(size_t n)
{
    lane8_avx_plan *plan;
    size_t root_count = 0;
    size_t i;

    if (n < 32 || (n & (n - 1)) != 0) {
        return NULL;
    }
    plan = (lane8_avx_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->inner_size = n / 8;
    plan->inner_levels = integer_log2(plan->inner_size);
    plan->permutation = (uint32_t *)malloc(
        plan->inner_size * sizeof(*plan->permutation));

    {
        size_t length;
        if (plan->inner_size == 2) {
            length = 2;
        } else {
            length = (plan->inner_levels & 1U) != 0 ? 8 : 4;
        }
        while (length < plan->inner_size) {
            root_count += 3 * (length - 1);
            length *= 4;
        }
    }
    if (root_count != 0) {
        plan->replicated_root = (lane8_avx_root *)aligned_alloc(
            64, round_up(root_count * sizeof(*plan->replicated_root), 64));
    }
    plan->finish_re = (float *)aligned_alloc(
        64, round_up(7 * plan->inner_size * sizeof(float), 64));
    plan->finish_im = (float *)aligned_alloc(
        64, round_up(7 * plan->inner_size * sizeof(float), 64));
    plan->work = (lane8_avx_row *)aligned_alloc(
        64, round_up(plan->inner_size * sizeof(*plan->work), 64));
    if (plan->permutation == NULL ||
        (root_count != 0 && plan->replicated_root == NULL) ||
        plan->finish_re == NULL || plan->finish_im == NULL ||
        plan->work == NULL) {
        lane8_avx_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        plan->permutation[i] =
            8 * reverse_mixed_radix((uint32_t)i, plan->inner_levels);
    }
    {
        size_t output = 0;
        size_t length;
        if (plan->inner_size == 2) {
            length = 2;
        } else {
            length = (plan->inner_levels & 1U) != 0 ? 8 : 4;
        }
        while (length < plan->inner_size) {
            const size_t root_stride = plan->inner_size / (4 * length);
            size_t k;
            for (k = 1; k < length; ++k) {
                unsigned multiple;
                for (multiple = 1; multiple <= 3; ++multiple) {
                    const float angle =
                        -2.0f * LANE8_PI *
                        (float)(multiple * k * root_stride) /
                        (float)plan->inner_size;
                    unsigned lane;
                    for (lane = 0; lane < 8; ++lane) {
                        plan->replicated_root[output].re[lane] = cosf(angle);
                        plan->replicated_root[output].im[lane] = sinf(angle);
                    }
                    ++output;
                }
            }
            length *= 4;
        }
    }
    {
        unsigned lane;
        for (lane = 1; lane < 8; ++lane) {
            float *re = plan->finish_re + (lane - 1) * plan->inner_size;
            float *im = plan->finish_im + (lane - 1) * plan->inner_size;
            for (i = 0; i < plan->inner_size; ++i) {
                const float angle =
                    -2.0f * LANE8_PI * (float)(lane * i) / (float)n;
                re[i] = cosf(angle);
                im[i] = sinf(angle);
            }
        }
    }
    return plan;
}

void lane8_avx_plan_destroy(lane8_avx_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->permutation);
    free(plan->replicated_root);
    free(plan->finish_re);
    free(plan->finish_im);
    free(plan->work);
    free(plan);
}
