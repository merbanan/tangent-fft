#include "bank8_avx_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define BANK8_PI 3.141592653589793238462643383279502884f

#if HAVE_TANGENT_X86_ASM
_Static_assert(offsetof(bank8_avx_plan, n) == 0,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, inner_size) == 8,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, inner_levels) == 16,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, permutation) == 24,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, twiddle) == 32,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, finish_factor) == 40,
               "update bank8 assembly plan offsets");
_Static_assert(offsetof(bank8_avx_plan, work) == 48,
               "update bank8 assembly plan offsets");
_Static_assert(sizeof(bank8_avx_work_row) == 64,
               "bank8 work rows must hold two YMM banks");
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

static void replicate(float value, bank8_avx_row *row)
{
    unsigned lane;

    for (lane = 0; lane < 8; ++lane) {
        row->value[lane] = value;
    }
}

static void make_twiddle(bank8_avx_twiddle *output,
                         size_t k,
                         size_t root_stride,
                         size_t inner_size)
{
    unsigned multiple;

    for (multiple = 1; multiple <= 3; ++multiple) {
        const float angle =
            -2.0f * BANK8_PI *
            (float)(multiple * k * root_stride) / (float)inner_size;
        bank8_avx_row *real_row =
            (bank8_avx_row *)output + 2 * (multiple - 1);
        bank8_avx_row *imag_row = real_row + 1;

        replicate(cosf(angle), real_row);
        replicate(sinf(angle), imag_row);
    }
}

static void make_finish_group(bank8_avx_row *output,
                              size_t first_frequency,
                              size_t n)
{
    unsigned residue;

    for (residue = 1; residue < 8; ++residue) {
        bank8_avx_row *real_row = output + 2 * (residue - 1);
        bank8_avx_row *imag_row = real_row + 1;
        unsigned frequency;

        for (frequency = 0; frequency < 4; ++frequency) {
            const float angle =
                -2.0f * BANK8_PI *
                (float)(residue * (first_frequency + frequency)) /
                (float)n;
            const float real = cosf(angle);
            const float imag = sinf(angle);

            real_row->value[2 * frequency] = real;
            real_row->value[2 * frequency + 1] = real;
            imag_row->value[2 * frequency] = imag;
            imag_row->value[2 * frequency + 1] = imag;
        }
    }
}

bank8_avx_plan *bank8_avx_plan_create(size_t n)
{
    bank8_avx_plan *plan;
    size_t twiddle_count = 0;
    size_t i;

    if (n < 32 || (n & (n - 1)) != 0) {
        return NULL;
    }

    plan = (bank8_avx_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->inner_size = n / 8;
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
        plan->twiddle = (bank8_avx_twiddle *)aligned_alloc(
            64, round_up(twiddle_count * sizeof(*plan->twiddle), 64));
    }
    plan->finish_factor = (bank8_avx_row *)aligned_alloc(
        64, round_up(14 * (plan->inner_size / 4) *
                     sizeof(*plan->finish_factor), 64));
    plan->work = (bank8_avx_work_row *)aligned_alloc(
        64, round_up(plan->inner_size * sizeof(*plan->work), 64));

    if (plan->permutation == NULL ||
        (twiddle_count != 0 && plan->twiddle == NULL) ||
        plan->finish_factor == NULL || plan->work == NULL) {
        bank8_avx_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        plan->permutation[i] = 8 * reverse_mixed_radix(
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
            plan->finish_factor + 14 * (i / 4), i, n);
    }
    return plan;
}

void bank8_avx_plan_destroy(bank8_avx_plan *plan)
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
