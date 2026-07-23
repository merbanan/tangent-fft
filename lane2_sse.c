#include "lane2_sse.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#define LANE2_PI 3.141592653589793238462643383279502884f

typedef struct {
    float re[4];
    float signed_im[4];
} lane2_sse_root;

typedef struct {
    fft_complex value[2];
} lane2_sse_row;

struct lane2_sse_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *mixed_permutation;
    lane2_sse_root *replicated_root;
    lane2_sse_root *finish_root;
    lane2_sse_row *work;
};

_Static_assert(offsetof(lane2_sse_plan, inner_size) == 8,
               "update lane2 assembly plan offsets");
_Static_assert(offsetof(lane2_sse_plan, inner_levels) == 16,
               "update lane2 assembly plan offsets");
_Static_assert(offsetof(lane2_sse_plan, mixed_permutation) == 24,
               "update lane2 assembly plan offsets");
_Static_assert(offsetof(lane2_sse_plan, replicated_root) == 32,
               "update lane2 assembly plan offsets");
_Static_assert(offsetof(lane2_sse_plan, finish_root) == 40,
               "update lane2 assembly plan offsets");
_Static_assert(offsetof(lane2_sse_plan, work) == 48,
               "update lane2 assembly plan offsets");
_Static_assert(sizeof(lane2_sse_root) == 32,
               "lane2 roots must occupy two XMM vectors");
_Static_assert(sizeof(lane2_sse_row) == 16,
               "lane2 rows must occupy one XMM vector");

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

static void set_replicated_root(lane2_sse_root *destination,
                                float re,
                                float im)
{
    unsigned lane;

    for (lane = 0; lane < 4; lane += 2) {
        destination->re[lane] = re;
        destination->re[lane + 1] = re;
        destination->signed_im[lane] = -im;
        destination->signed_im[lane + 1] = im;
    }
}

static void set_packed_root(lane2_sse_root *destination,
                            unsigned lane,
                            float re,
                            float im)
{
    const unsigned offset = 2 * lane;

    destination->re[offset] = re;
    destination->re[offset + 1] = re;
    destination->signed_im[offset] = -im;
    destination->signed_im[offset + 1] = im;
}

lane2_sse_plan *lane2_sse_plan_create(size_t n)
{
    lane2_sse_plan *plan;
    size_t i;
    size_t output = 0;
    size_t previous;

    if (n < 16 || (n & (n - 1)) != 0) {
        return NULL;
    }

    plan = (lane2_sse_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->inner_size = n / 2;
    plan->inner_levels = integer_log2(plan->inner_size);
    plan->mixed_permutation = (uint32_t *)malloc(
        plan->inner_size * sizeof(*plan->mixed_permutation));
    plan->replicated_root = (lane2_sse_root *)aligned_alloc(
        64,
        round_up(plan->inner_size * sizeof(*plan->replicated_root), 64));
    plan->finish_root = (lane2_sse_root *)aligned_alloc(
        64,
        round_up((plan->inner_size / 2) *
                 sizeof(*plan->finish_root), 64));
    plan->work = (lane2_sse_row *)aligned_alloc(
        64,
        round_up(plan->inner_size * sizeof(*plan->work), 64));

    if (plan->mixed_permutation == NULL ||
        plan->replicated_root == NULL ||
        plan->finish_root == NULL || plan->work == NULL) {
        lane2_sse_plan_destroy(plan);
        return NULL;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        plan->mixed_permutation[i] =
            2 * reverse_mixed_radix((uint32_t)i, plan->inner_levels);
    }

    previous = (plan->inner_levels & 1U) != 0 ? 8 : 4;
    while (previous < plan->inner_size) {
        const size_t root_stride =
            plan->inner_size / (4 * previous);
        size_t k;

        for (k = 1; k < previous; ++k) {
            unsigned multiple;

            for (multiple = 1; multiple <= 3; ++multiple) {
                const float angle =
                    -2.0f * LANE2_PI *
                    (float)(multiple * k * root_stride) /
                    (float)plan->inner_size;
                set_replicated_root(
                    &plan->replicated_root[output],
                    cosf(angle),
                    sinf(angle));
                ++output;
            }
        }
        previous *= 4;
    }

    for (i = 0; i < plan->inner_size; i += 2) {
        unsigned lane;

        for (lane = 0; lane < 2; ++lane) {
            const size_t frequency = i + lane;
            const float angle =
                -2.0f * LANE2_PI * (float)frequency / (float)n;
            set_packed_root(
                &plan->finish_root[i / 2],
                lane,
                cosf(angle),
                sinf(angle));
        }
    }

    return plan;
}

void lane2_sse_plan_destroy(lane2_sse_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->mixed_permutation);
    free(plan->replicated_root);
    free(plan->finish_root);
    free(plan->work);
    free(plan);
}
