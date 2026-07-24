#include "h16_hybrid.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    size_t begin;
    size_t length;
} h16_partition;

typedef struct {
    size_t **items;
    size_t *lengths;
    size_t count;
    size_t capacity;
} partition_builder;

struct h16_hybrid_plan {
    size_t n;
    unsigned levels;
    size_t *permutation;
    size_t *partition_indices;
    h16_partition *partitions;
    size_t partition_count;
    fft_complex **twiddles;
    fft_complex *uprooted;
    fft_complex *wht_work;
    float paired_real[8 * 8];
    float paired_imag[8 * 8];
    int paired_avx2_available;
};

#if HAVE_TANGENT_X86_ASM
extern void h16_paired_h8_avx2(float *real_values,
                               float *imaginary_values);
#endif

static int is_power_of_two(size_t n)
{
    return n != 0 && (n & (n - 1)) == 0;
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

static int paired_avx2_runtime_available(void)
{
#if HAVE_TANGENT_X86_ASM && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

static int compare_partition_length(const void *left_pointer,
                                    const void *right_pointer)
{
    const h16_partition *left =
        (const h16_partition *)left_pointer;
    const h16_partition *right =
        (const h16_partition *)right_pointer;

    if (left->length < right->length) {
        return -1;
    }
    if (left->length > right->length) {
        return 1;
    }
    return 0;
}

static int append_partition(partition_builder *builder,
                            const size_t *indices,
                            size_t length)
{
    size_t *copy;

    if (builder->count == builder->capacity) {
        const size_t new_capacity =
            builder->capacity == 0 ? 16 : builder->capacity * 2;
        size_t **new_items = (size_t **)realloc(
            builder->items, new_capacity * sizeof(*new_items));
        size_t *new_lengths = (size_t *)realloc(
            builder->lengths, new_capacity * sizeof(*new_lengths));

        if (new_items == NULL || new_lengths == NULL) {
            /*
             * A successful realloc remains owned by the builder.  Preserve it
             * so the common cleanup path can release everything.
             */
            if (new_items != NULL) {
                builder->items = new_items;
            }
            if (new_lengths != NULL) {
                builder->lengths = new_lengths;
            }
            return 0;
        }
        builder->items = new_items;
        builder->lengths = new_lengths;
        builder->capacity = new_capacity;
    }

    copy = (size_t *)malloc(length * sizeof(*copy));
    if (copy == NULL) {
        return 0;
    }
    for (size_t i = 0; i < length; ++i) {
        copy[i] = indices[i];
    }
    builder->items[builder->count] = copy;
    builder->lengths[builder->count] = length;
    ++builder->count;
    return 1;
}

static void destroy_partition_builder(partition_builder *builder)
{
    size_t i;
    for (i = 0; i < builder->count; ++i) {
        free(builder->items[i]);
    }
    free(builder->items);
    free(builder->lengths);
    builder->items = NULL;
    builder->lengths = NULL;
    builder->count = 0;
    builder->capacity = 0;
}

/*
 * Algorithm 11 (PARTITION) from Alman--Rao.  Indices refer to the
 * recursively split-radix-permuted input, not to natural input order.
 */
static int build_h_prime_partitions(partition_builder *result,
                                    size_t n,
                                    size_t base)
{
    partition_builder quarter = {0};
    size_t singleton;
    size_t i;

    if (n == 1) {
        singleton = base;
        return append_partition(result, &singleton, 1);
    }
    if (n == 2) {
        singleton = base;
        if (!append_partition(result, &singleton, 1)) {
            return 0;
        }
        singleton = base + 1;
        return append_partition(result, &singleton, 1);
    }

    if (!build_h_prime_partitions(result, n / 2, base) ||
        !build_h_prime_partitions(&quarter, n / 4, 0)) {
        destroy_partition_builder(&quarter);
        return 0;
    }

    for (i = 0; i < quarter.count; ++i) {
        const size_t old_length = quarter.lengths[i];
        const size_t new_length = old_length * 2;
        size_t *combined =
            (size_t *)malloc(new_length * sizeof(*combined));
        size_t j;

        if (combined == NULL) {
            destroy_partition_builder(&quarter);
            return 0;
        }
        for (j = 0; j < old_length; ++j) {
            combined[2 * j] =
                base + n / 2 + quarter.items[i][j];
            combined[2 * j + 1] =
                base + 3 * n / 4 + quarter.items[i][j];
        }
        if (!append_partition(result, combined, new_length)) {
            free(combined);
            destroy_partition_builder(&quarter);
            return 0;
        }
        free(combined);
    }
    destroy_partition_builder(&quarter);
    return 1;
}

/*
 * Produce the input order implied by recursively concatenating the
 * even, 4j+1, and 4j-1 split-radix branches.
 */
static int build_split_radix_permutation_recursive(
    const size_t *input,
    size_t n,
    size_t *output,
    size_t *output_position)
{
    size_t *children;
    size_t *even;
    size_t *odd_positive;
    size_t *odd_negative;
    size_t j;
    int ok;

    if (n <= 2) {
        for (j = 0; j < n; ++j) {
            output[(*output_position)++] = input[j];
        }
        return 1;
    }

    children = (size_t *)malloc(n * sizeof(*children));
    if (children == NULL) {
        return 0;
    }
    even = children;
    odd_positive = children + n / 2;
    odd_negative = children + 3 * n / 4;
    for (j = 0; j < n / 2; ++j) {
        even[j] = input[2 * j];
    }
    for (j = 0; j < n / 4; ++j) {
        odd_positive[j] = input[4 * j + 1];
        odd_negative[j] = input[(4 * j + n - 1) % n];
    }

    ok =
        build_split_radix_permutation_recursive(
            even, n / 2, output, output_position) &&
        build_split_radix_permutation_recursive(
            odd_positive, n / 4, output, output_position) &&
        build_split_radix_permutation_recursive(
            odd_negative, n / 4, output, output_position);
    free(children);
    return ok;
}

static void folklore_wht(fft_complex *data, size_t n)
{
    size_t width;

    for (width = 1; width < n; width *= 2) {
        size_t base;
        for (base = 0; base < n; base += 2 * width) {
            size_t j;
            for (j = 0; j < width; ++j) {
                const fft_complex a = data[base + j];
                const fft_complex b = data[base + width + j];
                data[base + j].re = a.re + b.re;
                data[base + j].im = a.im + b.im;
                data[base + width + j].re = a.re - b.re;
                data[base + width + j].im = a.im - b.im;
            }
        }
    }
}

static void scaled_h8(const fft_complex input[8], fft_complex output[8])
{
    fft_complex r1;
    fft_complex r2;
    fft_complex r3;
    fft_complex r4;
    fft_complex r5;
    fft_complex t;
    fft_complex middle;
    fft_complex middle_d;
    fft_complex middle_e;
    fft_complex middle_h;
    fft_complex temporary;

#define COMPLEX_ADD(destination, left, right) \
    do {                                      \
        (destination).re = (left).re + (right).re; \
        (destination).im = (left).im + (right).im; \
    } while (0)
#define COMPLEX_SUB(destination, left, right) \
    do {                                      \
        (destination).re = (left).re - (right).re; \
        (destination).im = (left).im - (right).im; \
    } while (0)

    COMPLEX_ADD(r1, input[1], input[2]);
    COMPLEX_ADD(r2, input[3], input[7]);
    COMPLEX_ADD(r3, input[5], input[6]);
    COMPLEX_ADD(r4, r1, r2);
    COMPLEX_ADD(r5, r3, input[4]);
    t.re = (r4.re + r5.re) * 0.5f;
    t.im = (r4.im + r5.im) * 0.5f;
    COMPLEX_SUB(middle, input[0], t);
    COMPLEX_ADD(middle_d, middle, input[3]);
    COMPLEX_ADD(middle_e, middle, input[4]);
    COMPLEX_ADD(middle_h, middle, input[7]);

    COMPLEX_ADD(output[0], input[0], t);
    COMPLEX_ADD(temporary, middle_e, input[2]);
    COMPLEX_ADD(output[1], temporary, input[6]);
    COMPLEX_ADD(temporary, middle_e, input[1]);
    COMPLEX_ADD(output[2], temporary, input[5]);
    COMPLEX_ADD(output[3], middle_e, r2);
    COMPLEX_ADD(output[4], middle_d, r1);
    COMPLEX_ADD(temporary, middle_h, input[2]);
    COMPLEX_ADD(output[5], temporary, input[5]);
    COMPLEX_ADD(temporary, middle_h, input[1]);
    COMPLEX_ADD(output[6], temporary, input[6]);
    COMPLEX_ADD(output[7], middle_d, r3);

#undef COMPLEX_ADD
#undef COMPLEX_SUB
}

/*
 * Proposed scaled-H16 recursion.  Branches 0 and 8 retain the scale;
 * the other fourteen branches are doubled.  The two H8 circuits consume
 * exactly that physical input contract.
 */
static void scaled_h16_wht(fft_complex *data, size_t n, unsigned scale_power)
{
    size_t block;
    size_t branch;
    size_t j;

    if (n < 16) {
        if (scale_power != 0) {
            const float scale = scalbnf(1.0f, (int)scale_power);
            for (j = 0; j < n; ++j) {
                data[j].re *= scale;
                data[j].im *= scale;
            }
        }
        folklore_wht(data, n);
        return;
    }

    block = n / 16;
    for (branch = 0; branch < 16; ++branch) {
        const unsigned child_scale =
            scale_power + (branch != 0 && branch != 8);
        scaled_h16_wht(data + branch * block, block, child_scale);
    }

    for (j = 0; j < block; ++j) {
        fft_complex plus[8];
        fft_complex minus[8];
        fft_complex plus_output[8];
        fft_complex minus_output[8];

        for (branch = 0; branch < 8; ++branch) {
            const fft_complex left = data[branch * block + j];
            const fft_complex right =
                data[(branch + 8) * block + j];
            plus[branch].re = left.re + right.re;
            plus[branch].im = left.im + right.im;
            minus[branch].re = left.re - right.re;
            minus[branch].im = left.im - right.im;
        }
        scaled_h8(plus, plus_output);
        scaled_h8(minus, minus_output);
        for (branch = 0; branch < 8; ++branch) {
            data[branch * block + j] = plus_output[branch];
            data[(branch + 8) * block + j] =
                minus_output[branch];
        }
    }
}

static void prepare_scaled_h16_children(fft_complex *data, size_t n)
{
    const size_t block = n / 16;
    size_t branch;

    for (branch = 0; branch < 16; ++branch) {
        const unsigned child_scale =
            branch != 0 && branch != 8;
        scaled_h16_wht(
            data + branch * block, block, child_scale);
    }
}

static void paired_h16_batch(h16_hybrid_plan *plan,
                             const h16_partition partitions[4],
                             fft_complex *data)
{
    const size_t length = partitions[0].length;
    const size_t block = length / 16;
    size_t transform;
    size_t coordinate;
    size_t j;

    for (transform = 0; transform < 4; ++transform) {
        fft_complex *work =
            plan->wht_work + transform * length;
        for (j = 0; j < length; ++j) {
            const size_t uprooted_index =
                plan->partition_indices[
                    partitions[transform].begin + j];
            work[j] =
                data[plan->permutation[uprooted_index]];
        }
        prepare_scaled_h16_children(work, length);
    }

    for (j = 0; j < block; ++j) {
        for (coordinate = 0; coordinate < 8; ++coordinate) {
            float *packed_real =
                plan->paired_real + coordinate * 8;
            float *packed_imaginary =
                plan->paired_imag + coordinate * 8;

            for (transform = 0; transform < 4; ++transform) {
                fft_complex *work =
                    plan->wht_work + transform * length;
                const fft_complex left =
                    work[coordinate * block + j];
                const fft_complex right =
                    work[(coordinate + 8) * block + j];
                const size_t plus_lane = 2 * transform;
                const size_t minus_lane = plus_lane + 1;

                packed_real[plus_lane] = left.re + right.re;
                packed_real[minus_lane] = left.re - right.re;
                packed_imaginary[plus_lane] =
                    left.im + right.im;
                packed_imaginary[minus_lane] =
                    left.im - right.im;
            }
        }

#if HAVE_TANGENT_X86_ASM
        h16_paired_h8_avx2(
            plan->paired_real, plan->paired_imag);
#endif

        for (coordinate = 0; coordinate < 8; ++coordinate) {
            const float *packed_real =
                plan->paired_real + coordinate * 8;
            const float *packed_imaginary =
                plan->paired_imag + coordinate * 8;

            for (transform = 0; transform < 4; ++transform) {
                fft_complex *work =
                    plan->wht_work + transform * length;
                const size_t plus_lane = 2 * transform;
                const size_t minus_lane = plus_lane + 1;

                work[coordinate * block + j].re =
                    packed_real[plus_lane];
                work[coordinate * block + j].im =
                    packed_imaginary[plus_lane];
                work[(coordinate + 8) * block + j].re =
                    packed_real[minus_lane];
                work[(coordinate + 8) * block + j].im =
                    packed_imaginary[minus_lane];
            }
        }
    }

    for (transform = 0; transform < 4; ++transform) {
        const fft_complex *work =
            plan->wht_work + transform * length;
        for (j = 0; j < length; ++j) {
            const size_t uprooted_index =
                plan->partition_indices[
                    partitions[transform].begin + j];
            plan->uprooted[uprooted_index] = work[j];
        }
    }
}

static void twiddle_network(const h16_hybrid_plan *plan,
                            const fft_complex *input,
                            fft_complex *output,
                            size_t n,
                            unsigned level)
{
    size_t quarter;
    size_t k;

    if (n == 1) {
        output[0] = input[0];
        return;
    }
    if (n == 2) {
        output[0].re = input[0].re + input[1].re;
        output[0].im = input[0].im + input[1].im;
        output[1].re = input[0].re - input[1].re;
        output[1].im = input[0].im - input[1].im;
        return;
    }

    quarter = n / 4;
    twiddle_network(
        plan, input, output, n / 2, level - 1);
    twiddle_network(plan,
                    input + n / 2,
                    output + n / 2,
                    n / 4,
                    level - 2);
    twiddle_network(plan,
                    input + 3 * n / 4,
                    output + 3 * n / 4,
                    n / 4,
                    level - 2);

    {
        const fft_complex a = output[0];
        const fft_complex z = output[quarter];
        const fft_complex b = output[2 * quarter];
        const fft_complex c_value = output[3 * quarter];

        output[0].re = a.re + b.re;
        output[0].im = a.im + b.im;
        output[2 * quarter].re = a.re - b.re;
        output[2 * quarter].im = a.im - b.im;
        output[quarter].re = z.re + c_value.im;
        output[quarter].im = z.im - c_value.re;
        output[3 * quarter].re = z.re - c_value.im;
        output[3 * quarter].im = z.im + c_value.re;
    }

    for (k = 1; k < quarter; ++k) {
        const fft_complex a = output[k];
        const fft_complex z = output[k + quarter];
        const fft_complex b = output[k + 2 * quarter];
        const fft_complex c_value = output[k + 3 * quarter];
        const float root_re = plan->twiddles[level][k].re;
        const float root_im = plan->twiddles[level][k].im;
        fft_complex u;
        fft_complex v;

        u.re = root_re * b.re - root_im * c_value.im;
        u.im = root_re * b.im + root_im * c_value.re;
        v.re = root_im * b.re + root_re * c_value.im;
        v.im = root_im * b.im - root_re * c_value.re;

        output[k].re = a.re + u.re;
        output[k].im = a.im + u.im;
        output[k + 2 * quarter].re = a.re - u.re;
        output[k + 2 * quarter].im = a.im - u.im;
        output[k + quarter].re = z.re + v.re;
        output[k + quarter].im = z.im + v.im;
        output[k + 3 * quarter].re = z.re - v.re;
        output[k + 3 * quarter].im = z.im - v.im;
    }
}

h16_hybrid_plan *h16_hybrid_plan_create(size_t n)
{
    h16_hybrid_plan *plan;
    partition_builder builder = {0};
    size_t *natural = NULL;
    size_t permutation_position = 0;
    size_t flat_position = 0;
    size_t i;
    const float pi = acosf(-1.0f);

    if (!is_power_of_two(n)) {
        return NULL;
    }
    plan = (h16_hybrid_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->levels = integer_log2(n);
    plan->paired_avx2_available =
        paired_avx2_runtime_available();
    plan->permutation =
        (size_t *)malloc(n * sizeof(*plan->permutation));
    natural = (size_t *)malloc(n * sizeof(*natural));
    plan->uprooted =
        (fft_complex *)malloc(n * sizeof(*plan->uprooted));
    plan->wht_work =
        (fft_complex *)malloc(n * sizeof(*plan->wht_work));
    plan->twiddles = (fft_complex **)calloc(
        plan->levels + 1, sizeof(*plan->twiddles));
    if (plan->permutation == NULL || natural == NULL ||
        plan->uprooted == NULL || plan->wht_work == NULL ||
        plan->twiddles == NULL) {
        free(natural);
        h16_hybrid_plan_destroy(plan);
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        natural[i] = i;
    }
    if (!build_split_radix_permutation_recursive(
            natural, n, plan->permutation, &permutation_position) ||
        permutation_position != n ||
        !build_h_prime_partitions(&builder, n, 0)) {
        free(natural);
        destroy_partition_builder(&builder);
        h16_hybrid_plan_destroy(plan);
        return NULL;
    }
    free(natural);

    plan->partition_count = builder.count;
    plan->partitions = (h16_partition *)malloc(
        builder.count * sizeof(*plan->partitions));
    plan->partition_indices =
        (size_t *)malloc(n * sizeof(*plan->partition_indices));
    if (plan->partitions == NULL || plan->partition_indices == NULL) {
        destroy_partition_builder(&builder);
        h16_hybrid_plan_destroy(plan);
        return NULL;
    }
    for (i = 0; i < builder.count; ++i) {
        size_t j;
        plan->partitions[i].begin = flat_position;
        plan->partitions[i].length = builder.lengths[i];
        for (j = 0; j < builder.lengths[i]; ++j) {
            plan->partition_indices[flat_position++] =
                builder.items[i][j];
        }
    }
    destroy_partition_builder(&builder);
    if (flat_position != n) {
        h16_hybrid_plan_destroy(plan);
        return NULL;
    }
    qsort(plan->partitions,
          plan->partition_count,
          sizeof(*plan->partitions),
          compare_partition_length);

    for (i = 2; i <= plan->levels; ++i) {
        const size_t size = (size_t)1 << i;
        const size_t count = size / 4;
        size_t k;
        plan->twiddles[i] =
            (fft_complex *)malloc(count * sizeof(*plan->twiddles[i]));
        if (plan->twiddles[i] == NULL) {
            h16_hybrid_plan_destroy(plan);
            return NULL;
        }
        for (k = 0; k < count; ++k) {
            const float angle =
                2.0f * pi * (float)k / (float)size;
            plan->twiddles[i][k].re = cosf(angle);
            plan->twiddles[i][k].im = -sinf(angle);
        }
    }
    return plan;
}

void h16_hybrid_plan_destroy(h16_hybrid_plan *plan)
{
    unsigned level;
    if (plan == NULL) {
        return;
    }
    if (plan->twiddles != NULL) {
        for (level = 0; level <= plan->levels; ++level) {
            free(plan->twiddles[level]);
        }
    }
    free(plan->twiddles);
    free(plan->permutation);
    free(plan->partition_indices);
    free(plan->partitions);
    free(plan->uprooted);
    free(plan->wht_work);
    free(plan);
}

static int h16_hybrid_execute_internal(h16_hybrid_plan *plan,
                                      fft_complex *data,
                                      int use_paired_avx2)
{
    size_t partition_number;

    if (plan == NULL || data == NULL) {
        return -1;
    }
    partition_number = 0;
    while (partition_number < plan->partition_count) {
        const h16_partition partition =
            plan->partitions[partition_number];
        size_t j;

        if (use_paired_avx2 && partition.length >= 16 &&
            partition_number + 4 <= plan->partition_count &&
            plan->partitions[partition_number + 1].length ==
                partition.length &&
            plan->partitions[partition_number + 2].length ==
                partition.length &&
            plan->partitions[partition_number + 3].length ==
                partition.length) {
            paired_h16_batch(
                plan, plan->partitions + partition_number, data);
            partition_number += 4;
            continue;
        }

        for (j = 0; j < partition.length; ++j) {
            const size_t uprooted_index =
                plan->partition_indices[partition.begin + j];
            plan->wht_work[j] =
                data[plan->permutation[uprooted_index]];
        }
        scaled_h16_wht(plan->wht_work, partition.length, 0);
        for (j = 0; j < partition.length; ++j) {
            const size_t uprooted_index =
                plan->partition_indices[partition.begin + j];
            plan->uprooted[uprooted_index] = plan->wht_work[j];
        }
        ++partition_number;
    }

    twiddle_network(
        plan, plan->uprooted, data, plan->n, plan->levels);
    return 0;
}

int h16_hybrid_fft_execute(h16_hybrid_plan *plan, fft_complex *data)
{
    return h16_hybrid_execute_internal(plan, data, 0);
}

int h16_hybrid_paired_avx2_execute(h16_hybrid_plan *plan,
                                   fft_complex *data)
{
    if (plan == NULL || !plan->paired_avx2_available) {
        return -1;
    }
    return h16_hybrid_execute_internal(plan, data, 1);
}

int h16_hybrid_paired_avx2_available(const h16_hybrid_plan *plan)
{
    return plan != NULL && plan->paired_avx2_available;
}
