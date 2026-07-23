#include "fft.h"
#include "ffmpeg_fft.h"
#include "lane4_portable.h"
#if HAVE_TANGENT_X86_ASM
#include "lane4_fft.h"
#include "tangent_x86_asm.h"
#endif

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef TANGENT_PROFILE
#include <stdio.h>
#include <time.h>
#endif

#define FFT_PI 3.141592653589793238462643383279502884

typedef enum {
    CONST_IDENTITY,
    CONST_NEGATE,
    CONST_POS_I,
    CONST_NEG_I,
    CONST_REAL,
    CONST_IMAG,
    CONST_UNIT_DIAGONAL,
    CONST_DIAGONAL,
    CONST_TANGENT_REAL,
    CONST_TANGENT_IMAG,
    CONST_GENERIC
} constant_kind;

typedef struct {
    float re;
    float im;
    unsigned char kind;
} fft_constant;

typedef enum {
    TANGENT_NORMAL,
    TANGENT_S,
    TANGENT_S2,
    TANGENT_S4
} tangent_kind;

typedef struct {
    uint32_t offset;
    uint8_t level;
    unsigned char kind;
    uint16_t reserved;
} tangent_node;

_Static_assert(sizeof(tangent_node) == 8,
               "tangent schedule nodes must remain cache-compact");

struct fft_plan {
    size_t n;
    unsigned levels;
    unsigned table_levels;

    /* scale[p] stores one period of s_(2^p), of length max(1, 2^(p-2)). */
    float **scale;

    /* Constants indexed by transform level p and butterfly index k. */
    fft_constant **root;
    fft_complex **scaled_twiddle;
    fft_complex **tangent_twiddle;
    float **tangent_value;

    /* Real output-scale ratios used by the S2 and S4 routines. */
    float **s2_low;
    float **s2_high;
    float **s4_0;
    float **s4_1;
    float **s4_2;
    float **s4_3;
    fft_complex **x86_s2_low;
    fft_complex **x86_s2_high;
    fft_complex **x86_s4_0;
    fft_complex **x86_s4_1;
    fft_complex **x86_s4_2;
    fft_complex **x86_s4_3;

    fft_complex *scratch;
    ffmpeg_fft_plan *ffmpeg;
    lane4_portable_plan *lane4_portable;
#if HAVE_TANGENT_X86_ASM
    lane4_fft_plan *lane4;
    lane4_avx_fft_plan *lane4_avx;
    lane4_avx_fma_fft_plan *lane4_avx_fma;
    lane4_avx2_fft_plan *lane4_avx2;
#endif
    uint32_t *tangent_permutation;
    uint32_t **blocked_permutation;
    uint32_t *tangent_bases;
    size_t tangent_base_count;
    size_t tangent_base_capacity;
    tangent_node *tangent_nodes;
    size_t tangent_node_count;
    size_t tangent_node_capacity;
    uint32_t *x86_base_offsets;
    size_t x86_base_start[4];
    size_t x86_base_count[4];
    uint32_t *x86_node_offsets;
    size_t *x86_node_start;
    size_t *x86_node_count;
    uint32_t *x86_leaf_offsets;
    size_t x86_leaf_start[5 * 4];
    size_t x86_leaf_count[5 * 4];
    fft_complex *x86_leaf_tables;
    int tangent_x86_asm_available;
    unsigned lane4_cpu_features;
};

enum {
    LANE4_CPU_SSE = 1U << 0,
    LANE4_CPU_SSE2 = 1U << 1,
    LANE4_CPU_SSE3 = 1U << 2,
    LANE4_CPU_SSSE3 = 1U << 3,
    LANE4_CPU_SSE41 = 1U << 4,
    LANE4_CPU_SSE42 = 1U << 5,
    LANE4_CPU_AVX = 1U << 6,
    LANE4_CPU_FMA = 1U << 7,
    LANE4_CPU_AVX2 = 1U << 8
};

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

static size_t level_size(unsigned level)
{
    return (size_t)1 << level;
}

static int tangent_x86_runtime_available(void)
{
#if HAVE_TANGENT_X86_ASM && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

static unsigned lane4_runtime_features(void)
{
#if HAVE_TANGENT_X86_ASM && (defined(__GNUC__) || defined(__clang__))
    unsigned result = 0;

    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse")) {
        result |= LANE4_CPU_SSE;
    }
    if (__builtin_cpu_supports("sse2")) {
        result |= LANE4_CPU_SSE2;
    }
    if (__builtin_cpu_supports("sse3")) {
        result |= LANE4_CPU_SSE3;
    }
    if (__builtin_cpu_supports("ssse3")) {
        result |= LANE4_CPU_SSSE3;
    }
    if (__builtin_cpu_supports("sse4.1")) {
        result |= LANE4_CPU_SSE41;
    }
    if (__builtin_cpu_supports("sse4.2")) {
        result |= LANE4_CPU_SSE42;
    }
    if (__builtin_cpu_supports("avx")) {
        result |= LANE4_CPU_AVX;
    }
    if (__builtin_cpu_supports("fma")) {
        result |= LANE4_CPU_FMA;
    }
    if (__builtin_cpu_supports("avx2")) {
        result |= LANE4_CPU_AVX2;
    }
    return result;
#else
    return 0;
#endif
}

static int nearly_equal(float a, float b)
{
    const float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return fabsf(a - b) <= 32.0f * FLT_EPSILON * scale;
}

static fft_constant make_constant(float re, float im)
{
    fft_constant result;
    const float ar = fabsf(re);
    const float ai = fabsf(im);

    result.re = re;
    result.im = im;
    result.kind = CONST_GENERIC;

    if (nearly_equal(im, 0.0)) {
        if (nearly_equal(re, 1.0)) {
            result.kind = CONST_IDENTITY;
        } else if (nearly_equal(re, -1.0)) {
            result.kind = CONST_NEGATE;
        } else {
            result.kind = CONST_REAL;
        }
    } else if (nearly_equal(re, 0.0)) {
        if (nearly_equal(im, 1.0)) {
            result.kind = CONST_POS_I;
        } else if (nearly_equal(im, -1.0)) {
            result.kind = CONST_NEG_I;
        } else {
            result.kind = CONST_IMAG;
        }
    } else if (nearly_equal(ar, ai)) {
        result.kind = nearly_equal(ar, 1.0)
                          ? CONST_UNIT_DIAGONAL
                          : CONST_DIAGONAL;
    } else if (nearly_equal(ar, 1.0)) {
        result.kind = CONST_TANGENT_REAL;
    } else if (nearly_equal(ai, 1.0)) {
        result.kind = CONST_TANGENT_IMAG;
    }

    return result;
}

/*
 * Multiplies by a preclassified constant. The diagonal and tangent cases use
 * the reduced-operation factorizations central to the compared algorithms.
 */
static fft_complex multiply_constant(fft_constant c, fft_complex z)
{
    fft_complex result;
    const float sign_re = c.re < 0.0f ? -1.0f : 1.0f;
    const float sign_im = c.im < 0.0f ? -1.0f : 1.0f;

    switch ((constant_kind)c.kind) {
    case CONST_IDENTITY:
        return z;
    case CONST_NEGATE:
        result.re = -z.re;
        result.im = -z.im;
        return result;
    case CONST_POS_I:
        result.re = -z.im;
        result.im = z.re;
        return result;
    case CONST_NEG_I:
        result.re = z.im;
        result.im = -z.re;
        return result;
    case CONST_REAL:
        result.re = c.re * z.re;
        result.im = c.re * z.im;
        return result;
    case CONST_IMAG:
        result.re = -c.im * z.im;
        result.im = c.im * z.re;
        return result;
    case CONST_UNIT_DIAGONAL:
        result.re = sign_re * z.re - sign_im * z.im;
        result.im = sign_im * z.re + sign_re * z.im;
        return result;
    case CONST_DIAGONAL: {
        const float magnitude = fabsf(c.re);
        result.re = magnitude * (sign_re * z.re - sign_im * z.im);
        result.im = magnitude * (sign_im * z.re + sign_re * z.im);
        return result;
    }
    case CONST_TANGENT_REAL:
        result.re = sign_re * z.re - c.im * z.im;
        result.im = c.im * z.re + sign_re * z.im;
        return result;
    case CONST_TANGENT_IMAG:
        result.re = c.re * z.re - sign_im * z.im;
        result.im = sign_im * z.re + c.re * z.im;
        return result;
    case CONST_GENERIC:
    default:
        result.re = c.re * z.re - c.im * z.im;
        result.im = c.im * z.re + c.re * z.im;
        return result;
    }
}

static fft_complex multiply_conjugate_constant(fft_constant c, fft_complex z)
{
    fft_complex conjugated_input = {z.re, -z.im};
    fft_complex result = multiply_constant(c, conjugated_input);
    result.im = -result.im;
    return result;
}

static fft_complex complex_add(fft_complex a, fft_complex b)
{
    fft_complex result = {a.re + b.re, a.im + b.im};
    return result;
}

static fft_complex complex_subtract(fft_complex a, fft_complex b)
{
    fft_complex result = {a.re - b.re, a.im - b.im};
    return result;
}

static fft_complex multiply_i(fft_complex z)
{
    fft_complex result = {-z.im, z.re};
    return result;
}

static fft_complex multiply_minus_i(fft_complex z)
{
    fft_complex result = {z.im, -z.re};
    return result;
}

static fft_complex scale_complex(fft_complex z, float scale)
{
    if (scale == 1.0f) {
        return z;
    }
    z.re *= scale;
    z.im *= scale;
    return z;
}

static size_t scale_period(unsigned level)
{
    return level < 2 ? 1 : level_size(level - 2);
}

static float scale_value(const fft_plan *plan, unsigned level, size_t k)
{
    const size_t period = scale_period(level);
    return plan->scale[level][k % period];
}

static int allocate_float_table(float ***table, unsigned count)
{
    *table = (float **)calloc(count, sizeof(**table));
    return *table != NULL;
}

static int allocate_constant_table(fft_constant ***table, unsigned count)
{
    *table = (fft_constant **)calloc(count, sizeof(**table));
    return *table != NULL;
}

static int allocate_complex_table(fft_complex ***table, unsigned count)
{
    *table = (fft_complex **)calloc(count, sizeof(**table));
    return *table != NULL;
}

static void free_float_table(float **table, unsigned count)
{
    unsigned i;
    if (table == NULL) {
        return;
    }
    for (i = 0; i < count; ++i) {
        free(table[i]);
    }
    free(table);
}

static void free_constant_table(fft_constant **table, unsigned count)
{
    unsigned i;
    if (table == NULL) {
        return;
    }
    for (i = 0; i < count; ++i) {
        free(table[i]);
    }
    free(table);
}

static void free_complex_table(fft_complex **table, unsigned count)
{
    unsigned i;
    if (table == NULL) {
        return;
    }
    for (i = 0; i < count; ++i) {
        free(table[i]);
    }
    free(table);
}

static int create_scale_tables(fft_plan *plan)
{
    unsigned level;

    if (!allocate_float_table(&plan->scale, plan->table_levels)) {
        return 0;
    }

    for (level = 0; level < plan->table_levels; ++level) {
        const size_t period = scale_period(level);
        size_t k;

        plan->scale[level] = (float *)malloc(period * sizeof(float));
        if (plan->scale[level] == NULL) {
            return 0;
        }

        if (level < 2) {
            plan->scale[level][0] = 1.0f;
            continue;
        }

        for (k = 0; k < period; ++k) {
            const size_t n = level_size(level);
            const float angle =
                2.0f * (float)FFT_PI * (float)k / (float)n;
            const float previous = scale_value(plan, level - 2, k);
            plan->scale[level][k] =
                previous * (k <= n / 8 ? cosf(angle) : sinf(angle));
        }
    }
    return 1;
}

static int create_float_level(float **table, unsigned level, size_t count)
{
    table[level] = (float *)malloc(count * sizeof(float));
    return table[level] != NULL;
}

static int create_constant_level(fft_constant **table,
                                 unsigned level,
                                 size_t count)
{
    table[level] = (fft_constant *)malloc(count * sizeof(fft_constant));
    return table[level] != NULL;
}

static int create_complex_level(fft_complex **table,
                                unsigned level,
                                size_t count)
{
    table[level] = (fft_complex *)malloc(count * sizeof(fft_complex));
    return table[level] != NULL;
}

static int create_transform_tables(fft_plan *plan)
{
    unsigned level;
    const unsigned count = plan->table_levels;

    if (!allocate_constant_table(&plan->root, count) ||
        !allocate_complex_table(&plan->scaled_twiddle, count) ||
        !allocate_complex_table(&plan->tangent_twiddle, count) ||
        !allocate_float_table(&plan->tangent_value, count) ||
        !allocate_float_table(&plan->s2_low, count) ||
        !allocate_float_table(&plan->s2_high, count) ||
        !allocate_float_table(&plan->s4_0, count) ||
        !allocate_float_table(&plan->s4_1, count) ||
        !allocate_float_table(&plan->s4_2, count) ||
        !allocate_float_table(&plan->s4_3, count) ||
        !allocate_complex_table(&plan->x86_s2_low, count) ||
        !allocate_complex_table(&plan->x86_s2_high, count) ||
        !allocate_complex_table(&plan->x86_s4_0, count) ||
        !allocate_complex_table(&plan->x86_s4_1, count) ||
        !allocate_complex_table(&plan->x86_s4_2, count) ||
        !allocate_complex_table(&plan->x86_s4_3, count)) {
        return 0;
    }

    for (level = 1; level <= plan->levels; ++level) {
        const size_t n = level_size(level);
        const size_t half = n / 2;
        size_t k;

        if (!create_constant_level(plan->root, level, half)) {
            return 0;
        }
        for (k = 0; k < half; ++k) {
            const float angle =
                -2.0f * (float)FFT_PI * (float)k / (float)n;
            plan->root[level][k] =
                make_constant(cosf(angle), sinf(angle));
        }

        if (level < 2) {
            continue;
        }

        {
            const size_t quarter = n / 4;
            if (!create_complex_level(plan->scaled_twiddle,
                                      level,
                                      quarter) ||
                !create_complex_level(plan->tangent_twiddle,
                                      level,
                                      quarter) ||
                !create_float_level(plan->tangent_value,
                                    level,
                                    quarter) ||
                !create_float_level(plan->s2_low, level, quarter) ||
                !create_float_level(plan->s2_high, level, quarter) ||
                !create_float_level(plan->s4_0, level, quarter) ||
                !create_float_level(plan->s4_1, level, quarter) ||
                !create_float_level(plan->s4_2, level, quarter) ||
                !create_float_level(plan->s4_3, level, quarter) ||
                !create_complex_level(plan->x86_s2_low, level, quarter) ||
                !create_complex_level(plan->x86_s2_high, level, quarter) ||
                !create_complex_level(plan->x86_s4_0, level, quarter) ||
                !create_complex_level(plan->x86_s4_1, level, quarter) ||
                !create_complex_level(plan->x86_s4_2, level, quarter) ||
                !create_complex_level(plan->x86_s4_3, level, quarter)) {
                return 0;
            }

            for (k = 0; k < quarter; ++k) {
                const float s_quarter = scale_value(plan, level - 2, k);
                const float s_n = scale_value(plan, level, k);
                const fft_constant root = plan->root[level][k];
                const float tangent_factor = s_quarter / s_n;

                plan->scaled_twiddle[level][k].re =
                    root.re * s_quarter;
                plan->scaled_twiddle[level][k].im =
                    root.im * s_quarter;
                plan->tangent_value[level][k] =
                    k <= n / 8 ? root.im * tangent_factor
                               : root.re * tangent_factor;
                plan->tangent_twiddle[level][k].re =
                    k <= n / 8 ? 1.0f
                               : plan->tangent_value[level][k];
                plan->tangent_twiddle[level][k].im =
                    k <= n / 8 ? plan->tangent_value[level][k]
                               : -1.0f;

                plan->s2_low[level][k] =
                    s_n / scale_value(plan, level + 1, k);
                plan->s2_high[level][k] =
                    s_n / scale_value(plan, level + 1, k + quarter);

                plan->s4_0[level][k] =
                    s_n / scale_value(plan, level + 2, k);
                plan->s4_1[level][k] =
                    s_n / scale_value(plan, level + 2, k + quarter);
                plan->s4_2[level][k] =
                    s_n / scale_value(plan, level + 2, k + 2 * quarter);
                plan->s4_3[level][k] =
                    s_n / scale_value(plan, level + 2, k + 3 * quarter);

                plan->x86_s2_low[level][k].re =
                    plan->x86_s2_low[level][k].im =
                        plan->s2_low[level][k];
                plan->x86_s2_high[level][k].re =
                    plan->x86_s2_high[level][k].im =
                        plan->s2_high[level][k];
                plan->x86_s4_0[level][k].re =
                    plan->x86_s4_0[level][k].im = plan->s4_0[level][k];
                plan->x86_s4_1[level][k].re =
                    plan->x86_s4_1[level][k].im = plan->s4_1[level][k];
                plan->x86_s4_2[level][k].re =
                    plan->x86_s4_2[level][k].im = plan->s4_2[level][k];
                plan->x86_s4_3[level][k].re =
                    plan->x86_s4_3[level][k].im = plan->s4_3[level][k];
            }
        }
    }
    return 1;
}

static void reorder_permutation(uint32_t *indices,
                                unsigned level,
                                uint32_t *temporary)
{
    const size_t n = level_size(level);
    size_t k;

    if (level < 2) {
        return;
    }

    {
        const size_t half = n / 2;
        const size_t quarter = n / 4;

        for (k = 0; k < half; ++k) {
            temporary[k] = indices[2 * k];
        }
        for (k = 0; k < quarter; ++k) {
            temporary[half + k] = indices[4 * k + 1];
            temporary[half + quarter + k] =
                indices[k == 0 ? n - 1 : 4 * k - 1];
        }
        memcpy(indices, temporary, n * sizeof(*indices));

        reorder_permutation(indices, level - 1, temporary);
        reorder_permutation(indices + half, level - 2, temporary);
        reorder_permutation(indices + half + quarter,
                            level - 2,
                            temporary);
    }
}

static int append_tangent_node(fft_plan *plan,
                               size_t offset,
                               unsigned level,
                               tangent_kind kind)
{
    tangent_node *node;
    if (plan->tangent_node_count == plan->tangent_node_capacity) {
        return 0;
    }
    node = &plan->tangent_nodes[plan->tangent_node_count++];
    node->offset = (uint32_t)offset;
    node->level = (uint8_t)level;
    node->kind = (unsigned char)kind;
    node->reserved = 0;
    return 1;
}

static int append_tangent_base(fft_plan *plan,
                               size_t offset,
                               tangent_kind kind)
{
    const uint32_t encoded =
        (uint32_t)offset | ((uint32_t)kind << 30);
    if (plan->tangent_base_count == plan->tangent_base_capacity) {
        return 0;
    }
    plan->tangent_bases[plan->tangent_base_count++] = encoded;
    return 1;
}

static int build_tangent_nodes(fft_plan *plan,
                               size_t offset,
                               unsigned level,
                               tangent_kind kind)
{
    size_t half;
    size_t quarter;
    tangent_kind even_kind;

    if (level == 0) {
        return 1;
    }
    if (level == 1) {
        return append_tangent_base(plan, offset, kind);
    }

    half = level_size(level - 1);
    quarter = level_size(level - 2);
    switch (kind) {
    case TANGENT_NORMAL:
        even_kind = TANGENT_NORMAL;
        break;
    case TANGENT_S:
        even_kind = TANGENT_S2;
        break;
    case TANGENT_S2:
        even_kind = TANGENT_S4;
        break;
    case TANGENT_S4:
    default:
        even_kind = TANGENT_S2;
        break;
    }

    if (!build_tangent_nodes(plan, offset, level - 1, even_kind) ||
        !build_tangent_nodes(plan,
                             offset + half,
                             level - 2,
                             TANGENT_S) ||
        !build_tangent_nodes(plan,
                             offset + half + quarter,
                             level - 2,
                             TANGENT_S)) {
        return 0;
    }
    return append_tangent_node(plan, offset, level, kind);
}

static int append_x86_leaf(tangent_node *leaves,
                           size_t capacity,
                           size_t *count,
                           size_t offset,
                           unsigned level,
                           tangent_kind kind)
{
    tangent_node *leaf;
    if (*count == capacity) {
        return 0;
    }
    leaf = &leaves[(*count)++];
    leaf->offset = (uint32_t)offset;
    leaf->level = (uint8_t)level;
    leaf->kind = (unsigned char)kind;
    leaf->reserved = 0;
    return 1;
}

/*
 * Cut the recursive tree at 8- and 16-point subtransforms.  The assembly
 * kernel evaluates each complete leaf while its samples remain hot, avoiding
 * the separate base, level-2, level-3 and level-4 passes.
 */
static int build_x86_leaves(tangent_node *leaves,
                            size_t capacity,
                            size_t *count,
                            size_t offset,
                            unsigned level,
                            tangent_kind kind)
{
    size_t half;
    size_t quarter;
    tangent_kind even_kind;

    if (level <= 4) {
        return append_x86_leaf(
            leaves, capacity, count, offset, level, kind);
    }

    half = level_size(level - 1);
    quarter = level_size(level - 2);
    switch (kind) {
    case TANGENT_NORMAL:
        even_kind = TANGENT_NORMAL;
        break;
    case TANGENT_S:
        even_kind = TANGENT_S2;
        break;
    case TANGENT_S2:
        even_kind = TANGENT_S4;
        break;
    case TANGENT_S4:
    default:
        even_kind = TANGENT_S2;
        break;
    }

    return build_x86_leaves(leaves,
                            capacity,
                            count,
                            offset,
                            level - 1,
                            even_kind) &&
           build_x86_leaves(leaves,
                            capacity,
                            count,
                            offset + half,
                            level - 2,
                            TANGENT_S) &&
           build_x86_leaves(leaves,
                            capacity,
                            count,
                            offset + half + quarter,
                            level - 2,
                            TANGENT_S);
}

enum {
    X86_LEAF_SCALED,
    X86_LEAF_TANGENT,
    X86_LEAF_S2_LOW,
    X86_LEAF_S2_HIGH,
    X86_LEAF_S4_0,
    X86_LEAF_S4_1,
    X86_LEAF_S4_2,
    X86_LEAF_S4_3,
    X86_LEAF_TABLE_COUNT
};

static int create_x86_leaf_tables(fft_plan *plan)
{
    const size_t level_stride = 4;
    const size_t table_stride = 5 * level_stride;
    const size_t entry_count = X86_LEAF_TABLE_COUNT * table_stride;
    size_t bytes = entry_count * sizeof(*plan->x86_leaf_tables);
    unsigned level;

    bytes = (bytes + 63U) & ~(size_t)63U;
    plan->x86_leaf_tables =
        (fft_complex *)aligned_alloc(64, bytes);
    if (plan->x86_leaf_tables == NULL) {
        return 0;
    }
    memset(plan->x86_leaf_tables, 0, bytes);

    for (level = 2; level <= 4 && level <= plan->levels; ++level) {
        const size_t quarter = level_size(level - 2);
        size_t k;
        for (k = 0; k < quarter; ++k) {
#define LEAF_ENTRY(table_id)                                            \
    plan->x86_leaf_tables[(table_id) * table_stride +                  \
                          (size_t)level * level_stride + k]
            LEAF_ENTRY(X86_LEAF_SCALED) =
                plan->scaled_twiddle[level][k];
            LEAF_ENTRY(X86_LEAF_TANGENT) =
                plan->tangent_twiddle[level][k];
            LEAF_ENTRY(X86_LEAF_S2_LOW) =
                (fft_complex){plan->s2_low[level][k],
                              plan->s2_low[level][k]};
            LEAF_ENTRY(X86_LEAF_S2_HIGH) =
                (fft_complex){plan->s2_high[level][k],
                              plan->s2_high[level][k]};
            LEAF_ENTRY(X86_LEAF_S4_0) =
                (fft_complex){plan->s4_0[level][k],
                              plan->s4_0[level][k]};
            LEAF_ENTRY(X86_LEAF_S4_1) =
                (fft_complex){plan->s4_1[level][k],
                              plan->s4_1[level][k]};
            LEAF_ENTRY(X86_LEAF_S4_2) =
                (fft_complex){plan->s4_2[level][k],
                              plan->s4_2[level][k]};
            LEAF_ENTRY(X86_LEAF_S4_3) =
                (fft_complex){plan->s4_3[level][k],
                              plan->s4_3[level][k]};
#undef LEAF_ENTRY
        }
    }
    return 1;
}

static int group_tangent_schedule(fft_plan *plan)
{
    const size_t group_count = plan->table_levels * 4;
    const size_t cursor_count = group_count < 5 * 4 ? 5 * 4 : group_count;
    tangent_node *leaves;
    size_t leaf_count = 0;
    size_t *cursor;
    size_t offset = 0;
    size_t i;
    unsigned kind;

    plan->x86_base_offsets =
        (uint32_t *)malloc((plan->tangent_base_count != 0
                                ? plan->tangent_base_count
                                : 1) *
                           sizeof(uint32_t));
    plan->x86_node_offsets =
        (uint32_t *)malloc((plan->tangent_node_count != 0
                                ? plan->tangent_node_count
                                : 1) *
                           sizeof(uint32_t));
    plan->x86_node_start =
        (size_t *)calloc(group_count, sizeof(size_t));
    plan->x86_node_count =
        (size_t *)calloc(group_count, sizeof(size_t));
    plan->x86_leaf_offsets =
        (uint32_t *)malloc(plan->n * sizeof(*plan->x86_leaf_offsets));
    leaves = (tangent_node *)malloc(plan->n * sizeof(*leaves));
    cursor = (size_t *)calloc(cursor_count, sizeof(size_t));
    if (plan->x86_base_offsets == NULL || plan->x86_node_offsets == NULL ||
        plan->x86_node_start == NULL || plan->x86_node_count == NULL ||
        plan->x86_leaf_offsets == NULL || leaves == NULL ||
        cursor == NULL) {
        free(leaves);
        free(cursor);
        return 0;
    }

    for (i = 0; i < plan->tangent_base_count; ++i) {
        ++plan->x86_base_count[plan->tangent_bases[i] >> 30];
    }
    for (kind = 0; kind < 4; ++kind) {
        plan->x86_base_start[kind] = offset;
        offset += plan->x86_base_count[kind];
        cursor[kind] = plan->x86_base_start[kind];
    }
    for (i = 0; i < plan->tangent_base_count; ++i) {
        const uint32_t encoded = plan->tangent_bases[i];
        const unsigned base_kind = encoded >> 30;
        plan->x86_base_offsets[cursor[base_kind]++] =
            encoded & UINT32_C(0x3fffffff);
    }

    memset(cursor, 0, cursor_count * sizeof(*cursor));
    for (i = 0; i < plan->tangent_node_count; ++i) {
        const tangent_node *node = &plan->tangent_nodes[i];
        ++plan->x86_node_count[(size_t)node->level * 4 + node->kind];
    }
    offset = 0;
    for (i = 0; i < group_count; ++i) {
        plan->x86_node_start[i] = offset;
        cursor[i] = offset;
        offset += plan->x86_node_count[i];
    }
    for (i = 0; i < plan->tangent_node_count; ++i) {
        const tangent_node *node = &plan->tangent_nodes[i];
        const size_t group = (size_t)node->level * 4 + node->kind;
        plan->x86_node_offsets[cursor[group]++] = node->offset;
    }

    if (!build_x86_leaves(leaves,
                          plan->n,
                          &leaf_count,
                          0,
                          plan->levels,
                          TANGENT_NORMAL)) {
        free(leaves);
        free(cursor);
        return 0;
    }
    memset(cursor, 0, cursor_count * sizeof(*cursor));
    for (i = 0; i < leaf_count; ++i) {
        const size_t group =
            (size_t)leaves[i].level * 4 + leaves[i].kind;
        ++plan->x86_leaf_count[group];
    }
    offset = 0;
    for (i = 0; i < 5 * 4; ++i) {
        plan->x86_leaf_start[i] = offset;
        cursor[i] = offset;
        offset += plan->x86_leaf_count[i];
    }
    for (i = 0; i < leaf_count; ++i) {
        const size_t group =
            (size_t)leaves[i].level * 4 + leaves[i].kind;
        plan->x86_leaf_offsets[cursor[group]++] = leaves[i].offset;
    }
    free(leaves);
    free(cursor);
    return 1;
}

static int create_tangent_schedule(fft_plan *plan)
{
    uint32_t *temporary;
    size_t i;
    unsigned level;

    plan->tangent_permutation =
        (uint32_t *)malloc(plan->n * sizeof(*plan->tangent_permutation));
    plan->blocked_permutation =
        (uint32_t **)calloc(plan->table_levels,
                            sizeof(*plan->blocked_permutation));
    temporary = (uint32_t *)malloc(plan->n * sizeof(*temporary));
    plan->tangent_base_capacity = plan->n;
    plan->tangent_bases =
        (uint32_t *)malloc(plan->tangent_base_capacity *
                           sizeof(*plan->tangent_bases));
    plan->tangent_node_capacity = plan->n;
    plan->tangent_nodes =
        (tangent_node *)malloc(plan->tangent_node_capacity *
                               sizeof(*plan->tangent_nodes));
    if (plan->tangent_permutation == NULL ||
        plan->blocked_permutation == NULL || temporary == NULL ||
        plan->tangent_bases == NULL || plan->tangent_nodes == NULL) {
        free(temporary);
        return 0;
    }

    for (i = 0; i < plan->n; ++i) {
        plan->tangent_permutation[i] = (uint32_t)i;
    }
    reorder_permutation(plan->tangent_permutation,
                        plan->levels,
                        temporary);

    if (plan->n >= ((size_t)1 << 22)) {
        for (level = plan->levels - 4;
             level <= plan->levels - 2;
             ++level) {
            const size_t count = level_size(level);
            plan->blocked_permutation[level] =
                (uint32_t *)malloc(count * sizeof(uint32_t));
            if (plan->blocked_permutation[level] == NULL) {
                free(temporary);
                return 0;
            }
            for (i = 0; i < count; ++i) {
                plan->blocked_permutation[level][i] = (uint32_t)i;
            }
            reorder_permutation(plan->blocked_permutation[level],
                                level,
                                temporary);
        }
    }
    free(temporary);

    return build_tangent_nodes(plan, 0, plan->levels, TANGENT_NORMAL) &&
           group_tangent_schedule(plan);
}

fft_plan *fft_plan_create(size_t n)
{
    fft_plan *plan;
    const unsigned size_bits = (unsigned)(sizeof(size_t) * 8);

    if (!is_power_of_two(n)) {
        return NULL;
    }

    plan = (fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }

    plan->n = n;
    plan->levels = integer_log2(n);
    plan->tangent_x86_asm_available = tangent_x86_runtime_available();
    plan->lane4_cpu_features = lane4_runtime_features();
    if (plan->levels + 2 >= size_bits) {
        fft_plan_destroy(plan);
        return NULL;
    }
    plan->table_levels = plan->levels + 3;

    plan->scratch = (fft_complex *)aligned_alloc(
        64,
        ((n * sizeof(*plan->scratch) + 63) / 64) * 64);
    if (n <= 131072) {
        plan->ffmpeg = ffmpeg_fft_plan_create(n);
        if (n >= 16) {
            plan->lane4_portable = lane4_portable_plan_create(n);
        }
#if HAVE_TANGENT_X86_ASM
        if (n >= 16 &&
            (plan->lane4_cpu_features & LANE4_CPU_AVX) != 0) {
            plan->lane4_avx = lane4_avx_fft_plan_create(n);
        }
        if (n >= 16 &&
            (plan->lane4_cpu_features &
             (LANE4_CPU_AVX | LANE4_CPU_FMA)) ==
                (LANE4_CPU_AVX | LANE4_CPU_FMA)) {
            plan->lane4_avx_fma = lane4_avx_fma_fft_plan_create(n);
        }
        if (n >= 16 &&
            (plan->lane4_cpu_features & LANE4_CPU_AVX2) != 0) {
            plan->lane4_avx2 = lane4_avx2_fft_plan_create(n);
        }
        if (n >= 16 &&
            (plan->lane4_cpu_features &
             (LANE4_CPU_AVX2 | LANE4_CPU_FMA)) ==
                (LANE4_CPU_AVX2 | LANE4_CPU_FMA)) {
            plan->lane4 = lane4_fft_plan_create(n);
        }
#endif
    }
    if (plan->scratch == NULL ||
        (n <= 131072 && plan->ffmpeg == NULL) ||
        (n >= 16 && n <= 131072 && plan->lane4_portable == NULL) ||
#if HAVE_TANGENT_X86_ASM
        (n >= 16 && n <= 131072 &&
         (plan->lane4_cpu_features & LANE4_CPU_AVX) != 0 &&
         plan->lane4_avx == NULL) ||
        (n >= 16 && n <= 131072 &&
         (plan->lane4_cpu_features &
          (LANE4_CPU_AVX | LANE4_CPU_FMA)) ==
             (LANE4_CPU_AVX | LANE4_CPU_FMA) &&
         plan->lane4_avx_fma == NULL) ||
        (n >= 16 && n <= 131072 &&
         (plan->lane4_cpu_features & LANE4_CPU_AVX2) != 0 &&
         plan->lane4_avx2 == NULL) ||
        (n >= 16 && n <= 131072 &&
         (plan->lane4_cpu_features &
          (LANE4_CPU_AVX2 | LANE4_CPU_FMA)) ==
             (LANE4_CPU_AVX2 | LANE4_CPU_FMA) &&
         plan->lane4 == NULL) ||
#endif
        !create_scale_tables(plan) ||
        !create_transform_tables(plan) ||
        !create_x86_leaf_tables(plan) ||
        !create_tangent_schedule(plan)) {
        fft_plan_destroy(plan);
        return NULL;
    }

    return plan;
}

void fft_plan_destroy(fft_plan *plan)
{
    if (plan == NULL) {
        return;
    }

    free(plan->scratch);
    ffmpeg_fft_plan_destroy(plan->ffmpeg);
    lane4_portable_plan_destroy(plan->lane4_portable);
#if HAVE_TANGENT_X86_ASM
    lane4_fft_plan_destroy(plan->lane4);
    lane4_avx_fft_plan_destroy(plan->lane4_avx);
    lane4_avx_fma_fft_plan_destroy(plan->lane4_avx_fma);
    lane4_avx2_fft_plan_destroy(plan->lane4_avx2);
#endif
    free_float_table(plan->scale, plan->table_levels);
    free_constant_table(plan->root, plan->table_levels);
    free_complex_table(plan->scaled_twiddle, plan->table_levels);
    free_complex_table(plan->tangent_twiddle, plan->table_levels);
    free_float_table(plan->tangent_value, plan->table_levels);
    free_float_table(plan->s2_low, plan->table_levels);
    free_float_table(plan->s2_high, plan->table_levels);
    free_float_table(plan->s4_0, plan->table_levels);
    free_float_table(plan->s4_1, plan->table_levels);
    free_float_table(plan->s4_2, plan->table_levels);
    free_float_table(plan->s4_3, plan->table_levels);
    free_complex_table(plan->x86_s2_low, plan->table_levels);
    free_complex_table(plan->x86_s2_high, plan->table_levels);
    free_complex_table(plan->x86_s4_0, plan->table_levels);
    free_complex_table(plan->x86_s4_1, plan->table_levels);
    free_complex_table(plan->x86_s4_2, plan->table_levels);
    free_complex_table(plan->x86_s4_3, plan->table_levels);
    free(plan->tangent_permutation);
    if (plan->blocked_permutation != NULL) {
        unsigned level;
        for (level = 0; level < plan->table_levels; ++level) {
            free(plan->blocked_permutation[level]);
        }
        free(plan->blocked_permutation);
    }
    free(plan->tangent_bases);
    free(plan->tangent_nodes);
    free(plan->x86_base_offsets);
    free(plan->x86_node_offsets);
    free(plan->x86_node_start);
    free(plan->x86_node_count);
    free(plan->x86_leaf_offsets);
    free(plan->x86_leaf_tables);
    free(plan);
}

size_t fft_plan_size(const fft_plan *plan)
{
    return plan == NULL ? 0 : plan->n;
}

int fft_plan_supports(const fft_plan *plan, fft_algorithm algorithm)
{
    if (plan == NULL) {
        return 0;
    }
    if (algorithm == FFT_FFMPEG) {
        return plan->ffmpeg != NULL;
    }
    if (algorithm == FFT_TANGENT_X86_ASM) {
        return plan->tangent_x86_asm_available;
    }
    if (algorithm == FFT_LANE4_C) {
        return plan->lane4_portable != NULL;
    }
    if (algorithm >= FFT_LANE4_SSE &&
        algorithm <= FFT_LANE4_SSE42) {
#if HAVE_TANGENT_X86_ASM
        const unsigned feature =
            1U << (unsigned)(algorithm - FFT_LANE4_SSE);
        return plan->lane4_portable != NULL &&
               (plan->lane4_cpu_features & feature) != 0;
#else
        return 0;
#endif
    }
    if (algorithm == FFT_LANE4_AVX) {
#if HAVE_TANGENT_X86_ASM
        return plan->lane4_avx != NULL;
#else
        return 0;
#endif
    }
    if (algorithm == FFT_LANE4_AVX_FMA) {
#if HAVE_TANGENT_X86_ASM
        return plan->lane4_avx_fma != NULL;
#else
        return 0;
#endif
    }
    if (algorithm == FFT_LANE4_AVX2) {
#if HAVE_TANGENT_X86_ASM
        return plan->lane4_avx2 != NULL;
#else
        return 0;
#endif
    }
    if (algorithm == FFT_LANE4_AVX2_FMA) {
#if HAVE_TANGENT_X86_ASM
        return plan->lane4 != NULL;
#else
        return 0;
#endif
    }
    return algorithm >= FFT_RADIX2 && algorithm < FFT_ALGORITHM_COUNT;
}

static void radix2_fft(const fft_plan *plan, fft_complex *data)
{
    const size_t n = plan->n;
    size_t i;
    size_t j = 0;
    unsigned level;

    for (i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const fft_complex temporary = data[i];
            data[i] = data[j];
            data[j] = temporary;
        }
    }

    for (level = 1; level <= plan->levels; ++level) {
        const size_t block = level_size(level);
        const size_t half = block / 2;
        size_t start;
        for (start = 0; start < n; start += block) {
            size_t k;
            for (k = 0; k < half; ++k) {
                const fft_complex even = data[start + k];
                const fft_complex odd = multiply_constant(
                    plan->root[level][k], data[start + k + half]);
                data[start + k] = complex_add(even, odd);
                data[start + k + half] = complex_subtract(even, odd);
            }
        }
    }
}

/*
 * Reorders a conjugate-pair split-radix input into:
 *   x[2k], x[4k+1], x[4k-1] (with x[-1] = x[n-1]).
 */
static void gather_conjugate_pair_to(const fft_complex *data,
                                     unsigned level,
                                     fft_complex *scratch)
{
    const size_t n = level_size(level);
    const size_t half = n / 2;
    const size_t quarter = n / 4;
    size_t k;

    for (k = 0; k < half; ++k) {
        scratch[k] = data[2 * k];
    }
    for (k = 0; k < quarter; ++k) {
        scratch[half + k] = data[4 * k + 1];
        scratch[half + quarter + k] =
            data[k == 0 ? n - 1 : 4 * k - 1];
    }
}

static void gather_conjugate_pair(fft_complex *data,
                                  unsigned level,
                                  fft_complex *scratch)
{
    const size_t n = level_size(level);
    gather_conjugate_pair_to(data, level, scratch);
    memcpy(data, scratch, n * sizeof(*data));
}

static void split_radix_recursive(const fft_plan *plan,
                                  fft_complex *data,
                                  unsigned level,
                                  fft_complex *scratch)
{
    const size_t n = level_size(level);
    size_t k;

    if (level == 0) {
        return;
    }
    if (level == 1) {
        const fft_complex a = data[0];
        const fft_complex b = data[1];
        data[0] = complex_add(a, b);
        data[1] = complex_subtract(a, b);
        return;
    }

    {
        const size_t half = n / 2;
        const size_t quarter = n / 4;

        gather_conjugate_pair(data, level, scratch);
        split_radix_recursive(plan, data, level - 1, scratch);
        split_radix_recursive(plan, data + half, level - 2, scratch);
        split_radix_recursive(plan,
                              data + half + quarter,
                              level - 2,
                              scratch);

        for (k = 0; k < quarter; ++k) {
            const fft_complex u0 = data[k];
            const fft_complex u1 = data[k + quarter];
            const fft_complex a =
                multiply_constant(plan->root[level][k], data[half + k]);
            const fft_complex b = multiply_conjugate_constant(
                plan->root[level][k], data[half + quarter + k]);
            fft_complex sum;
            fft_complex difference;

            sum = complex_add(a, b);
            difference = complex_subtract(a, b);

            scratch[k] = complex_add(u0, sum);
            scratch[k + half] = complex_subtract(u0, sum);
            scratch[k + quarter] =
                complex_add(u1, multiply_minus_i(difference));
            scratch[k + 3 * quarter] =
                complex_add(u1, multiply_i(difference));
        }
        memcpy(data, scratch, n * sizeof(*data));
    }
}

static void tangent_base(fft_complex *data, tangent_kind kind)
{
    const fft_complex a = data[0];
    const fft_complex b = data[1];
    data[0] = complex_add(a, b);
    data[1] = complex_subtract(a, b);

    /*
     * For N=2, s_N and s_2N are one. Only the S4 transform has a
     * non-trivial denominator: s_8,1 = 1/sqrt(2).
     */
    if (kind == TANGENT_S4) {
        const float square_root_two = 1.4142135623730950488f;
        data[1] = scale_complex(data[1], square_root_two);
    }
}

static void tangent_combine_normal(const fft_plan *plan,
                                   fft_complex *data,
                                   unsigned level)
{
    const size_t n = level_size(level);
    const size_t half = n / 2;
    const size_t quarter = n / 4;
    const fft_complex *factor = plan->scaled_twiddle[level];
    size_t k;

    for (k = 0; k < quarter; ++k) {
        const fft_complex u0 = data[k];
        const fft_complex u1 = data[k + quarter];
        const fft_complex z = data[half + k];
        const fft_complex zp = data[half + quarter + k];
        const float c = factor[k].re;
        const float s = factor[k].im;
        const fft_complex a = {
            c * z.re - s * z.im,
            s * z.re + c * z.im
        };
        const fft_complex b = {
            c * zp.re + s * zp.im,
            -s * zp.re + c * zp.im
        };
        const fft_complex sum = complex_add(a, b);
        const fft_complex difference = complex_subtract(a, b);

        data[k] = complex_add(u0, sum);
        data[k + quarter] =
            complex_add(u1, multiply_minus_i(difference));
        data[k + half] = complex_subtract(u0, sum);
        data[k + 3 * quarter] =
            complex_add(u1, multiply_i(difference));
    }
}

static void tangent_combine_s_range(const fft_plan *plan,
                                    fft_complex *data,
                                    unsigned level,
                                    size_t begin,
                                    size_t end,
                                    int high_region)
{
    const size_t n = level_size(level);
    const size_t half = n / 2;
    const size_t quarter = n / 4;
    const float *value = plan->tangent_value[level];
    size_t k;

    for (k = begin; k < end; ++k) {
        const fft_complex u0 = data[k];
        const fft_complex u1 = data[k + quarter];
        const fft_complex z = data[half + k];
        const fft_complex zp = data[half + quarter + k];
        const float v = value[k];
        fft_complex a;
        fft_complex b;
        if (!high_region) {
            a = (fft_complex){
                z.re - v * z.im,
                v * z.re + z.im
            };
            b = (fft_complex){
                zp.re + v * zp.im,
                -v * zp.re + zp.im
            };
        } else {
            a = (fft_complex){
                v * z.re + z.im,
                -z.re + v * z.im
            };
            b = (fft_complex){
                v * zp.re - zp.im,
                zp.re + v * zp.im
            };
        }
        const fft_complex sum = complex_add(a, b);
        const fft_complex difference = complex_subtract(a, b);

        data[k] = complex_add(u0, sum);
        data[k + quarter] =
            complex_add(u1, multiply_minus_i(difference));
        data[k + half] = complex_subtract(u0, sum);
        data[k + 3 * quarter] =
            complex_add(u1, multiply_i(difference));
    }
}

static void tangent_combine_s(const fft_plan *plan,
                              fft_complex *data,
                              unsigned level)
{
    const size_t quarter = level_size(level) / 4;
    const size_t middle = quarter / 2;
    tangent_combine_s_range(plan, data, level, 0, middle + 1, 0);
    tangent_combine_s_range(plan, data, level, middle + 1, quarter, 1);
}

static void tangent_combine_s2_range(const fft_plan *plan,
                                     fft_complex *data,
                                     unsigned level,
                                     size_t begin,
                                     size_t end,
                                     int high_region)
{
    const size_t n = level_size(level);
    const size_t half = n / 2;
    const size_t quarter = n / 4;
    const float *value = plan->tangent_value[level];
    const float *low_scale = plan->s2_low[level];
    const float *high_scale = plan->s2_high[level];
    size_t k;

    for (k = begin; k < end; ++k) {
        const fft_complex u0 = data[k];
        const fft_complex u1 = data[k + quarter];
        const fft_complex z = data[half + k];
        const fft_complex zp = data[half + quarter + k];
        const float v = value[k];
        fft_complex a;
        fft_complex b;
        if (!high_region) {
            a = (fft_complex){
                z.re - v * z.im,
                v * z.re + z.im
            };
            b = (fft_complex){
                zp.re + v * zp.im,
                -v * zp.re + zp.im
            };
        } else {
            a = (fft_complex){
                v * z.re + z.im,
                -z.re + v * z.im
            };
            b = (fft_complex){
                v * zp.re - zp.im,
                zp.re + v * zp.im
            };
        }
        const fft_complex low =
            scale_complex(complex_add(a, b), low_scale[k]);
        const fft_complex high =
            scale_complex(complex_subtract(a, b), high_scale[k]);

        data[k] = complex_add(u0, low);
        data[k + quarter] =
            complex_add(u1, multiply_minus_i(high));
        data[k + half] = complex_subtract(u0, low);
        data[k + 3 * quarter] =
            complex_add(u1, multiply_i(high));
    }
}

static void tangent_combine_s2(const fft_plan *plan,
                               fft_complex *data,
                               unsigned level)
{
    const size_t quarter = level_size(level) / 4;
    const size_t middle = quarter / 2;
    tangent_combine_s2_range(plan, data, level, 0, middle + 1, 0);
    tangent_combine_s2_range(plan, data, level, middle + 1, quarter, 1);
}

static void tangent_combine_s4_range(const fft_plan *plan,
                                     fft_complex *data,
                                     unsigned level,
                                     size_t begin,
                                     size_t end,
                                     int high_region)
{
    const size_t n = level_size(level);
    const size_t half = n / 2;
    const size_t quarter = n / 4;
    const float *value = plan->tangent_value[level];
    const float *scale0 = plan->s4_0[level];
    const float *scale1 = plan->s4_1[level];
    const float *scale2 = plan->s4_2[level];
    const float *scale3 = plan->s4_3[level];
    size_t k;

    for (k = begin; k < end; ++k) {
        const fft_complex u0 = data[k];
        const fft_complex u1 = data[k + quarter];
        const fft_complex z = data[half + k];
        const fft_complex zp = data[half + quarter + k];
        const float v = value[k];
        fft_complex a;
        fft_complex b;
        if (!high_region) {
            a = (fft_complex){
                z.re - v * z.im,
                v * z.re + z.im
            };
            b = (fft_complex){
                zp.re + v * zp.im,
                -v * zp.re + zp.im
            };
        } else {
            a = (fft_complex){
                v * z.re + z.im,
                -z.re + v * z.im
            };
            b = (fft_complex){
                v * zp.re - zp.im,
                zp.re + v * zp.im
            };
        }
        const fft_complex sum = complex_add(a, b);
        const fft_complex difference = complex_subtract(a, b);

        data[k] =
            scale_complex(complex_add(u0, sum), scale0[k]);
        data[k + quarter] = scale_complex(
            complex_add(u1, multiply_minus_i(difference)), scale1[k]);
        data[k + half] =
            scale_complex(complex_subtract(u0, sum), scale2[k]);
        data[k + 3 * quarter] = scale_complex(
            complex_add(u1, multiply_i(difference)), scale3[k]);
    }
}

static void tangent_combine_s4(const fft_plan *plan,
                               fft_complex *data,
                               unsigned level)
{
    const size_t quarter = level_size(level) / 4;
    const size_t middle = quarter / 2;
    tangent_combine_s4_range(plan, data, level, 0, middle + 1, 0);
    tangent_combine_s4_range(plan, data, level, middle + 1, quarter, 1);
}

static void tangent_small_fft(const fft_plan *plan, fft_complex *data)
{
    if (plan->n == 1) {
        return;
    }
    if (plan->n == 2) {
        tangent_base(data, TANGENT_NORMAL);
        return;
    }
    if (plan->n == 4) {
        const fft_complex x0 = data[0];
        const fft_complex x1 = data[1];
        const fft_complex x2 = data[2];
        const fft_complex x3 = data[3];
        const fft_complex even_sum = complex_add(x0, x2);
        const fft_complex even_difference = complex_subtract(x0, x2);
        const fft_complex odd_sum = complex_add(x1, x3);
        const fft_complex odd_difference = complex_subtract(x1, x3);

        data[0] = complex_add(even_sum, odd_sum);
        data[1] =
            complex_add(even_difference,
                        multiply_minus_i(odd_difference));
        data[2] = complex_subtract(even_sum, odd_sum);
        data[3] =
            complex_add(even_difference, multiply_i(odd_difference));
        return;
    }

    if (plan->n == 8) {
        fft_complex u[4];
        fft_complex z[2];
        fft_complex zp[2];
        size_t k;

        u[0] = data[0];
        u[1] = data[2];
        u[2] = data[4];
        u[3] = data[6];
        tangent_small_fft((const fft_plan *)&(fft_plan){
                              .n = 4
                          },
                          u);
        z[0] = complex_add(data[1], data[5]);
        z[1] = complex_subtract(data[1], data[5]);
        zp[0] = complex_add(data[7], data[3]);
        zp[1] = complex_subtract(data[7], data[3]);

        for (k = 0; k < 2; ++k) {
            const fft_complex u0 = u[k];
            const fft_complex u1 = u[k + 2];
            const fft_complex a =
                multiply_constant(plan->root[3][k], z[k]);
            const fft_complex b = multiply_conjugate_constant(
                plan->root[3][k], zp[k]);
            const fft_complex sum = complex_add(a, b);
            const fft_complex difference = complex_subtract(a, b);

            data[k] = complex_add(u0, sum);
            data[k + 2] =
                complex_add(u1, multiply_minus_i(difference));
            data[k + 4] = complex_subtract(u0, sum);
            data[k + 6] =
                complex_add(u1, multiply_i(difference));
        }
    }
}

static void permute_local_block(const fft_complex *input,
                                fft_complex *output,
                                const uint32_t *permutation,
                                size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i) {
        output[i] = input[permutation[i]];
    }
}

#if HAVE_TANGENT_X86_ASM
static inline void gather_tangent_leaf_x86(
    const fft_complex *input,
    fft_complex *output,
    const uint32_t *permutation,
    const uint32_t *offsets,
    size_t count,
    const fft_complex *tables,
    unsigned level,
    tangent_kind kind)
{
#define DISPATCH_GATHER(prefix)                                        \
    do {                                                               \
        switch (kind) {                                                \
        case TANGENT_NORMAL:                                           \
            prefix##_n(input, output, permutation, offsets, count, tables); \
            return;                                                    \
        case TANGENT_S:                                                \
            prefix##_s(input, output, permutation, offsets, count, tables); \
            return;                                                    \
        case TANGENT_S2:                                               \
            prefix##_s2(input, output, permutation, offsets, count, tables); \
            return;                                                    \
        case TANGENT_S4:                                               \
            prefix##_s4(input, output, permutation, offsets, count, tables); \
            return;                                                    \
        }                                                              \
    } while (0)

    if (level == 2) {
        DISPATCH_GATHER(tangent_x86_gather_leaf2);
    } else if (level == 3) {
        DISPATCH_GATHER(tangent_x86_gather_leaf3);
    } else {
        DISPATCH_GATHER(tangent_x86_gather_leaf4);
    }
#undef DISPATCH_GATHER
}
#endif

static void tangent_fft_scheduled(const fft_plan *plan,
                                  fft_complex *data,
                                  int use_x86_asm)
{
    fft_complex *work;
    int output_is_scratch = 1;
    size_t i;
#if HAVE_TANGENT_X86_ASM
    unsigned level;
    unsigned kind;
    const int fuse_x86_gather =
        use_x86_asm && plan->levels >= 2 &&
        plan->n < ((size_t)1 << 18);
#endif
#ifdef TANGENT_PROFILE
    struct timespec profile_time[5];
    static double profile_level_totals[32];
    const int profile_enabled =
        use_x86_asm && getenv("TANGENT_PROFILE") != NULL;
    if (profile_enabled) {
        timespec_get(&profile_time[0], TIME_UTC);
    }
#endif

    if (plan->n <= 8) {
        tangent_small_fft(plan, data);
        return;
    }

    if (plan->n >= ((size_t)1 << 22)) {
        const size_t half = plan->n / 2;
        const size_t quarter = plan->n / 4;
        const size_t eighth = plan->n / 8;
        const size_t sixteenth = plan->n / 16;
        const size_t third_child = half + quarter;
        const unsigned level = plan->levels;

        /*
         * Two regular gather levels keep each remaining random permutation
         * below the last-level-cache cliff.
         */
        gather_conjugate_pair_to(data, level, plan->scratch);
        gather_conjugate_pair_to(plan->scratch,
                                 level - 1,
                                 data);
        gather_conjugate_pair_to(plan->scratch + half,
                                 level - 2,
                                 data + half);
        gather_conjugate_pair_to(plan->scratch + third_child,
                                 level - 2,
                                 data + third_child);

        permute_local_block(data,
                            plan->scratch,
                            plan->blocked_permutation[level - 2],
                            quarter);
        permute_local_block(data + quarter,
                            plan->scratch + quarter,
                            plan->blocked_permutation[level - 3],
                            eighth);
        permute_local_block(data + quarter + eighth,
                            plan->scratch + quarter + eighth,
                            plan->blocked_permutation[level - 3],
                            eighth);

        permute_local_block(data + half,
                            plan->scratch + half,
                            plan->blocked_permutation[level - 3],
                            eighth);
        permute_local_block(data + half + eighth,
                            plan->scratch + half + eighth,
                            plan->blocked_permutation[level - 4],
                            sixteenth);
        permute_local_block(data + half + eighth + sixteenth,
                            plan->scratch + half + eighth + sixteenth,
                            plan->blocked_permutation[level - 4],
                            sixteenth);

        permute_local_block(data + third_child,
                            plan->scratch + third_child,
                            plan->blocked_permutation[level - 3],
                            eighth);
        permute_local_block(data + third_child + eighth,
                            plan->scratch + third_child + eighth,
                            plan->blocked_permutation[level - 4],
                            sixteenth);
        permute_local_block(data + third_child + eighth + sixteenth,
                            plan->scratch + third_child + eighth + sixteenth,
                            plan->blocked_permutation[level - 4],
                            sixteenth);
        work = plan->scratch;
    } else if (plan->n >= ((size_t)1 << 18)) {
        const size_t half = plan->n / 2;
        const size_t quarter = plan->n / 4;

        /*
         * One regular top-level gather breaks the global random permutation
         * into three cache-sized child permutations. Execute directly in the
         * caller's array afterward, which also removes the final full copy.
         */
        gather_conjugate_pair_to(data, plan->levels, plan->scratch);
        for (i = 0; i < half; ++i) {
            const size_t original = plan->tangent_permutation[i];
            data[i] = plan->scratch[original >> 1];
        }
        for (i = 0; i < quarter; ++i) {
            const size_t original =
                plan->tangent_permutation[half + i];
            data[half + i] =
                plan->scratch[half + ((original - 1) >> 2)];
        }
        for (i = 0; i < quarter; ++i) {
            const size_t original =
                plan->tangent_permutation[half + quarter + i];
            const size_t local =
                original == plan->n - 1 ? 0 : (original + 1) >> 2;
            data[half + quarter + i] =
                plan->scratch[half + quarter + local];
        }
        work = data;
        output_is_scratch = 0;
    } else {
#if HAVE_TANGENT_X86_ASM
        if (fuse_x86_gather) {
            work = plan->scratch;
        } else
#endif
        {
            for (i = 0; i < plan->n; ++i) {
                plan->scratch[i] = data[plan->tangent_permutation[i]];
            }
            work = plan->scratch;
        }
    }
#ifdef TANGENT_PROFILE
    if (profile_enabled) {
        timespec_get(&profile_time[1], TIME_UTC);
    }
#endif

#if HAVE_TANGENT_X86_ASM
    if (use_x86_asm) {
        if (fuse_x86_gather && plan->n == 32) {
            tangent_x86_gather_fft32_normal(
                data,
                work,
                plan->tangent_permutation,
                plan->x86_leaf_tables,
                plan->scaled_twiddle[5]);
            goto tangent_x86_nodes_done;
        }
        if (fuse_x86_gather && plan->n == 64) {
            tangent_x86_gather_fft64_normal(
                data,
                work,
                plan->tangent_permutation,
                plan->x86_leaf_tables,
                plan->scaled_twiddle[5],
                plan->scaled_twiddle[6]);
            goto tangent_x86_nodes_done;
        }
        if (plan->levels == 1) {
            for (kind = TANGENT_NORMAL; kind <= TANGENT_S2; ++kind) {
                const size_t begin = plan->x86_base_start[kind];
                const size_t count = plan->x86_base_count[kind];
                if (count != 0) {
                    tangent_x86_batch_base(
                        work, plan->x86_base_offsets + begin, count);
                }
            }
            {
                const size_t begin = plan->x86_base_start[TANGENT_S4];
                const size_t count = plan->x86_base_count[TANGENT_S4];
                if (count != 0) {
                    tangent_x86_batch_base_s4(
                        work, plan->x86_base_offsets + begin, count);
                }
            }
        } else if (plan->levels >= 2) {
            for (level = 2;
                 level <= 4 && level <= plan->levels;
                 ++level) {
                for (kind = 0; kind < 4; ++kind) {
                    const size_t group = (size_t)level * 4 + kind;
                    const size_t begin = plan->x86_leaf_start[group];
                    const size_t count = plan->x86_leaf_count[group];
                    if (count == 0) {
                        continue;
                    }
                    if (fuse_x86_gather) {
                        gather_tangent_leaf_x86(
                            data,
                            work,
                            plan->tangent_permutation,
                            plan->x86_leaf_offsets + begin,
                            count,
                            plan->x86_leaf_tables,
                            level,
                            (tangent_kind)kind);
                    } else if (level == 2) {
                        tangent_x86_batch_leaf2(
                            work,
                            plan->x86_leaf_offsets + begin,
                            count,
                            plan->x86_leaf_tables,
                            kind);
                    } else if (level == 3) {
                        tangent_x86_batch_leaf3(
                            work,
                            plan->x86_leaf_offsets + begin,
                            count,
                            plan->x86_leaf_tables,
                            kind);
                    } else {
                        tangent_x86_batch_leaf4(
                            work,
                            plan->x86_leaf_offsets + begin,
                            count,
                            plan->x86_leaf_tables,
                            kind);
                    }
                }
            }
        }
#ifdef TANGENT_PROFILE
        if (profile_enabled) {
            timespec_get(&profile_time[2], TIME_UTC);
        }
#endif

        for (level = 5; level <= plan->levels; ++level) {
#ifdef TANGENT_PROFILE
            struct timespec level_start;
            struct timespec level_end;
            if (profile_enabled) {
                timespec_get(&level_start, TIME_UTC);
            }
#endif
            if (level <= 4) {
                size_t group =
                    (size_t)level * 4 + TANGENT_NORMAL;
                size_t begin = plan->x86_node_start[group];
                size_t count = plan->x86_node_count[group];
                if (count != 0) {
                    if (level == 2) {
                        tangent_x86_batch_unscaled_q1(
                            work, plan->x86_node_offsets + begin, count);
                    } else if (level == 3) {
                        tangent_x86_batch_unscaled_q2(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->scaled_twiddle[level]);
                    } else {
                        tangent_x86_batch_unscaled_q4(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->scaled_twiddle[level]);
                    }
                }

                group = (size_t)level * 4 + TANGENT_S;
                begin = plan->x86_node_start[group];
                count = plan->x86_node_count[group];
                if (count != 0) {
                    if (level == 2) {
                        tangent_x86_batch_unscaled_q1(
                            work, plan->x86_node_offsets + begin, count);
                    } else if (level == 3) {
                        tangent_x86_batch_unscaled_q2(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level]);
                    } else {
                        tangent_x86_batch_unscaled_q4(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level]);
                    }
                }

                group = (size_t)level * 4 + TANGENT_S2;
                begin = plan->x86_node_start[group];
                count = plan->x86_node_count[group];
                if (count != 0) {
                    if (level == 2) {
                        tangent_x86_batch_s2_q1(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->s2_low[level],
                            plan->s2_high[level]);
                    } else if (level == 3) {
                        tangent_x86_batch_s2_q2(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level],
                            plan->s2_low[level],
                            plan->s2_high[level]);
                    } else {
                        tangent_x86_batch_s2_q4(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level],
                            plan->s2_low[level],
                            plan->s2_high[level]);
                    }
                }

                group = (size_t)level * 4 + TANGENT_S4;
                begin = plan->x86_node_start[group];
                count = plan->x86_node_count[group];
                if (count != 0) {
                    if (level == 2) {
                        tangent_x86_batch_s4_q1(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->s4_0[level],
                            plan->s4_1[level],
                            plan->s4_2[level],
                            plan->s4_3[level]);
                    } else if (level == 3) {
                        tangent_x86_batch_s4_q2(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level],
                            plan->s4_0[level],
                            plan->s4_1[level],
                            plan->s4_2[level],
                            plan->s4_3[level]);
                    } else {
                        tangent_x86_batch_s4_q4(
                            work,
                            plan->x86_node_offsets + begin,
                            count,
                            plan->tangent_twiddle[level],
                            plan->s4_0[level],
                            plan->s4_1[level],
                            plan->s4_2[level],
                            plan->s4_3[level]);
                    }
                }
#ifdef TANGENT_PROFILE
                if (profile_enabled) {
                    timespec_get(&level_end, TIME_UTC);
                    profile_level_totals[level] +=
                        (double)(level_end.tv_sec - level_start.tv_sec) *
                            1.0e9 +
                        (double)(level_end.tv_nsec - level_start.tv_nsec);
                }
#endif
                continue;
            }

            size_t group = (size_t)level * 4 + TANGENT_NORMAL;
            size_t begin = plan->x86_node_start[group];
            size_t count = plan->x86_node_count[group];
            const size_t quarter = level_size(level) / 4;
            if (count != 0) {
                tangent_x86_batch_unscaled_qn(
                    work,
                    plan->x86_node_offsets + begin,
                    count,
                    quarter,
                    plan->scaled_twiddle[level]);
            }

            group = (size_t)level * 4 + TANGENT_S;
            begin = plan->x86_node_start[group];
            count = plan->x86_node_count[group];
            if (count != 0) {
                tangent_x86_batch_tangent_qn(
                    work,
                    plan->x86_node_offsets + begin,
                    count,
                    quarter,
                    plan->tangent_twiddle[level]);
            }

            group = (size_t)level * 4 + TANGENT_S2;
            begin = plan->x86_node_start[group];
            count = plan->x86_node_count[group];
            if (count != 0) {
                tangent_x86_batch_s2_qn(
                    work,
                    plan->x86_node_offsets + begin,
                    count,
                    quarter,
                    plan->tangent_twiddle[level],
                    plan->x86_s2_low[level],
                    plan->x86_s2_high[level]);
            }

            group = (size_t)level * 4 + TANGENT_S4;
            begin = plan->x86_node_start[group];
            count = plan->x86_node_count[group];
            if (count != 0) {
                tangent_x86_batch_s4_qn(
                    work,
                    plan->x86_node_offsets + begin,
                    count,
                    quarter,
                    plan->tangent_twiddle[level],
                    plan->x86_s4_0[level],
                    plan->x86_s4_1[level],
                    plan->x86_s4_2[level],
                    plan->x86_s4_3[level]);
            }
#ifdef TANGENT_PROFILE
            if (profile_enabled) {
                timespec_get(&level_end, TIME_UTC);
                profile_level_totals[level] +=
                    (double)(level_end.tv_sec - level_start.tv_sec) * 1.0e9 +
                    (double)(level_end.tv_nsec - level_start.tv_nsec);
            }
#endif
        }
tangent_x86_nodes_done:
        (void)0;
#ifdef TANGENT_PROFILE
        if (profile_enabled) {
            timespec_get(&profile_time[3], TIME_UTC);
        }
#endif
    } else
#else
    (void)use_x86_asm;
#endif
    {
        for (i = 0; i < plan->tangent_base_count; ++i) {
            const uint32_t encoded = plan->tangent_bases[i];
            const size_t offset = encoded & UINT32_C(0x3fffffff);
            const tangent_kind base_kind =
                (tangent_kind)(encoded >> 30);
            tangent_base(work + offset, base_kind);
        }

        for (i = 0; i < plan->tangent_node_count; ++i) {
            const tangent_node *node = &plan->tangent_nodes[i];
            fft_complex *node_data = work + node->offset;
            switch ((tangent_kind)node->kind) {
            case TANGENT_NORMAL:
                tangent_combine_normal(plan, node_data, node->level);
                break;
            case TANGENT_S:
                tangent_combine_s(plan, node_data, node->level);
                break;
            case TANGENT_S2:
                tangent_combine_s2(plan, node_data, node->level);
                break;
            case TANGENT_S4:
                tangent_combine_s4(plan, node_data, node->level);
                break;
            }
        }
    }

    if (output_is_scratch) {
        memcpy(data, plan->scratch, plan->n * sizeof(*data));
    }
#ifdef TANGENT_PROFILE
    if (profile_enabled) {
        static double totals[4];
        static unsigned calls;
        unsigned phase;
        timespec_get(&profile_time[4], TIME_UTC);
        for (phase = 0; phase < 4; ++phase) {
            totals[phase] +=
                (double)(profile_time[phase + 1].tv_sec -
                         profile_time[phase].tv_sec) *
                    1.0e9 +
                (double)(profile_time[phase + 1].tv_nsec -
                         profile_time[phase].tv_nsec);
        }
        if (++calls == 500) {
            unsigned profile_level;
            fprintf(stderr,
                    "tangent profile ns: permutation %.0f, bases %.0f, "
                    "nodes %.0f, copy %.0f\n",
                    totals[0] / calls,
                    totals[1] / calls,
                    totals[2] / calls,
                    totals[3] / calls);
            for (profile_level = 2;
                 profile_level <= plan->levels;
                 ++profile_level) {
                fprintf(stderr,
                        "  level %u: %.0f ns\n",
                        profile_level,
                        profile_level_totals[profile_level] / calls);
            }
        }
    }
#endif
}

int fft_execute(fft_plan *plan, fft_algorithm algorithm, fft_complex *data)
{
    if (plan == NULL || data == NULL) {
        return -1;
    }
    if (algorithm >= FFT_LANE4_C &&
        algorithm <= FFT_LANE4_AVX2_FMA &&
        !fft_plan_supports(plan, algorithm)) {
        return -1;
    }

    switch (algorithm) {
    case FFT_RADIX2:
        radix2_fft(plan, data);
        return 0;
    case FFT_SPLIT_RADIX:
        split_radix_recursive(plan,
                              data,
                              plan->levels,
                              plan->scratch);
        return 0;
    case FFT_TANGENT:
        tangent_fft_scheduled(plan, data, 0);
        return 0;
    case FFT_TANGENT_X86_ASM:
        if (!plan->tangent_x86_asm_available) {
            return -1;
        }
        tangent_fft_scheduled(plan, data, 1);
        return 0;
    case FFT_LANE4_C:
        return lane4_c_execute(plan->lane4_portable, data);
#if HAVE_TANGENT_X86_ASM
    case FFT_LANE4_SSE:
        return lane4_sse_execute(plan->lane4_portable, data);
    case FFT_LANE4_SSE2:
        return lane4_sse2_execute(plan->lane4_portable, data);
    case FFT_LANE4_SSE3:
        return lane4_sse3_execute(plan->lane4_portable, data);
    case FFT_LANE4_SSSE3:
        return lane4_ssse3_execute(plan->lane4_portable, data);
    case FFT_LANE4_SSE41:
        return lane4_sse41_execute(plan->lane4_portable, data);
    case FFT_LANE4_SSE42:
        return lane4_sse42_execute(plan->lane4_portable, data);
    case FFT_LANE4_AVX:
        return lane4_avx_fft_execute(plan->lane4_avx, data);
    case FFT_LANE4_AVX_FMA:
        return lane4_avx_fma_fft_execute(plan->lane4_avx_fma, data);
    case FFT_LANE4_AVX2:
        return lane4_avx2_fft_execute(plan->lane4_avx2, data);
    case FFT_LANE4_AVX2_FMA:
        return lane4_fft_execute(plan->lane4, data);
#else
    case FFT_LANE4_SSE:
    case FFT_LANE4_SSE2:
    case FFT_LANE4_SSE3:
    case FFT_LANE4_SSSE3:
    case FFT_LANE4_SSE41:
    case FFT_LANE4_SSE42:
    case FFT_LANE4_AVX:
    case FFT_LANE4_AVX_FMA:
    case FFT_LANE4_AVX2:
    case FFT_LANE4_AVX2_FMA:
        return -1;
#endif
    case FFT_FFMPEG:
        return plan->ffmpeg == NULL
                   ? -1
                   : ffmpeg_fft_execute(plan->ffmpeg, data);
    default:
        return -1;
    }
}

const char *fft_algorithm_name(fft_algorithm algorithm)
{
    switch (algorithm) {
    case FFT_RADIX2:
        return "radix-2";
    case FFT_SPLIT_RADIX:
        return "split-radix";
    case FFT_TANGENT:
        return "tangent";
    case FFT_TANGENT_X86_ASM:
        return "tangent-x86-asm";
    case FFT_LANE4_C:
        return "lane4-c";
    case FFT_LANE4_SSE:
        return "lane4-sse";
    case FFT_LANE4_SSE2:
        return "lane4-sse2";
    case FFT_LANE4_SSE3:
        return "lane4-sse3";
    case FFT_LANE4_SSSE3:
        return "lane4-ssse3";
    case FFT_LANE4_SSE41:
        return "lane4-sse4.1";
    case FFT_LANE4_SSE42:
        return "lane4-sse4.2";
    case FFT_LANE4_AVX:
        return "lane4-avx";
    case FFT_LANE4_AVX_FMA:
        return "lane4-avx-fma";
    case FFT_LANE4_AVX2:
        return "lane4-avx2";
    case FFT_LANE4_AVX2_FMA:
        return "lane4-avx2-fma";
    case FFT_FFMPEG:
        return "ffmpeg-avtx";
    default:
        return "unknown";
    }
}

uint64_t fft_theoretical_flops(fft_algorithm algorithm, size_t n)
{
    const unsigned log_n = integer_log2(n);
    int64_t result;

    if (algorithm == FFT_FFMPEG ||
        (algorithm >= FFT_LANE4_C &&
         algorithm <= FFT_LANE4_AVX2_FMA)) {
        return 0;
    }
    if (!is_power_of_two(n) || n == 1) {
        return 0;
    }
    if (n == 2) {
        return 4;
    }

    switch (algorithm) {
    case FFT_RADIX2:
        result = 5 * (int64_t)n * log_n - 10 * (int64_t)n + 16;
        break;
    case FFT_SPLIT_RADIX:
        result = 4 * (int64_t)n * log_n - 6 * (int64_t)n + 8;
        break;
    case FFT_TANGENT:
    case FFT_TANGENT_X86_ASM: {
        const int parity = (log_n & 1U) != 0 ? -1 : 1;
        const int64_t numerator =
            102 * (int64_t)n * log_n - 124 * (int64_t)n -
            54 * (int64_t)log_n - 6 * parity * (int64_t)log_n +
            16 * parity + 216;
        result = numerator / 27;
        break;
    }
    case FFT_LANE4_C:
    case FFT_LANE4_SSE:
    case FFT_LANE4_SSE2:
    case FFT_LANE4_SSE3:
    case FFT_LANE4_SSSE3:
    case FFT_LANE4_SSE41:
    case FFT_LANE4_SSE42:
    case FFT_LANE4_AVX:
    case FFT_LANE4_AVX_FMA:
    case FFT_LANE4_AVX2:
    case FFT_LANE4_AVX2_FMA:
    case FFT_FFMPEG:
        return 0;
    default:
        return 0;
    }

    return (uint64_t)result;
}
