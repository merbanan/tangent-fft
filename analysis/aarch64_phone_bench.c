/*
 * Freestanding AArch64 benchmark for Linux/Android.
 *
 * It links the production lane2/lane4 assembly and FFmpeg's exact NEON
 * assembly without libc, so the resulting static ELF can run from
 * /data/local/tmp on an Android arm64 device.  Timing uses CNTVCT_EL0.
 */

#include "lane2_neon.h"
#include "lane4_portable.h"

#include <stddef.h>
#include <stdint.h>

enum {
    BENCH_MIN_N = 16,
    BENCH_MAX_N = 8192,
    SAMPLE_COUNT = 31
};

#define BENCH_PI 3.141592653589793238462643383279502884

typedef struct {
    int len;
    int inv;
    int *map;
} test_avtx_context;

#define DEFINE_TX_TABLE(length) \
    _Alignas(32) float ff_tx_tab_##length##_float[(length) / 4 + 1]

DEFINE_TX_TABLE(32);
DEFINE_TX_TABLE(64);
DEFINE_TX_TABLE(128);
DEFINE_TX_TABLE(256);
DEFINE_TX_TABLE(512);
DEFINE_TX_TABLE(1024);
DEFINE_TX_TABLE(2048);
DEFINE_TX_TABLE(4096);
DEFINE_TX_TABLE(8192);
DEFINE_TX_TABLE(16384);
DEFINE_TX_TABLE(32768);
DEFINE_TX_TABLE(65536);
DEFINE_TX_TABLE(131072);

#undef DEFINE_TX_TABLE

void ff_tx_fft_sr_ns_float_neon(test_avtx_context *context,
                                void *output,
                                void *input,
                                ptrdiff_t stride);
void ff_tx_fft_sr_float_neon(test_avtx_context *context,
                             void *output,
                             void *input,
                             ptrdiff_t stride);
void ff_tx_fft16_ns_float_neon(test_avtx_context *context,
                              void *output,
                              void *input,
                              ptrdiff_t stride);
void ff_tx_fft16_float_neon(test_avtx_context *context,
                           void *output,
                           void *input,
                           ptrdiff_t stride);
void ff_tx_fft32_ns_float_neon(test_avtx_context *context,
                              void *output,
                              void *input,
                              ptrdiff_t stride);
void ff_tx_fft32_float_neon(test_avtx_context *context,
                           void *output,
                           void *input,
                           ptrdiff_t stride);

static _Alignas(64) unsigned char arena[2U << 20];
static size_t arena_used;
static _Alignas(64) fft_complex lane2_data[BENCH_MAX_N];
static _Alignas(64) fft_complex lane4_data[BENCH_MAX_N];
static _Alignas(64) fft_complex ffmpeg_input[BENCH_MAX_N];
static _Alignas(64) fft_complex ffmpeg_permuted[BENCH_MAX_N];
static _Alignas(64) fft_complex ffmpeg_output[BENCH_MAX_N];
static int split_radix_map[BENCH_MAX_N];

static void *arena_allocate(size_t size, size_t alignment)
{
    const size_t begin =
        (arena_used + alignment - 1) & ~(alignment - 1);
    unsigned char *result;

    if (begin > sizeof(arena) || size > sizeof(arena) - begin) {
        return NULL;
    }
    result = arena + begin;
    arena_used = begin + size;
    return result;
}

void *malloc(size_t size)
{
    return arena_allocate(size, 16);
}

void *calloc(size_t count, size_t size)
{
    unsigned char *result;
    size_t bytes;
    size_t i;

    if (count != 0 && size > (size_t)-1 / count) {
        return NULL;
    }
    bytes = count * size;
    result = (unsigned char *)arena_allocate(bytes, 16);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < bytes; ++i) {
        result[i] = 0;
    }
    return result;
}

void *aligned_alloc(size_t alignment, size_t size)
{
    return arena_allocate(size, alignment);
}

void free(void *pointer)
{
    (void)pointer;
}

static double reduce_angle(double value)
{
    const double period = 2.0 * BENCH_PI;

    while (value > BENCH_PI) {
        value -= period;
    }
    while (value < -BENCH_PI) {
        value += period;
    }
    return value;
}

static double test_sin(double value)
{
    const double x = reduce_angle(value);
    const double x2 = x * x;
    double term = x;
    double result = x;
    unsigned k;

    for (k = 1; k <= 10; ++k) {
        term *= -x2 / (double)((2U * k) * (2U * k + 1U));
        result += term;
    }
    return result;
}

static double test_cos(double value)
{
    const double x = reduce_angle(value);
    const double x2 = x * x;
    double term = 1.0;
    double result = 1.0;
    unsigned k;

    for (k = 1; k <= 10; ++k) {
        term *= -x2 / (double)((2U * k - 1U) * (2U * k));
        result += term;
    }
    return result;
}

float sinf(float value)
{
    return (float)test_sin((double)value);
}

float cosf(float value)
{
    return (float)test_cos((double)value);
}

static int split_radix_permutation(int index, int length, int inverse)
{
    length >>= 1;
    if (length <= 1) {
        return index & 1;
    }
    if ((index & length) == 0) {
        return split_radix_permutation(index, length, inverse) * 2;
    }
    length >>= 1;
    return split_radix_permutation(index, length, inverse) * 4 +
           1 - 2 * (((index & length) == 0) ^ inverse);
}

static void parity_map_generator(int *map,
                                 int n,
                                 int inverse,
                                 int offset,
                                 int length,
                                 int basis)
{
    int i;

    length >>= 1;
    if (length <= basis) {
        int even_index = offset;
        int odd_index = even_index + length;

        for (i = 0; i < length; ++i) {
            const int k1 =
                -split_radix_permutation(offset + i * 2, n, inverse) &
                (n - 1);
            const int k2 =
                -split_radix_permutation(offset + i * 2 + 1, n, inverse) &
                (n - 1);

            map[even_index++] = k1;
            map[odd_index++] = k2;
        }
        return;
    }
    parity_map_generator(map, n, inverse, offset, length, basis);
    parity_map_generator(map, n, inverse, offset + length,
                         length >> 1, basis);
    parity_map_generator(map, n, inverse,
                         offset + length + (length >> 1),
                         length >> 1, basis);
}

static void initialize_table(float *table, int n)
{
    int i;

    for (i = 0; i < n / 4; ++i) {
        table[i] =
            (float)test_cos(2.0 * BENCH_PI * (double)i / (double)n);
    }
    table[n / 4] = 0.0f;
}

static void initialize_tables(int n)
{
    initialize_table(ff_tx_tab_32_float, 32);
    if (n >= 64) {
        initialize_table(ff_tx_tab_64_float, 64);
    }
    if (n >= 128) {
        initialize_table(ff_tx_tab_128_float, 128);
    }
    if (n >= 256) {
        initialize_table(ff_tx_tab_256_float, 256);
    }
    if (n >= 512) {
        initialize_table(ff_tx_tab_512_float, 512);
    }
    if (n >= 1024) {
        initialize_table(ff_tx_tab_1024_float, 1024);
    }
    if (n >= 2048) {
        initialize_table(ff_tx_tab_2048_float, 2048);
    }
    if (n >= 4096) {
        initialize_table(ff_tx_tab_4096_float, 4096);
    }
    if (n >= 8192) {
        initialize_table(ff_tx_tab_8192_float, 8192);
    }
}

static void initialize_ffmpeg(int n)
{
    int i;

    parity_map_generator(split_radix_map, n, 0, 0, n, 4);
    initialize_tables(n);
    for (i = 0; i < n; ++i) {
        ffmpeg_permuted[i] = ffmpeg_input[split_radix_map[i]];
    }
}

static uint64_t timer_frequency(void)
{
    uint64_t result;

    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(result));
    return result;
}

static uint64_t timer_read(void)
{
    uint64_t result;

    __asm__ volatile("isb\n\tmrs %0, cntvct_el0"
                     : "=r"(result)
                     :
                     : "memory");
    return result;
}

static long linux_write(int descriptor,
                        const void *buffer,
                        size_t length)
{
    register long x0 __asm__("x0") = descriptor;
    register const void *x1 __asm__("x1") = buffer;
    register size_t x2 __asm__("x2") = length;
    register long x8 __asm__("x8") = 64;

    __asm__ volatile("svc 0"
                     : "+r"(x0)
                     : "r"(x1), "r"(x2), "r"(x8)
                     : "memory");
    return x0;
}

static void linux_exit(int status)
{
    register long x0 __asm__("x0") = status;
    register long x8 __asm__("x8") = 93;

    __asm__ volatile("svc 0"
                     :
                     : "r"(x0), "r"(x8)
                     : "memory");
    for (;;) {
    }
}

static size_t append_text(char *output, size_t used, const char *text)
{
    while (*text != '\0') {
        output[used++] = *text++;
    }
    return used;
}

static size_t append_u64(char *output, size_t used, uint64_t value)
{
    char reverse[32];
    size_t count = 0;

    do {
        reverse[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    while (count != 0) {
        output[used++] = reverse[--count];
    }
    return used;
}

static void print_result(size_t n,
                         const char *name,
                         uint64_t median_ticks,
                         uint64_t minimum_ticks,
                         uint64_t frequency,
                         size_t iterations)
{
    char output[192];
    const uint64_t median_ns =
        ((median_ticks * 1000000000ULL) / frequency) / iterations;
    const uint64_t minimum_ns =
        ((minimum_ticks * 1000000000ULL) / frequency) / iterations;
    size_t used = 0;

    used = append_u64(output, used, n);
    output[used++] = ' ';
    used = append_text(output, used, name);
    output[used++] = ' ';
    used = append_u64(output, used, median_ns);
    output[used++] = ' ';
    used = append_u64(output, used, minimum_ns);
    output[used++] = ' ';
    used = append_u64(output, used, iterations);
    output[used++] = '\n';
    (void)linux_write(1, output, used);
}

static void sort_samples(uint64_t *samples)
{
    size_t i;

    for (i = 1; i < SAMPLE_COUNT; ++i) {
        const uint64_t value = samples[i];
        size_t position = i;

        while (position != 0 && samples[position - 1] > value) {
            samples[position] = samples[position - 1];
            --position;
        }
        samples[position] = value;
    }
}

typedef enum {
    BENCH_LANE2,
    BENCH_LANE4,
    BENCH_FFMPEG_NATURAL,
    BENCH_FFMPEG_PREPERMUTED
} benchmark_kind;

static void run_ffmpeg(test_avtx_context *context,
                       int n,
                       int natural)
{
    if (n == 16) {
        if (natural) {
            ff_tx_fft16_float_neon(
                context, ffmpeg_output, ffmpeg_input, 8);
        } else {
            ff_tx_fft16_ns_float_neon(
                context, ffmpeg_output, ffmpeg_permuted, 8);
        }
    } else if (n == 32) {
        if (natural) {
            ff_tx_fft32_float_neon(
                context, ffmpeg_output, ffmpeg_input, 8);
        } else {
            ff_tx_fft32_ns_float_neon(
                context, ffmpeg_output, ffmpeg_permuted, 8);
        }
    } else if (natural) {
        ff_tx_fft_sr_float_neon(
            context, ffmpeg_output, ffmpeg_input, 8);
    } else {
        ff_tx_fft_sr_ns_float_neon(
            context, ffmpeg_output, ffmpeg_permuted, 8);
    }
}

static void measure(size_t n,
                    benchmark_kind kind,
                    lane2_neon_plan *lane2,
                    lane4_portable_plan *lane4,
                    test_avtx_context *context,
                    uint64_t frequency,
                    size_t iterations)
{
    uint64_t samples[SAMPLE_COUNT];
    const char *name;
    size_t sample;

    for (sample = 0; sample < 64; ++sample) {
        if (kind == BENCH_LANE2) {
            (void)lane2_neon_execute(lane2, lane2_data);
        } else if (kind == BENCH_LANE4) {
            (void)lane4_neon_execute(lane4, lane4_data);
        } else {
            run_ffmpeg(context, (int)n,
                       kind == BENCH_FFMPEG_NATURAL);
        }
    }
    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        uint64_t start;

        start = timer_read();
        if (kind == BENCH_LANE2) {
            for (iteration = 0; iteration < iterations; ++iteration) {
                (void)lane2_neon_execute(lane2, lane2_data);
            }
            name = "lane2-neon";
        } else if (kind == BENCH_LANE4) {
            for (iteration = 0; iteration < iterations; ++iteration) {
                (void)lane4_neon_execute(lane4, lane4_data);
            }
            name = "lane4-neon-fused";
        } else if (kind == BENCH_FFMPEG_NATURAL) {
            for (iteration = 0; iteration < iterations; ++iteration) {
                run_ffmpeg(context, (int)n, 1);
            }
            name = "ffmpeg-neon-natural";
        } else {
            for (iteration = 0; iteration < iterations; ++iteration) {
                run_ffmpeg(context, (int)n, 0);
            }
            name = "ffmpeg-neon-prepermuted";
        }
        samples[sample] = timer_read() - start;
    }
    sort_samples(samples);
    print_result(n, name,
                 samples[SAMPLE_COUNT / 2], samples[0],
                 frequency, iterations);
}

void _start(void)
{
    static const char header[] =
        "AArch64 FFT benchmark: N algorithm median_ns minimum_ns iterations\n";
    static const char allocation_failure[] =
        "FAIL: planner arena exhausted\n";
    const uint64_t frequency = timer_frequency();
    size_t n;

    (void)linux_write(1, header, sizeof(header) - 1);
    for (n = BENCH_MIN_N; n <= BENCH_MAX_N; n *= 2) {
        lane2_neon_plan *lane2;
        lane4_portable_plan *lane4;
        test_avtx_context context;
        size_t iterations = 8388608 / n;

        if (iterations < 512) {
            iterations = 512;
        }
        arena_used = 0;
        lane2 = lane2_neon_plan_create(n);
        lane4 = lane4_portable_plan_create(n);
        if (lane2 == NULL || lane4 == NULL) {
            (void)linux_write(
                2, allocation_failure, sizeof(allocation_failure) - 1);
            linux_exit(1);
        }
        initialize_ffmpeg((int)n);
        context.len = (int)n;
        context.inv = 0;
        context.map = split_radix_map;

        measure(n, BENCH_LANE2, lane2, lane4,
                &context, frequency, iterations);
        measure(n, BENCH_LANE4, lane2, lane4,
                &context, frequency, iterations);
        measure(n, BENCH_FFMPEG_NATURAL, lane2, lane4,
                &context, frequency, iterations);
        measure(n, BENCH_FFMPEG_PREPERMUTED, lane2, lane4,
                &context, frequency, iterations);
        lane2_neon_plan_destroy(lane2);
        lane4_portable_plan_destroy(lane4);
    }
    linux_exit(0);
}
