/*
 * Freestanding correctness runner for FFmpeg's AArch64 FFT kernels.
 *
 * This supplies the cosine tables and split-radix gather map normally built
 * by libavutil, allowing the exact assembly object to run under qemu-aarch64
 * without an AArch64 libc/sysroot.
 */

#include <stddef.h>
#include <stdint.h>

#ifndef TEST_MIN_N
#define TEST_MIN_N 16
#endif
#ifndef TEST_MAX_N
#define TEST_MAX_N 256
#endif
#ifndef TEST_FFMPEG_NATURAL
#define TEST_FFMPEG_NATURAL 0
#endif
#define TEST_PI 3.141592653589793238462643383279502884

typedef struct {
    float re;
    float im;
} test_complex;

typedef struct {
    double re;
    double im;
} reference_complex;

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

static _Alignas(64) test_complex input[TEST_MAX_N];
static _Alignas(64) test_complex permuted[TEST_MAX_N];
static _Alignas(64) test_complex output[TEST_MAX_N];
static reference_complex reference[TEST_MAX_N];
static int split_radix_map[TEST_MAX_N];

static double reduce_angle(double value)
{
    const double period = 2.0 * TEST_PI;

    while (value > TEST_PI) {
        value -= period;
    }
    while (value < -TEST_PI) {
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
    parity_map_generator(map, n, inverse, offset + length + (length >> 1),
                         length >> 1, basis);
}

static void make_split_radix_map(int n)
{
    parity_map_generator(split_radix_map, n, 0, 0, n, 4);
}

static void initialize_table(float *table, int n)
{
    int i;

    for (i = 0; i < n / 4; ++i) {
        table[i] = (float)test_cos(2.0 * TEST_PI * (double)i / (double)n);
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
    if (n >= 16384) {
        initialize_table(ff_tx_tab_16384_float, 16384);
    }
    if (n >= 32768) {
        initialize_table(ff_tx_tab_32768_float, 32768);
    }
    if (n >= 65536) {
        initialize_table(ff_tx_tab_65536_float, 65536);
    }
    if (n >= 131072) {
        initialize_table(ff_tx_tab_131072_float, 131072);
    }
}

static void reference_fft(const test_complex *source,
                          reference_complex *destination,
                          size_t n)
{
    size_t i;
    size_t j = 0;
    size_t block;

    for (i = 0; i < n; ++i) {
        destination[i].re = (double)source[i].re;
        destination[i].im = (double)source[i].im;
    }
    for (i = 1; i < n; ++i) {
        size_t bit = n >> 1;

        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const reference_complex temporary = destination[i];
            destination[i] = destination[j];
            destination[j] = temporary;
        }
    }
    for (block = 2; block <= n; block *= 2) {
        const size_t half = block / 2;
        size_t k;

        for (k = 0; k < half; ++k) {
            const double angle =
                -2.0 * TEST_PI * (double)k / (double)block;
            const double wr = test_cos(angle);
            const double wi = test_sin(angle);
            size_t start;

            for (start = 0; start < n; start += block) {
                const size_t low = start + k;
                const size_t high = low + half;
                const reference_complex even = destination[low];
                const reference_complex odd = destination[high];
                const double rotated_re = odd.re * wr - odd.im * wi;
                const double rotated_im = odd.re * wi + odd.im * wr;

                destination[low].re = even.re + rotated_re;
                destination[low].im = even.im + rotated_im;
                destination[high].re = even.re - rotated_re;
                destination[high].im = even.im - rotated_im;
            }
        }
    }
}

static int check_size(int n)
{
    test_avtx_context context;
    uint32_t state = 0x715517U ^ (uint32_t)n;
    double reference_scale = 1.0;
    double maximum_error = 0.0;
    int i;

    for (i = 0; i < n; ++i) {
        state = state * 1664525U + 1013904223U;
        input[i].re = ((float)(state >> 8) / 8388608.0f) - 1.0f;
        state = state * 1664525U + 1013904223U;
        input[i].im = ((float)(state >> 8) / 8388608.0f) - 1.0f;
    }

    reference_fft(input, reference, (size_t)n);
    make_split_radix_map(n);
    initialize_tables(n);
    for (i = 0; i < n; ++i) {
        const double magnitude =
            (reference[i].re < 0.0 ? -reference[i].re : reference[i].re) +
            (reference[i].im < 0.0 ? -reference[i].im : reference[i].im);

        permuted[i] = input[split_radix_map[i]];
        if (magnitude > reference_scale) {
            reference_scale = magnitude;
        }
    }

    context.len = n;
    context.inv = 0;
    context.map = split_radix_map;
    if (n == 16) {
        if (TEST_FFMPEG_NATURAL) {
            ff_tx_fft16_float_neon(&context, output, input, 8);
        } else {
            ff_tx_fft16_ns_float_neon(&context, output, permuted, 8);
        }
    } else if (n == 32) {
        if (TEST_FFMPEG_NATURAL) {
            ff_tx_fft32_float_neon(&context, output, input, 8);
        } else {
            ff_tx_fft32_ns_float_neon(&context, output, permuted, 8);
        }
    } else if (TEST_FFMPEG_NATURAL) {
        ff_tx_fft_sr_float_neon(&context, output, input, 8);
    } else {
        ff_tx_fft_sr_ns_float_neon(&context, output, permuted, 8);
    }

    for (i = 0; i < n; ++i) {
        const double re_difference =
            (double)output[i].re - reference[i].re;
        const double im_difference =
            (double)output[i].im - reference[i].im;
        const double re_error =
            re_difference < 0.0 ? -re_difference : re_difference;
        const double im_error =
            im_difference < 0.0 ? -im_difference : im_difference;

        if (re_error > maximum_error) {
            maximum_error = re_error;
        }
        if (im_error > maximum_error) {
            maximum_error = im_error;
        }
    }

    return maximum_error / reference_scale < 2.0e-5;
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

void _start(void)
{
    static const char pass_message[] =
        "PASS: FFmpeg NEON FFT assembly\n";
    static const char fail_message[] =
        "FAIL: FFmpeg NEON FFT assembly\n";
    int n;

    for (n = TEST_MIN_N; n <= TEST_MAX_N; n *= 2) {
        if (!check_size(n)) {
            (void)linux_write(2, fail_message, sizeof(fail_message) - 1);
            linux_exit(1);
        }
    }
    (void)linux_write(1, pass_message, sizeof(pass_message) - 1);
    linux_exit(0);
}
