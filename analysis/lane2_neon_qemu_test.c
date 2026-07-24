/*
 * Freestanding AArch64 correctness runner for lane2-neon.
 *
 * This deliberately supplies only the tiny allocator, trigonometric
 * functions, and Linux syscalls needed to exercise the production planner
 * and assembly under qemu-aarch64.  It is not part of the benchmark binary.
 */

#include "lane2_neon.h"
#include "lane4_portable.h"

#include <stddef.h>
#include <stdint.h>

#ifndef TEST_MIN_N
#define TEST_MIN_N 16
#endif
#ifndef TEST_MAX_N
#define TEST_MAX_N 8192
#endif
#ifndef TEST_LANE2
#define TEST_LANE2 1
#endif
#define TEST_PI 3.141592653589793238462643383279502884

static _Alignas(64) unsigned char arena[1U << 20];
static size_t arena_used;
static fft_complex input[TEST_MAX_N];
typedef struct {
    double re;
    double im;
} reference_complex;
static reference_complex reference[TEST_MAX_N];

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
    size_t bytes;
    unsigned char *result;
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
        term *= -x2 /
                (double)((2U * k) * (2U * k + 1U));
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
        term *= -x2 /
                (double)((2U * k - 1U) * (2U * k));
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

static void reference_fft(const fft_complex *source,
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

static int check_size(size_t n)
{
    lane2_neon_plan *plan;
    uint32_t state = 0x12345678U ^ (uint32_t)n;
    double reference_scale = 1.0;
    double maximum_error = 0.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        state = state * 1664525U + 1013904223U;
        input[i].re =
            ((float)(state >> 8) / 8388608.0f) - 1.0f;
        state = state * 1664525U + 1013904223U;
        input[i].im =
            ((float)(state >> 8) / 8388608.0f) - 1.0f;
    }
    reference_fft(input, reference, n);
    for (i = 0; i < n; ++i) {
        const double magnitude =
            (reference[i].re < 0.0 ? -reference[i].re : reference[i].re) +
            (reference[i].im < 0.0 ? -reference[i].im : reference[i].im);
        if (magnitude > reference_scale) {
            reference_scale = magnitude;
        }
    }

    plan = lane2_neon_plan_create(n);
    if (plan == NULL || lane2_neon_execute(plan, input) != 0) {
        return 0;
    }
    for (i = 0; i < n; ++i) {
        const double re_difference =
            (double)input[i].re - reference[i].re;
        const double im_difference =
            (double)input[i].im - reference[i].im;
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
    lane2_neon_plan_destroy(plan);
    return maximum_error / reference_scale < 2.0e-5;
}

static int check_lane4_size(size_t n)
{
    lane4_portable_plan *plan;
    uint32_t state = 0x41a4e4U ^ (uint32_t)n;
    double reference_scale = 1.0;
    double maximum_error = 0.0;
    size_t i;

    for (i = 0; i < n; ++i) {
        state = state * 1664525U + 1013904223U;
        input[i].re =
            ((float)(state >> 8) / 8388608.0f) - 1.0f;
        state = state * 1664525U + 1013904223U;
        input[i].im =
            ((float)(state >> 8) / 8388608.0f) - 1.0f;
    }
    reference_fft(input, reference, n);
    for (i = 0; i < n; ++i) {
        const double magnitude =
            (reference[i].re < 0.0 ? -reference[i].re : reference[i].re) +
            (reference[i].im < 0.0 ? -reference[i].im : reference[i].im);
        if (magnitude > reference_scale) {
            reference_scale = magnitude;
        }
    }

    plan = lane4_portable_plan_create(n);
    if (plan == NULL || lane4_neon_execute(plan, input) != 0) {
        return 0;
    }
    for (i = 0; i < n; ++i) {
        const double re_difference =
            (double)input[i].re - reference[i].re;
        const double im_difference =
            (double)input[i].im - reference[i].im;
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
    lane4_portable_plan_destroy(plan);
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
        "PASS: lane2-neon and fused lane4-neon, N=16..8192\n";
    static const char fail_message[] =
        "FAIL: lane2-neon production planner/assembly\n";
    size_t n;

    if (TEST_LANE2) {
        for (n = TEST_MIN_N; n <= TEST_MAX_N; n *= 2) {
            if (!check_size(n)) {
                (void)linux_write(2, fail_message,
                                  sizeof(fail_message) - 1);
                linux_exit(1);
            }
        }
    }
    arena_used = 0;
    for (n = TEST_MIN_N; n <= TEST_MAX_N; n *= 2) {
        if (!check_lane4_size(n)) {
            (void)linux_write(2, fail_message, sizeof(fail_message) - 1);
            linux_exit(1);
        }
    }
    (void)linux_write(1, pass_message, sizeof(pass_message) - 1);
    linux_exit(0);
}
