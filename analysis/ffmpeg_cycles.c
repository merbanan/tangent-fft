#include "ffmpeg_fft.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

enum { SAMPLE_COUNT = 31 };

static volatile float cycle_sink;

static int compare_u64(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t read_tsc_start(void)
{
    _mm_lfence();
    return __rdtsc();
}

static uint64_t read_tsc_end(void)
{
    unsigned auxiliary;
    const uint64_t value = __rdtscp(&auxiliary);
    _mm_lfence();
    return value;
}

int main(int argc, char **argv)
{
    unsigned first_power = 4;
    unsigned last_power = 13;
    unsigned power;

    if (argc == 2) {
        char *end;
        const unsigned long selected = strtoul(argv[1], &end, 10);
        if (*argv[1] == '\0' || *end != '\0' ||
            selected < 4 || selected > 13) {
            fprintf(stderr, "usage: %s [power: 4..13]\n", argv[0]);
            return EXIT_FAILURE;
        }
        first_power = last_power = (unsigned)selected;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [power: 4..13]\n", argv[0]);
        return EXIT_FAILURE;
    }

    puts("FFmpeg AVTX cycles/execute (median, adapter copies included)");
    printf("%10s %14s %12s %8s\n",
           "N", "median cycles", "iterations", "samples");

    for (power = first_power; power <= last_power; ++power) {
        const size_t n = (size_t)1 << power;
        size_t iterations = 1048576 / n;
        size_t warm_iterations = 67108864 / n;
        ffmpeg_fft_plan *plan = ffmpeg_fft_plan_create(n);
        fft_complex *data = (fft_complex *)calloc(n, sizeof(*data));
        uint64_t samples[SAMPLE_COUNT];
        size_t sample;

        if (iterations < 128) {
            iterations = 128;
        }
        if (plan == NULL || data == NULL) {
            fprintf(stderr, "allocation failed for N=%zu\n", n);
            ffmpeg_fft_plan_destroy(plan);
            free(data);
            return EXIT_FAILURE;
        }

        if (warm_iterations < 8192) {
            warm_iterations = 8192;
        }
        for (sample = 0; sample < warm_iterations; ++sample) {
            if (ffmpeg_fft_execute(plan, data) != 0) {
                fprintf(stderr, "FFmpeg execution failed for N=%zu\n", n);
                ffmpeg_fft_plan_destroy(plan);
                free(data);
                return EXIT_FAILURE;
            }
        }

        for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
            size_t iteration;
            const uint64_t start = read_tsc_start();
            for (iteration = 0; iteration < iterations; ++iteration) {
                (void)ffmpeg_fft_execute(plan, data);
            }
            samples[sample] = read_tsc_end() - start;
        }

        qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
        cycle_sink += data[(n * 7) / 11].re;
        printf("%10zu %14.3f %12zu %8d\n",
               n,
               (double)samples[SAMPLE_COUNT / 2] / (double)iterations,
               iterations,
               SAMPLE_COUNT);
        ffmpeg_fft_plan_destroy(plan);
        free(data);
    }
    return cycle_sink == 12345.0f ? EXIT_FAILURE : EXIT_SUCCESS;
}
