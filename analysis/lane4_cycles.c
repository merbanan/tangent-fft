#include "fft.h"
#include "analysis/x86_tsc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { SAMPLE_COUNT = 31 };

static volatile float cycle_sink;

static int compare_u64(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static int measure(fft_plan *plan,
                   fft_algorithm algorithm,
                   fft_complex *data,
                   size_t iterations,
                   double *cycles)
{
    uint64_t samples[SAMPLE_COUNT];
    size_t sample;

    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        const uint64_t start = x86_tsc_start();

        for (iteration = 0; iteration < iterations; ++iteration) {
            if (fft_execute(plan, algorithm, data) != 0) {
                return 0;
            }
        }
        samples[sample] = x86_tsc_end() - start;
    }
    qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
    *cycles =
        (double)samples[SAMPLE_COUNT / 2] / (double)iterations;
    cycle_sink += data[(fft_plan_size(plan) * 7) / 11].re;
    return 1;
}

int main(int argc, char **argv)
{
    unsigned first_power = 4;
    unsigned last_power = 13;
    unsigned power;
    const fft_algorithm algorithms[] = {
        FFT_LANE4_AVX,
        FFT_LANE4_AVX_FMA,
        FFT_LANE4_AVX2,
        FFT_LANE4_AVX2_FMA,
        FFT_LANE4_SSE,
        FFT_FFMPEG
    };

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

    puts("Lane4 comparison cycles/execute (median; public in-place API)");
    printf("%10s %-18s %14s %12s\n",
           "N", "algorithm", "median cycles", "iterations");

    for (power = first_power; power <= last_power; ++power) {
        const size_t n = (size_t)1 << power;
        size_t iterations = 2097152 / n;
        size_t warm_iterations = 67108864 / n;
        fft_plan *plan = fft_plan_create(n);
        fft_complex *data = (fft_complex *)calloc(n, sizeof(*data));
        size_t algorithm_index;
        size_t iteration;

        if (iterations < 256) {
            iterations = 256;
        }
        if (warm_iterations < 8192) {
            warm_iterations = 8192;
        }
        if (plan == NULL || data == NULL) {
            fprintf(stderr, "allocation failed for N=%zu\n", n);
            fft_plan_destroy(plan);
            free(data);
            return EXIT_FAILURE;
        }

        for (iteration = 0; iteration < warm_iterations; ++iteration) {
            (void)fft_execute(plan, FFT_LANE4_AVX2_FMA, data);
        }

        for (algorithm_index = 0;
             algorithm_index < sizeof(algorithms) / sizeof(algorithms[0]);
             ++algorithm_index) {
            const fft_algorithm algorithm = algorithms[algorithm_index];
            double cycles;

            if (!fft_plan_supports(plan, algorithm)) {
                continue;
            }
            if (!measure(plan, algorithm, data, iterations, &cycles)) {
                fprintf(stderr, "execution failed for N=%zu\n", n);
                fft_plan_destroy(plan);
                free(data);
                return EXIT_FAILURE;
            }
            printf("%10zu %-18s %14.3f %12zu\n",
                   n,
                   fft_algorithm_name(algorithm),
                   cycles,
                   iterations);
        }
        fft_plan_destroy(plan);
        free(data);
    }

    return cycle_sink == 12345.0f ? EXIT_FAILURE : EXIT_SUCCESS;
}
