#include "fft.h"
#include "analysis/x86_tsc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { SAMPLE_COUNT = 41 };

static volatile float cycle_sink;

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static uint64_t run_block(fft_plan *plan,
                          fft_algorithm algorithm,
                          fft_complex *data,
                          size_t iterations)
{
    size_t iteration;
    const uint64_t start = x86_tsc_start();

    for (iteration = 0; iteration < iterations; ++iteration) {
        (void)fft_execute(plan, algorithm, data);
    }
    return x86_tsc_end() - start;
}

static int compare_pair(fft_plan *plan,
                        fft_algorithm reference,
                        fft_complex *reference_data,
                        fft_complex *candidate_data,
                        size_t iterations,
                        double *reference_cycles,
                        double *candidate_cycles,
                        double *candidate_over_reference)
{
    double references[SAMPLE_COUNT];
    double candidates[SAMPLE_COUNT];
    double ratios[SAMPLE_COUNT];
    size_t sample;

    if (!fft_plan_supports(plan, reference) ||
        !fft_plan_supports(plan, FFT_BANK8_AVX2_FMA)) {
        return 0;
    }
    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        uint64_t reference_count;
        uint64_t candidate_count;

        if ((sample & 1U) == 0) {
            reference_count = run_block(
                plan, reference, reference_data, iterations);
            candidate_count = run_block(
                plan, FFT_BANK8_AVX2_FMA, candidate_data, iterations);
        } else {
            candidate_count = run_block(
                plan, FFT_BANK8_AVX2_FMA, candidate_data, iterations);
            reference_count = run_block(
                plan, reference, reference_data, iterations);
        }
        references[sample] =
            (double)reference_count / (double)iterations;
        candidates[sample] =
            (double)candidate_count / (double)iterations;
        ratios[sample] = (double)candidate_count / (double)reference_count;
    }
    qsort(references, SAMPLE_COUNT, sizeof(*references), compare_double);
    qsort(candidates, SAMPLE_COUNT, sizeof(*candidates), compare_double);
    qsort(ratios, SAMPLE_COUNT, sizeof(*ratios), compare_double);
    *reference_cycles = references[SAMPLE_COUNT / 2];
    *candidate_cycles = candidates[SAMPLE_COUNT / 2];
    *candidate_over_reference = ratios[SAMPLE_COUNT / 2];
    cycle_sink += reference_data[fft_plan_size(plan) / 3].re;
    cycle_sink += candidate_data[fft_plan_size(plan) / 5].im;
    return 1;
}

int main(void)
{
    const fft_algorithm references[] = {
        FFT_LANE4_AVX,
        FFT_LANE4_AVX2_FMA,
        FFT_LANE8_AVX2_FMA,
        FFT_FFMPEG
    };
    unsigned power;

    puts("Bank8 paired cycles/execute (41 alternating samples)");
    printf("%7s %-18s %12s %12s %11s\n",
           "N", "reference", "reference", "bank8", "bank8/ref");

    for (power = 5; power <= 13; ++power) {
        const size_t n = (size_t)1 << power;
        size_t iterations = 4194304 / n;
        fft_plan *plan = fft_plan_create(n);
        fft_complex *reference_data =
            (fft_complex *)calloc(n, sizeof(*reference_data));
        fft_complex *candidate_data =
            (fft_complex *)calloc(n, sizeof(*candidate_data));
        size_t index;

        if (iterations < 512) {
            iterations = 512;
        }
        if (plan == NULL || reference_data == NULL ||
            candidate_data == NULL) {
            fprintf(stderr, "allocation failed for N=%zu\n", n);
            fft_plan_destroy(plan);
            free(reference_data);
            free(candidate_data);
            return EXIT_FAILURE;
        }
        for (index = 0; index < 8192; ++index) {
            (void)fft_execute(
                plan, FFT_BANK8_AVX2_FMA, candidate_data);
            (void)fft_execute(
                plan, FFT_LANE4_AVX2_FMA, reference_data);
        }
        for (index = 0;
             index < sizeof(references) / sizeof(references[0]);
             ++index) {
            double reference_cycles;
            double candidate_cycles;
            double ratio;

            if (compare_pair(plan, references[index],
                             reference_data, candidate_data, iterations,
                             &reference_cycles, &candidate_cycles, &ratio)) {
                printf("%7zu %-18s %12.2f %12.2f %10.4f%c\n",
                       n, fft_algorithm_name(references[index]),
                       reference_cycles, candidate_cycles, ratio,
                       ratio < 1.0 ? '*' : ' ');
            }
        }
        fft_plan_destroy(plan);
        free(reference_data);
        free(candidate_data);
    }
    return cycle_sink == 12345.0f ? EXIT_FAILURE : EXIT_SUCCESS;
}
