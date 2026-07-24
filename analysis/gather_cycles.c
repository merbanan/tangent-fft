#include "fft.h"
#include "analysis/x86_tsc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SAMPLE_COUNT = 31 };

typedef void (*permute_function)(const fft_complex *,
                                 fft_complex *,
                                 const uint32_t *,
                                 size_t);

void analysis_vgather_permute(const fft_complex *input,
                              fft_complex *output,
                              const uint32_t *permutation,
                              size_t count);
void analysis_scalar_permute(const fft_complex *input,
                             fft_complex *output,
                             const uint32_t *permutation,
                             size_t count);

static volatile float cycle_sink;

static int compare_u64(const void *left, const void *right)
{
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint32_t reverse_bits(uint32_t value, unsigned bits)
{
    uint32_t result = 0;
    unsigned bit;

    for (bit = 0; bit < bits; ++bit) {
        result = (result << 1) | (value & 1U);
        value >>= 1;
    }
    return result;
}

static double measure(permute_function function,
                      const fft_complex *input,
                      fft_complex *output,
                      const uint32_t *permutation,
                      size_t n,
                      size_t iterations)
{
    uint64_t samples[SAMPLE_COUNT];
    size_t sample;

    for (sample = 0; sample < SAMPLE_COUNT; ++sample) {
        size_t iteration;
        const uint64_t start = x86_tsc_start();

        for (iteration = 0; iteration < iterations; ++iteration) {
            function(input, output, permutation, n);
        }
        samples[sample] = x86_tsc_end() - start;
    }
    qsort(samples, SAMPLE_COUNT, sizeof(*samples), compare_u64);
    cycle_sink += output[(n * 7) / 11].re;
    return (double)samples[SAMPLE_COUNT / 2] / (double)iterations;
}

int main(void)
{
    unsigned power;

    puts("AVX2 gather versus scalar permutation (median cycles/copy)");
    printf("%8s %14s %14s %12s\n",
           "N", "vgatherdpd", "scalar loads", "scalar gain");

    for (power = 4; power <= 13; ++power) {
        const size_t n = (size_t)1 << power;
        const size_t iterations =
            n < 4096 ? 4194304 / n : 1024;
        fft_complex *input =
            (fft_complex *)aligned_alloc(32, n * sizeof(*input));
        fft_complex *gathered =
            (fft_complex *)aligned_alloc(32, n * sizeof(*gathered));
        fft_complex *scalar =
            (fft_complex *)aligned_alloc(32, n * sizeof(*scalar));
        uint32_t *permutation =
            (uint32_t *)aligned_alloc(32, n * sizeof(*permutation));
        double gather_cycles;
        double scalar_cycles;
        size_t i;

        if (input == NULL || gathered == NULL || scalar == NULL ||
            permutation == NULL) {
            free(input);
            free(gathered);
            free(scalar);
            free(permutation);
            return EXIT_FAILURE;
        }
        for (i = 0; i < n; ++i) {
            input[i].re = (float)i;
            input[i].im = (float)(i ^ 0x55U);
            permutation[i] = reverse_bits((uint32_t)i, power);
        }
        analysis_vgather_permute(input, gathered, permutation, n);
        analysis_scalar_permute(input, scalar, permutation, n);
        if (memcmp(gathered, scalar, n * sizeof(*scalar)) != 0) {
            fprintf(stderr, "permutation mismatch at N=%zu\n", n);
            free(input);
            free(gathered);
            free(scalar);
            free(permutation);
            return EXIT_FAILURE;
        }

        gather_cycles = measure(analysis_vgather_permute,
                                input, gathered, permutation,
                                n, iterations);
        scalar_cycles = measure(analysis_scalar_permute,
                                input, scalar, permutation,
                                n, iterations);
        printf("%8zu %14.3f %14.3f %11.2f%%\n",
               n,
               gather_cycles,
               scalar_cycles,
               100.0 * (gather_cycles / scalar_cycles - 1.0));

        free(input);
        free(gathered);
        free(scalar);
        free(permutation);
    }
    return cycle_sink == 12345.0f ? EXIT_FAILURE : EXIT_SUCCESS;
}
