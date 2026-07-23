#ifndef TANGENT_FFT_H
#define TANGENT_FFT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float re;
    float im;
} fft_complex;

typedef enum {
    FFT_RADIX2 = 0,
    FFT_SPLIT_RADIX = 1,
    FFT_TANGENT = 2,
    FFT_TANGENT_X86_ASM = 3,
    FFT_TANGENT_SSE = 4,
    FFT_TANGENT_SSE2 = 5,
    FFT_TANGENT_SSE3 = 6,
    FFT_TANGENT_SSSE3 = 7,
    FFT_TANGENT_SSE41 = 8,
    FFT_TANGENT_SSE42 = 9,
    FFT_LANE4_C = 10,
    FFT_LANE4_SSE = 11,
    FFT_LANE4_SSE2 = 12,
    FFT_LANE4_SSE3 = 13,
    FFT_LANE4_SSSE3 = 14,
    FFT_LANE4_SSE41 = 15,
    FFT_LANE4_SSE42 = 16,
    FFT_LANE4_AVX = 17,
    FFT_LANE4_AVX_FMA = 18,
    FFT_LANE4_AVX2 = 19,
    FFT_LANE4_AVX2_FMA = 20,
    FFT_LANE4_RADIX4 = FFT_LANE4_AVX2_FMA,
    FFT_LANE2_SSE = 21,
    FFT_FFMPEG = 22,
    FFT_LANE8_AVX2_FMA = 23,
    FFT_HW_SSE = 24,
    FFT_ALGORITHM_COUNT = 25
} fft_algorithm;

typedef struct fft_plan fft_plan;

/*
 * Creates reusable tables and scratch space for a power-of-two transform.
 * Plan creation is intentionally separate from execution so benchmarks can
 * exclude allocation and trigonometric-table generation.
 */
fft_plan *fft_plan_create(size_t n);
void fft_plan_destroy(fft_plan *plan);

size_t fft_plan_size(const fft_plan *plan);
int fft_plan_supports(const fft_plan *plan, fft_algorithm algorithm);

/*
 * Computes the conventional, unnormalised forward DFT:
 *
 *   X[k] = sum_j x[j] exp(-2*pi*i*j*k/n)
 *
 * The transform is performed in-place. Returns 0 on success.
 */
int fft_execute(fft_plan *plan, fft_algorithm algorithm, fft_complex *data);

/*
 * Computes the conventional normalized inverse DFT:
 *
 *   x[j] = (1/n) sum_k X[k] exp(+2*pi*i*j*k/n)
 *
 * The transform is performed in-place. Returns 0 on success.
 */
int fft_inverse_execute(fft_plan *plan,
                        fft_algorithm algorithm,
                        fft_complex *data);

const char *fft_algorithm_name(fft_algorithm algorithm);

/*
 * Published real-arithmetic counts (additions plus multiplications), with
 * trivial roots of unity handled at their reduced costs.
 */
uint64_t fft_theoretical_flops(fft_algorithm algorithm, size_t n);

#endif
