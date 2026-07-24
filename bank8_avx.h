#ifndef TANGENT_FFT_BANK8_AVX_H
#define TANGENT_FFT_BANK8_AVX_H

#include "fft.h"

#include <stddef.h>

typedef struct bank8_avx_plan bank8_avx_plan;

bank8_avx_plan *bank8_avx_plan_create(size_t n);
void bank8_avx_plan_destroy(bank8_avx_plan *plan);

/*
 * Dual-bank radix-8 SIMD FFT.  The hot path is handwritten AVX2/FMA
 * assembly; the planner uses scalar float tables only.
 */
int bank8_avx_execute(const bank8_avx_plan *plan, fft_complex *data);

#endif
