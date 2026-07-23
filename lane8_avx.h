#ifndef TANGENT_FFT_LANE8_AVX_H
#define TANGENT_FFT_LANE8_AVX_H

#include "fft.h"

typedef struct lane8_avx_plan lane8_avx_plan;

lane8_avx_plan *lane8_avx_plan_create(size_t n);
void lane8_avx_plan_destroy(lane8_avx_plan *plan);

/* AVX2/FMA execution is implemented entirely in lane8_avx.asm. */
int lane8_avx_execute(lane8_avx_plan *plan, fft_complex *data);

#endif
