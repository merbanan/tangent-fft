#ifndef TANGENT_FFT_LANE4_H
#define TANGENT_FFT_LANE4_H

#include "fft.h"

typedef struct lane4_fft_plan lane4_fft_plan;
typedef struct lane4_avx_fft_plan lane4_avx_fft_plan;
typedef struct lane4_avx_fma_fft_plan lane4_avx_fma_fft_plan;
typedef struct lane4_avx2_fft_plan lane4_avx2_fft_plan;

lane4_fft_plan *lane4_fft_plan_create(size_t n);
void lane4_fft_plan_destroy(lane4_fft_plan *plan);
int lane4_fft_execute(lane4_fft_plan *plan, fft_complex *data);

lane4_avx_fft_plan *lane4_avx_fft_plan_create(size_t n);
void lane4_avx_fft_plan_destroy(lane4_avx_fft_plan *plan);
int lane4_avx_fft_execute(lane4_avx_fft_plan *plan, fft_complex *data);

lane4_avx_fma_fft_plan *lane4_avx_fma_fft_plan_create(size_t n);
void lane4_avx_fma_fft_plan_destroy(lane4_avx_fma_fft_plan *plan);
int lane4_avx_fma_fft_execute(lane4_avx_fma_fft_plan *plan,
                              fft_complex *data);

lane4_avx2_fft_plan *lane4_avx2_fft_plan_create(size_t n);
void lane4_avx2_fft_plan_destroy(lane4_avx2_fft_plan *plan);
int lane4_avx2_fft_execute(lane4_avx2_fft_plan *plan,
                           fft_complex *data);

#endif
