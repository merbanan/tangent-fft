#ifndef TANGENT_FFT_BANK8_AVX_INTERNAL_H
#define TANGENT_FFT_BANK8_AVX_INTERNAL_H

#include "bank8_avx.h"

#include <stdint.h>

typedef struct {
    float value[8];
} bank8_avx_row;

typedef struct {
    bank8_avx_row w1_re;
    bank8_avx_row w1_im;
    bank8_avx_row w2_re;
    bank8_avx_row w2_im;
    bank8_avx_row w3_re;
    bank8_avx_row w3_im;
} bank8_avx_twiddle;

typedef struct {
    bank8_avx_row bank[2];
} bank8_avx_work_row;

struct bank8_avx_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    bank8_avx_twiddle *twiddle;
    bank8_avx_row *finish_factor;
    bank8_avx_work_row *work;
};

#endif
