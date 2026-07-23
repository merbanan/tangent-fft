#ifndef TANGENT_FFT_LANE8_AVX_INTERNAL_H
#define TANGENT_FFT_LANE8_AVX_INTERNAL_H

#include "lane8_avx.h"

#include <stdint.h>

typedef struct {
    float re[8];
    float im[8];
} lane8_avx_row;

typedef lane8_avx_row lane8_avx_root;

struct lane8_avx_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    lane8_avx_root *replicated_root;
    float *finish_re;
    float *finish_im;
    lane8_avx_row *work;
};

#endif
