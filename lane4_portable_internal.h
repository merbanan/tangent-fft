#ifndef TANGENT_FFT_LANE4_PORTABLE_INTERNAL_H
#define TANGENT_FFT_LANE4_PORTABLE_INTERNAL_H

#include "lane4_portable.h"

#include <stdint.h>

typedef struct {
    float re[4];
    float im[4];
} lane4_portable_row;

struct lane4_portable_plan {
    size_t n;
    size_t inner_size;
    unsigned inner_levels;
    uint32_t *permutation;
    fft_complex *root;
    float *finish_re;
    float *finish_im;
    lane4_portable_row *work;
};

#endif
