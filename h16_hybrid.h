#ifndef TANGENT_H16_HYBRID_H
#define TANGENT_H16_HYBRID_H

#include "fft.h"

#include <stddef.h>

typedef struct h16_hybrid_plan h16_hybrid_plan;

h16_hybrid_plan *h16_hybrid_plan_create(size_t n);
void h16_hybrid_plan_destroy(h16_hybrid_plan *plan);
int h16_hybrid_fft_execute(h16_hybrid_plan *plan, fft_complex *data);
int h16_hybrid_paired_avx2_execute(h16_hybrid_plan *plan,
                                   fft_complex *data);
int h16_hybrid_paired_avx2_available(const h16_hybrid_plan *plan);

#endif
