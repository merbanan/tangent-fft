#ifndef TANGENT_FFT_LANE2_NEON_H
#define TANGENT_FFT_LANE2_NEON_H

#include "fft.h"

#include <stddef.h>

typedef struct lane2_neon_plan lane2_neon_plan;

lane2_neon_plan *lane2_neon_plan_create(size_t n);
void lane2_neon_plan_destroy(lane2_neon_plan *plan);

int lane2_neon_execute(lane2_neon_plan *plan, fft_complex *data);

#endif
