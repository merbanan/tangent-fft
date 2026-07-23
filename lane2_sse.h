#ifndef TANGENT_FFT_LANE2_SSE_H
#define TANGENT_FFT_LANE2_SSE_H

#include "fft.h"

#include <stddef.h>

typedef struct lane2_sse_plan lane2_sse_plan;

lane2_sse_plan *lane2_sse_plan_create(size_t n);
void lane2_sse_plan_destroy(lane2_sse_plan *plan);

int lane2_sse_execute(lane2_sse_plan *plan, fft_complex *data);

#endif
