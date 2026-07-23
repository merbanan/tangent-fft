#ifndef TANGENT_FFT_LANE4_PORTABLE_H
#define TANGENT_FFT_LANE4_PORTABLE_H

#include "fft.h"

typedef struct lane4_portable_plan lane4_portable_plan;

lane4_portable_plan *lane4_portable_plan_create(size_t n);
void lane4_portable_plan_destroy(lane4_portable_plan *plan);

int lane4_c_execute(lane4_portable_plan *plan, fft_complex *data);

#if HAVE_TANGENT_X86_ASM
int lane4_sse_execute(lane4_portable_plan *plan, fft_complex *data);
int lane4_sse2_execute(lane4_portable_plan *plan, fft_complex *data);
int lane4_sse3_execute(lane4_portable_plan *plan, fft_complex *data);
int lane4_ssse3_execute(lane4_portable_plan *plan, fft_complex *data);
int lane4_sse41_execute(lane4_portable_plan *plan, fft_complex *data);
int lane4_sse42_execute(lane4_portable_plan *plan, fft_complex *data);
#endif

#endif
