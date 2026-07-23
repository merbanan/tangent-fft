#ifndef TANGENT_FFT_FFMPEG_ADAPTER_H
#define TANGENT_FFT_FFMPEG_ADAPTER_H

#include "fft.h"

typedef struct ffmpeg_fft_plan ffmpeg_fft_plan;

ffmpeg_fft_plan *ffmpeg_fft_plan_create(size_t n);
ffmpeg_fft_plan *ffmpeg_sse_fft_plan_create(size_t n);
void ffmpeg_fft_plan_destroy(ffmpeg_fft_plan *plan);
int ffmpeg_fft_execute(ffmpeg_fft_plan *plan, fft_complex *data);

#endif
