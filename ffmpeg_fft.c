#include "ffmpeg_fft.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "libavutil/tx.h"

struct ffmpeg_fft_plan {
    size_t n;
    AVTXContext *context;
    av_tx_fn transform;
    AVComplexFloat *input;
    AVComplexFloat *output;
};

_Static_assert(sizeof(fft_complex) == sizeof(AVComplexFloat),
               "FFmpeg and local float-complex layouts must match");

static void *aligned_buffer(size_t bytes)
{
    const size_t alignment = 64;
    const size_t rounded =
        (bytes + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, rounded);
}

ffmpeg_fft_plan *ffmpeg_fft_plan_create(size_t n)
{
    ffmpeg_fft_plan *plan;
    const size_t bytes = n * sizeof(AVComplexFloat);

    if (n == 0 || n > INT_MAX || bytes / sizeof(AVComplexFloat) != n) {
        return NULL;
    }

    plan = (ffmpeg_fft_plan *)calloc(1, sizeof(*plan));
    if (plan == NULL) {
        return NULL;
    }
    plan->n = n;
    plan->input = (AVComplexFloat *)aligned_buffer(bytes);
    plan->output = (AVComplexFloat *)aligned_buffer(bytes);
    if (plan->input == NULL || plan->output == NULL ||
        av_tx_init(&plan->context,
                   &plan->transform,
                   AV_TX_FLOAT_FFT,
                   0,
                   (int)n,
                   NULL,
                   0) < 0) {
        ffmpeg_fft_plan_destroy(plan);
        return NULL;
    }
    return plan;
}

void ffmpeg_fft_plan_destroy(ffmpeg_fft_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    av_tx_uninit(&plan->context);
    free(plan->input);
    free(plan->output);
    free(plan);
}

int ffmpeg_fft_execute(ffmpeg_fft_plan *plan, fft_complex *data)
{
    const size_t bytes =
        plan == NULL ? 0 : plan->n * sizeof(AVComplexFloat);
    if (plan == NULL || data == NULL || plan->transform == NULL) {
        return -1;
    }

    memcpy(plan->input, data, bytes);
    plan->transform(plan->context,
                    plan->output,
                    plan->input,
                    (ptrdiff_t)sizeof(AVComplexFloat));
    memcpy(data, plan->output, bytes);
    return 0;
}
