#include "ffmpeg_fft.h"

#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "libavutil/tx.h"
#include "libavutil/cpu.h"

struct ffmpeg_fft_plan {
    size_t n;
    AVTXContext *context;
    av_tx_fn transform;
    AVComplexFloat *input;
    AVComplexFloat *output;
};

_Static_assert(sizeof(fft_complex) == sizeof(AVComplexFloat),
               "FFmpeg and local float-complex layouts must match");

/*
 * av_force_cpu_flags() changes process-global state. Serialize AVTX plan
 * creation through this adapter so its native and restricted plans cannot
 * race each other.
 */
static pthread_mutex_t cpu_flags_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *aligned_buffer(size_t bytes)
{
    const size_t alignment = 64;
    const size_t rounded =
        (bytes + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, rounded);
}

static ffmpeg_fft_plan *ffmpeg_fft_plan_create_unlocked(size_t n)
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

ffmpeg_fft_plan *ffmpeg_fft_plan_create(size_t n)
{
    ffmpeg_fft_plan *plan;

    pthread_mutex_lock(&cpu_flags_mutex);
    plan = ffmpeg_fft_plan_create_unlocked(n);
    pthread_mutex_unlock(&cpu_flags_mutex);
    return plan;
}

ffmpeg_fft_plan *ffmpeg_sse_fft_plan_create(size_t n)
{
#if defined(__i386__) || defined(__x86_64__)
    int saved_flags;
    int sse_flags;
    const int sse_cap =
        AV_CPU_FLAG_MMX | AV_CPU_FLAG_MMXEXT |
        AV_CPU_FLAG_3DNOW | AV_CPU_FLAG_3DNOWEXT |
        AV_CPU_FLAG_SSE | AV_CPU_FLAG_SSE2 | AV_CPU_FLAG_SSE3 |
        AV_CPU_FLAG_SSSE3 | AV_CPU_FLAG_SSE4 | AV_CPU_FLAG_SSE42 |
        AV_CPU_FLAG_SSE2SLOW | AV_CPU_FLAG_SSE3SLOW |
        AV_CPU_FLAG_SSSE3SLOW | AV_CPU_FLAG_ATOM |
        AV_CPU_FLAG_CMOV;
    ffmpeg_fft_plan *plan;

    /*
     * AVTX resolves codelet function pointers during plan creation. Restore
     * the process-wide native mask immediately; this context keeps calling
     * the SSE codelets selected while the restricted mask was active.
     */
    pthread_mutex_lock(&cpu_flags_mutex);
    saved_flags = av_get_cpu_flags();
    sse_flags = saved_flags & sse_cap;
    av_force_cpu_flags(sse_flags);
    plan = ffmpeg_fft_plan_create_unlocked(n);
    av_force_cpu_flags(saved_flags);
    pthread_mutex_unlock(&cpu_flags_mutex);
    return plan;
#else
    (void)n;
    return NULL;
#endif
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
