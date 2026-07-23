#ifndef TANGENT_FFT_SSE_ASM_H
#define TANGENT_FFT_SSE_ASM_H

#include "fft.h"

void tangent_sse_batch_base(fft_complex *data,
                            const uint32_t *offsets,
                            size_t node_count,
                            int s4_kind);
void tangent_sse_batch_leaf2(fft_complex *data,
                             const uint32_t *offsets,
                             size_t leaf_count,
                             const fft_complex *tables,
                             unsigned kind);
void tangent_sse_batch_leaf3(fft_complex *data,
                             const uint32_t *offsets,
                             size_t leaf_count,
                             const fft_complex *tables,
                             unsigned kind);
void tangent_sse_batch_leaf4(fft_complex *data,
                             const uint32_t *offsets,
                             size_t leaf_count,
                             const fft_complex *tables,
                             unsigned kind);
void tangent_sse3_batch_leaf2(fft_complex *data,
                              const uint32_t *offsets,
                              size_t leaf_count,
                              const fft_complex *tables,
                              unsigned kind);
void tangent_sse3_batch_leaf3(fft_complex *data,
                              const uint32_t *offsets,
                              size_t leaf_count,
                              const fft_complex *tables,
                              unsigned kind);
void tangent_sse3_batch_leaf4(fft_complex *data,
                              const uint32_t *offsets,
                              size_t leaf_count,
                              const fft_complex *tables,
                              unsigned kind);
#define DECLARE_SSE_GATHER_LEAF(name)                                  \
    void name(const fft_complex *input,                                \
              fft_complex *output,                                     \
              const uint32_t *permutation,                             \
              const uint32_t *offsets,                                 \
              size_t leaf_count,                                       \
              const fft_complex *tables,                              \
              unsigned kind)
DECLARE_SSE_GATHER_LEAF(tangent_sse_gather_leaf2);
DECLARE_SSE_GATHER_LEAF(tangent_sse_gather_leaf3);
DECLARE_SSE_GATHER_LEAF(tangent_sse_gather_leaf4);
DECLARE_SSE_GATHER_LEAF(tangent_sse3_gather_leaf2);
DECLARE_SSE_GATHER_LEAF(tangent_sse3_gather_leaf3);
DECLARE_SSE_GATHER_LEAF(tangent_sse3_gather_leaf4);
#undef DECLARE_SSE_GATHER_LEAF

void tangent_sse_batch_unscaled(fft_complex *data,
                                const uint32_t *offsets,
                                size_t node_count,
                                size_t quarter,
                                const fft_complex *factor);
void tangent_sse3_batch_unscaled(fft_complex *data,
                                 const uint32_t *offsets,
                                 size_t node_count,
                                 size_t quarter,
                                 const fft_complex *factor);
void tangent_sse_batch_tangent(fft_complex *data,
                               const uint32_t *offsets,
                               size_t node_count,
                               size_t quarter,
                               const fft_complex *factor);
void tangent_sse3_batch_tangent(fft_complex *data,
                                const uint32_t *offsets,
                                size_t node_count,
                                size_t quarter,
                                const fft_complex *factor);

void tangent_sse_batch_s2(fft_complex *data,
                          const uint32_t *offsets,
                          size_t node_count,
                          size_t quarter,
                          const fft_complex *factor,
                          const fft_complex *low_scale,
                          const fft_complex *high_scale);
void tangent_sse3_batch_s2(fft_complex *data,
                           const uint32_t *offsets,
                           size_t node_count,
                           size_t quarter,
                           const fft_complex *factor,
                           const fft_complex *low_scale,
                           const fft_complex *high_scale);

void tangent_sse_batch_s4(fft_complex *data,
                          const uint32_t *offsets,
                          size_t node_count,
                          size_t quarter,
                          const fft_complex *factor,
                          const fft_complex *scale0,
                          const fft_complex *scale1,
                          const fft_complex *scale2,
                          const fft_complex *scale3);
void tangent_sse3_batch_s4(fft_complex *data,
                           const uint32_t *offsets,
                           size_t node_count,
                           size_t quarter,
                           const fft_complex *factor,
                           const fft_complex *scale0,
                           const fft_complex *scale1,
                           const fft_complex *scale2,
                           const fft_complex *scale3);

#endif
