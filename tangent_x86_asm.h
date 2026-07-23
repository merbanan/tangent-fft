#ifndef TANGENT_FFT_X86_ASM_H
#define TANGENT_FFT_X86_ASM_H

#include "fft.h"

void tangent_x86_permute(const fft_complex *input,
                         fft_complex *output,
                         const uint32_t *permutation,
                         size_t count);
void tangent_x86_batch_base(fft_complex *data,
                            const uint32_t *offsets,
                            size_t count);
void tangent_x86_batch_base_s4(fft_complex *data,
                               const uint32_t *offsets,
                               size_t count);
void tangent_x86_batch_leaf2(fft_complex *data,
                             const uint32_t *offsets,
                             size_t count,
                             const fft_complex *tables,
                             unsigned kind);
void tangent_x86_batch_leaf3(fft_complex *data,
                             const uint32_t *offsets,
                             size_t count,
                             const fft_complex *tables,
                             unsigned kind);
void tangent_x86_batch_leaf4(fft_complex *data,
                             const uint32_t *offsets,
                             size_t count,
                             const fft_complex *tables,
                             unsigned kind);
#define DECLARE_GATHER_LEAF(level, suffix)                              \
    void tangent_x86_gather_leaf##level##_##suffix(                    \
        const fft_complex *input,                                      \
        fft_complex *output,                                           \
        const uint32_t *permutation,                                   \
        const uint32_t *offsets,                                       \
        size_t count,                                                   \
        const fft_complex *tables)

DECLARE_GATHER_LEAF(2, n);
DECLARE_GATHER_LEAF(2, s);
DECLARE_GATHER_LEAF(2, s2);
DECLARE_GATHER_LEAF(2, s4);
DECLARE_GATHER_LEAF(3, n);
DECLARE_GATHER_LEAF(3, s);
DECLARE_GATHER_LEAF(3, s2);
DECLARE_GATHER_LEAF(3, s4);
DECLARE_GATHER_LEAF(4, n);
DECLARE_GATHER_LEAF(4, s);
DECLARE_GATHER_LEAF(4, s2);
DECLARE_GATHER_LEAF(4, s4);
#undef DECLARE_GATHER_LEAF
void tangent_x86_gather_fft32_normal(const fft_complex *input,
                                     fft_complex *output,
                                     const uint32_t *permutation,
                                     const fft_complex *tables,
                                     const fft_complex *factor);
void tangent_x86_gather_fft64_normal(const fft_complex *input,
                                     fft_complex *output,
                                     const uint32_t *permutation,
                                     const fft_complex *tables,
                                     const fft_complex *level5_factor,
                                     const fft_complex *level6_factor);
/*
 * All counts are multiples of four complex samples.  data and each coefficient
 * pointer are already advanced to the first butterfly in the requested range.
 */
void tangent_x86_normal(fft_complex *data,
                        size_t quarter,
                        const fft_complex *factor,
                        size_t count);

void tangent_x86_s_low(fft_complex *data,
                       size_t quarter,
                       const float *value,
                       size_t count);
void tangent_x86_s_high(fft_complex *data,
                        size_t quarter,
                        const float *value,
                        size_t count);

void tangent_x86_s2_low(fft_complex *data,
                        size_t quarter,
                        const float *value,
                        const float *low_scale,
                        const float *high_scale,
                        size_t count);
void tangent_x86_s2_high(fft_complex *data,
                         size_t quarter,
                         const float *value,
                         const float *low_scale,
                         const float *high_scale,
                         size_t count);

void tangent_x86_s4_low(fft_complex *data,
                        size_t quarter,
                        const float *value,
                        const float *scale0,
                        const float *scale1,
                        const float *scale2,
                        const float *scale3,
                        size_t count);
void tangent_x86_s4_high(fft_complex *data,
                         size_t quarter,
                         const float *value,
                         const float *scale0,
                         const float *scale1,
                         const float *scale2,
                         const float *scale3,
                         size_t count);

/*
 * Batched small-node kernels. Each offset is measured in complex samples from
 * data. Coefficients are shared by every node in a batch and remain resident
 * in vector registers.
 */
void tangent_x86_batch_unscaled_q1(fft_complex *data,
                                   const uint32_t *offsets,
                                   size_t node_count);
void tangent_x86_batch_unscaled_q2(fft_complex *data,
                                   const uint32_t *offsets,
                                   size_t node_count,
                                   const fft_complex *factor);
void tangent_x86_batch_unscaled_q4(fft_complex *data,
                                   const uint32_t *offsets,
                                   size_t node_count,
                                   const fft_complex *factor);

void tangent_x86_batch_s2_q1(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const float *low_scale,
                             const float *high_scale);
void tangent_x86_batch_s2_q2(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const fft_complex *factor,
                             const float *low_scale,
                             const float *high_scale);
void tangent_x86_batch_s2_q4(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const fft_complex *factor,
                             const float *low_scale,
                             const float *high_scale);

void tangent_x86_batch_s4_q1(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const float *scale0,
                             const float *scale1,
                             const float *scale2,
                             const float *scale3);
void tangent_x86_batch_s4_q2(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const fft_complex *factor,
                             const float *scale0,
                             const float *scale1,
                             const float *scale2,
                             const float *scale3);
void tangent_x86_batch_s4_q4(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             const fft_complex *factor,
                             const float *scale0,
                             const float *scale1,
                             const float *scale2,
                             const float *scale3);

void tangent_x86_batch_unscaled_qn(fft_complex *data,
                                   const uint32_t *offsets,
                                   size_t node_count,
                                   size_t quarter,
                                   const fft_complex *factor);
void tangent_x86_batch_tangent_qn(fft_complex *data,
                                  const uint32_t *offsets,
                                  size_t node_count,
                                  size_t quarter,
                                  const fft_complex *factor);
void tangent_x86_batch_s2_qn(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             size_t quarter,
                             const fft_complex *factor,
                             const fft_complex *low_scale,
                             const fft_complex *high_scale);
void tangent_x86_batch_s4_qn(fft_complex *data,
                             const uint32_t *offsets,
                             size_t node_count,
                             size_t quarter,
                             const fft_complex *factor,
                             const fft_complex *scale0,
                             const fft_complex *scale1,
                             const fft_complex *scale2,
                             const fft_complex *scale3);

#endif
