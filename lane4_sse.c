#include "lane4_portable_internal.h"

#include <xmmintrin.h>

#ifndef LANE4_SSE_EXECUTE
#error "LANE4_SSE_EXECUTE must name this ISA-specific entry point"
#endif

static inline void store_complex4(fft_complex *output,
                                  __m128 re,
                                  __m128 im)
{
    _mm_storeu_ps((float *)output, _mm_unpacklo_ps(re, im));
    _mm_storeu_ps((float *)(output + 2), _mm_unpackhi_ps(re, im));
}

static inline void multiply_vectors(__m128 value_re,
                                    __m128 value_im,
                                    __m128 factor_re,
                                    __m128 factor_im,
                                    __m128 *output_re,
                                    __m128 *output_im)
{
    *output_re = _mm_sub_ps(
        _mm_mul_ps(value_re, factor_re),
        _mm_mul_ps(value_im, factor_im));
    *output_im = _mm_add_ps(
        _mm_mul_ps(value_re, factor_im),
        _mm_mul_ps(value_im, factor_re));
}

static inline void finish4(lane4_portable_plan *plan,
                           fft_complex *data,
                           size_t frequency)
{
    const size_t inner_size = plan->inner_size;
    __m128 re0 = _mm_load_ps(plan->work[frequency].re);
    __m128 re1 = _mm_load_ps(plan->work[frequency + 1].re);
    __m128 re2 = _mm_load_ps(plan->work[frequency + 2].re);
    __m128 re3 = _mm_load_ps(plan->work[frequency + 3].re);
    __m128 im0 = _mm_load_ps(plan->work[frequency].im);
    __m128 im1 = _mm_load_ps(plan->work[frequency + 1].im);
    __m128 im2 = _mm_load_ps(plan->work[frequency + 2].im);
    __m128 im3 = _mm_load_ps(plan->work[frequency + 3].im);
    __m128 ac_re;
    __m128 ac_im;
    __m128 ac_difference_re;
    __m128 ac_difference_im;
    __m128 bd_re;
    __m128 bd_im;
    __m128 difference_re;
    __m128 difference_im;
    __m128 output0_re;
    __m128 output0_im;
    __m128 output1_re;
    __m128 output1_im;
    __m128 output2_re;
    __m128 output2_im;
    __m128 output3_re;
    __m128 output3_im;

    _MM_TRANSPOSE4_PS(re0, re1, re2, re3);
    _MM_TRANSPOSE4_PS(im0, im1, im2, im3);

    multiply_vectors(
        re1, im1,
        _mm_load_ps(plan->finish_re + frequency),
        _mm_load_ps(plan->finish_im + frequency),
        &re1, &im1);
    multiply_vectors(
        re2, im2,
        _mm_load_ps(plan->finish_re + inner_size + frequency),
        _mm_load_ps(plan->finish_im + inner_size + frequency),
        &re2, &im2);
    multiply_vectors(
        re3, im3,
        _mm_load_ps(plan->finish_re + 2 * inner_size + frequency),
        _mm_load_ps(plan->finish_im + 2 * inner_size + frequency),
        &re3, &im3);

    ac_re = _mm_add_ps(re0, re2);
    ac_im = _mm_add_ps(im0, im2);
    ac_difference_re = _mm_sub_ps(re0, re2);
    ac_difference_im = _mm_sub_ps(im0, im2);
    bd_re = _mm_add_ps(re1, re3);
    bd_im = _mm_add_ps(im1, im3);
    difference_re = _mm_sub_ps(re1, re3);
    difference_im = _mm_sub_ps(im1, im3);

    output0_re = _mm_add_ps(ac_re, bd_re);
    output0_im = _mm_add_ps(ac_im, bd_im);
    output1_re = _mm_add_ps(ac_difference_re, difference_im);
    output1_im = _mm_sub_ps(ac_difference_im, difference_re);
    output2_re = _mm_sub_ps(ac_re, bd_re);
    output2_im = _mm_sub_ps(ac_im, bd_im);
    output3_re = _mm_sub_ps(ac_difference_re, difference_im);
    output3_im = _mm_add_ps(ac_difference_im, difference_re);

    store_complex4(data + frequency, output0_re, output0_im);
    store_complex4(
        data + inner_size + frequency, output1_re, output1_im);
    store_complex4(
        data + 2 * inner_size + frequency, output2_re, output2_im);
    store_complex4(
        data + 3 * inner_size + frequency, output3_re, output3_im);
}

int LANE4_SSE_EXECUTE(lane4_portable_plan *plan, fft_complex *data)
{
    size_t i;
    size_t length;

    if (plan == NULL || data == NULL) {
        return -1;
    }

    for (i = 0; i < plan->inner_size; ++i) {
        const float *input =
            (const float *)(data + plan->permutation[i]);
        const __m128 low = _mm_loadu_ps(input);
        const __m128 high = _mm_loadu_ps(input + 4);
        const __m128 re = _mm_shuffle_ps(
            low, high, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 im = _mm_shuffle_ps(
            low, high, _MM_SHUFFLE(3, 1, 3, 1));
        _mm_store_ps(plan->work[i].re, re);
        _mm_store_ps(plan->work[i].im, im);
    }

    for (length = 2; length <= plan->inner_size; length *= 2) {
        const size_t half = length / 2;
        const size_t root_stride = plan->inner_size / length;
        size_t start;

        for (start = 0; start < plan->inner_size; start += length) {
            size_t k;
            for (k = 0; k < half; ++k) {
                lane4_portable_row *a = plan->work + start + k;
                lane4_portable_row *b = a + half;
                const fft_complex factor = plan->root[k * root_stride];
                const __m128 a_re = _mm_load_ps(a->re);
                const __m128 a_im = _mm_load_ps(a->im);
                const __m128 source_re = _mm_load_ps(b->re);
                const __m128 source_im = _mm_load_ps(b->im);
                __m128 b_re;
                __m128 b_im;

                multiply_vectors(
                    source_re, source_im,
                    _mm_set1_ps(factor.re),
                    _mm_set1_ps(factor.im),
                    &b_re, &b_im);
                _mm_store_ps(a->re, _mm_add_ps(a_re, b_re));
                _mm_store_ps(a->im, _mm_add_ps(a_im, b_im));
                _mm_store_ps(b->re, _mm_sub_ps(a_re, b_re));
                _mm_store_ps(b->im, _mm_sub_ps(a_im, b_im));
            }
        }
    }

    for (i = 0; i < plan->inner_size; i += 4) {
        finish4(plan, data, i);
    }
    return 0;
}
