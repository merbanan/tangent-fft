#include "lane8_avx_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void lane8_avx_base4_fma3(const fft_complex *, const uint32_t *,
                          lane8_avx_row *, size_t);
void lane8_avx_base8_fma3(const fft_complex *, const uint32_t *,
                          lane8_avx_row *, size_t);
void lane8_avx_stage_fma3(lane8_avx_row *, size_t, size_t,
                          const lane8_avx_root *);
void lane8_avx_finish_fma3(lane8_avx_row *, fft_complex *, size_t,
                           const float *, const float *);
void lane8_avx_finish8_fma3(lane8_avx_row *, fft_complex *, size_t,
                            const float *, const float *);

static uint64_t ticks(void)
{
    unsigned lo;
    unsigned hi;
    __asm__ volatile("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) : : "memory");
    return ((uint64_t)hi << 32) | lo;
}

int main(void)
{
    static const size_t sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    size_t s;
    puts("N       base       stages     finish      total");
    for (s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        const size_t n = sizes[s];
        lane8_avx_plan *p = lane8_avx_plan_create(n);
        fft_complex *data = aligned_alloc(64, n * sizeof(*data));
        const unsigned repeats = n < 1024 ? 10000 : 2000;
        uint64_t base_cycles = 0;
        uint64_t stage_cycles = 0;
        uint64_t finish_cycles = 0;
        unsigned r;
        size_t i;

        for (i = 0; i < n; ++i) {
            data[i].re = (float)((i * 17 + 3) % 31) / 31.0f;
            data[i].im = (float)((i * 11 + 7) % 29) / 29.0f;
        }
        for (r = 0; r < repeats; ++r) {
            uint64_t begin = ticks();
            if ((p->inner_levels & 1U) != 0) {
                lane8_avx_base8_fma3(data, p->permutation,
                                     p->work, p->inner_size);
            } else {
                lane8_avx_base4_fma3(data, p->permutation,
                                     p->work, p->inner_size);
            }
            base_cycles += ticks() - begin;
        }
        {
            size_t previous = (p->inner_levels & 1U) != 0 ? 8 : 4;
            const lane8_avx_root *root = p->replicated_root;
            while (previous < p->inner_size) {
                for (r = 0; r < repeats; ++r) {
                    const uint64_t begin = ticks();
                    lane8_avx_stage_fma3(
                        p->work, previous, p->inner_size, root);
                    stage_cycles += ticks() - begin;
                }
                root += 3 * (previous - 1);
                previous *= 4;
            }
        }
        for (r = 0; r < repeats; ++r) {
            const uint64_t begin = ticks();
            if (p->inner_size == 4) {
                lane8_avx_finish_fma3(
                    p->work, data, p->inner_size,
                    p->finish_re, p->finish_im);
            } else {
                lane8_avx_finish8_fma3(
                    p->work, data, p->inner_size,
                    p->finish_re, p->finish_im);
            }
            finish_cycles += ticks() - begin;
        }
        printf("%-5zu %10.1f %10.1f %10.1f %10.1f\n",
               n,
               (double)base_cycles / repeats,
               (double)stage_cycles / repeats,
               (double)finish_cycles / repeats,
               (double)(base_cycles + stage_cycles + finish_cycles) / repeats);
        free(data);
        lane8_avx_plan_destroy(p);
    }
    return 0;
}
