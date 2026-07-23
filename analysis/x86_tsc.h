#ifndef TANGENT_FFT_ANALYSIS_X86_TSC_H
#define TANGENT_FFT_ANALYSIS_X86_TSC_H

#include <stdint.h>

uint64_t x86_tsc_start(void);
uint64_t x86_tsc_end(void);

#endif
