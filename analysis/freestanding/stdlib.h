#ifndef TANGENT_FFT_FREESTANDING_STDLIB_H
#define TANGENT_FFT_FREESTANDING_STDLIB_H

#include <stddef.h>

void *malloc(size_t size);
void *calloc(size_t count, size_t size);
void *aligned_alloc(size_t alignment, size_t size);
void free(void *pointer);

#endif
