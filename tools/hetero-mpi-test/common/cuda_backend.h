#include <cstddef>
#pragma once

void* gpu_malloc(size_t nbytes);
void  gpu_free(void* ptr);
void  gpu_memset(void* ptr, int val, size_t nbytes);