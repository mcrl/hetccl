#pragma once
#include <cstddef>

// Unified GPU backend interface
// Each rank can use different backend (CUDA or HIP)

void* gpu_malloc(size_t nbytes);
void  gpu_free(void* ptr);
void  gpu_memset(void* ptr, int val, size_t nbytes);

// Initialize backend based on platform selection
void gpu_backend_init(const char* platform); // "cuda" or "hip"
