#include "cuda_backend.h"
#include <cuda_runtime.h>

void* gpu_malloc(size_t nbytes) {
    void* p; cudaMalloc(&p, nbytes); return p;
}

void gpu_free(void* ptr) {
    cudaFree(ptr);
}

void gpu_memset(void* ptr, int val, size_t nbytes) {
    cudaMemset(ptr, val, nbytes);
}