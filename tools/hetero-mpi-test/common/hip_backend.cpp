#include "hip_backend.h"
#include <hip/hip_runtime.h>

void* gpu_malloc(size_t nbytes) {
    void* p; hipMalloc(&p, nbytes); return p;
}

void gpu_free(void* ptr) {
    hipFree(ptr);
}

void gpu_memset(void* ptr, int val, size_t nbytes) {
    hipMemset(ptr, val, nbytes);
}