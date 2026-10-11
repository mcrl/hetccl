// gpu_backend.cpp - Runtime backend selector
#include "gpu_backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Function pointers for backend implementation
static void* (*backend_malloc)(size_t) = nullptr;
static void  (*backend_free)(void*) = nullptr;
static void  (*backend_memset)(void*, int, size_t) = nullptr;

// Declare external backend implementations
extern "C" {
    void* cuda_malloc_impl(size_t nbytes);
    void  cuda_free_impl(void* ptr);
    void  cuda_memset_impl(void* ptr, int val, size_t nbytes);
    
    void* hip_malloc_impl(size_t nbytes);
    void  hip_free_impl(void* ptr);
    void  hip_memset_impl(void* ptr, int val, size_t nbytes);
}

void gpu_backend_init(const char* platform) {
    if (strcmp(platform, "cuda") == 0) {
        backend_malloc = cuda_malloc_impl;
        backend_free = cuda_free_impl;
        backend_memset = cuda_memset_impl;
    } else if (strcmp(platform, "hip") == 0) {
        backend_malloc = hip_malloc_impl;
        backend_free = hip_free_impl;
        backend_memset = hip_memset_impl;
    } else {
        fprintf(stderr, "Unknown platform: %s\n", platform);
        exit(1);
    }
}

void* gpu_malloc(size_t nbytes) {
    return backend_malloc(nbytes);
}

void gpu_free(void* ptr) {
    backend_free(ptr);
}

void gpu_memset(void* ptr, int val, size_t nbytes) {
    backend_memset(ptr, val, nbytes);
}
