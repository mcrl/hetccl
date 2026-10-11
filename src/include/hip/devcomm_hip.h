#pragma once

#include "devcomm.h"
#include <hip/hip_runtime.h>

#define SCCL_HIP_MAX_NTHREADS 512

// Number of named barriers supported by CUDA
#define SCCL_HIP_MAX_GROUPS (SCCL_HIP_MAX_NTHREADS/HIP_WARP_SIZE)

// The amount of dynamic shmem per block
constexpr int scclHipShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return 0;
}