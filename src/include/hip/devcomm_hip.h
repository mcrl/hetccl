#pragma once

#include "devcomm.h"
#include <hip/hip_runtime.h>

#define HETCCL_HIP_MAX_NTHREADS 512

// Number of named barriers supported by CUDA
#define HETCCL_HIP_MAX_GROUPS (HETCCL_HIP_MAX_NTHREADS/HIP_WARP_SIZE)

// The amount of dynamic shmem per block
constexpr int hetcclHipShmemDynamicSize(int cudaArch = NCCL_CUDA_ARCH) {
  return 0;
}