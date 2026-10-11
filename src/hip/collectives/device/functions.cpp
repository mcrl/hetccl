#include "hip/hip_runtime.h"
/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hip/devcomm_hip.h"
#include "collectives_private.h"
#include "common.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[scclHipShmemScratchWarpSize()*(SCCL_HIP_MAX_NTHREADS/HIP_WARP_SIZE)/sizeof(ulong2)];
#endif

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
