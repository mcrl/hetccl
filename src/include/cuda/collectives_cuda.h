#pragma once

#include "collectives.h"

#define CUDA_FUNC_INDEX_P2P 0
#if defined(BUILD_ALLREDUCE_ONLY)
#define CUDA_FUNC_INDEX(func, devredop, ncclType, al, pr) ((((((func - ncclFuncAllReduce)*1 + (devredop - ncclDevSum))*1) + (ncclType - ncclFloat))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#else
#define CUDA_FUNC_INDEX(func, devredop, ncclType, al, pr) (1+ncclNumTypes+(((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#endif

#define HETCCL_CUDA_FUNC_NAME(func, algo, proto, devredop, type) \
  hetcclCudaFunction_##func##_##algo##_##proto##_##devredop##_##type

#define HETCCL_CUDA_ONERANK_REDUCE_NAME(devredop, type) \
  hetcclCudaFunction_OneRankReduce_##devredop##_##type

#define HETCCL_CUDA_KERN_NAME(func, algo, proto, devredop, type) \
  hetcclCudaKernel_##func##_##algo##_##proto##_##devredop##_##type

#define HETCCL_CUDA_KERN_NAME_DEBUG(func, algo, proto, devredop, type) \
  hetcclCudaKernelDebug_##func##_##algo##_##proto##_##devredop##_##type

#define HETCCL_CUDA_IMPL_NAME(func, algo, proto) \
  hetcclCuda##func##algo##proto
