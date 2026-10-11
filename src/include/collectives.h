/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

enum ncclDevRedOp_t {
  ncclDevSum, ncclDevProd, ncclDevMax, ncclDevMin,
  ncclDevPreMulSum, ncclDevSumPostDiv,
  ncclNumDevRedOps
};
struct ncclDevRedOpFull {
  ncclDevRedOp_t op;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

#define HIP_FUNC_INDEX_P2P (ncclNumTypes+NCCL_NUM_FUNCTIONS*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS*ncclNumTypes*ncclNumDevRedOps)
#define HIP_FUNC_INDEX_ALLTOALL_PIVOT (HIP_FUNC_INDEX_P2P+1)
#if defined(BUILD_ALLREDUCE_ONLY)
// Only AllReduce, Sum, float
#define HIP_FUNC_INDEX(func, devredop, ncclType, al, pr) ((((((func - ncclFuncAllReduce)*1 + (devredop - ncclDevSum))*1) + (ncclType - ncclFloat))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#else
#define HIP_FUNC_INDEX(func, devredop, ncclType, al, pr) ((((((func)*ncclNumDevRedOps + (devredop))*ncclNumTypes) + (ncclType))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))
#endif

#define SINGLE_ARG(...) __VA_ARGS__
#define CONCAT(a,b) a##b
#define MACRO_IF(cond, t, f) CONCAT(MACRO_IF_, cond)(SINGLE_ARG(t), SINGLE_ARG(f))
#define MACRO_IF_0(t, f) f
#define MACRO_IF_1(t, f) t

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above
#define ALLTOALL_PIVOT_SLICESTEPS 2
#define ALLTOALL_PIVOT_CHUNKSTEPS 4

// We can't use the enum identifiers like ncclSum, ncclFloat, etc since this
// macro will be used in preprocessor conditionals where enums have no meaning.
#define NCCL_NVLS_SUPPORTS(/*ncclDataType_t*/ type, /*ncclDevRedOp_t*/ red) \
  (((type==2 || type==3) && (red==0 || red==2 || red==3)) || \
   ((type==4 || type==5) && (red==0 || red==2 || red==3)) || \
   ((type==6 || type==9) && (red==0 || red==2 || red==3)) || \
   (type==7 && red==0) || \
   (type==8 && red==0))

#endif
