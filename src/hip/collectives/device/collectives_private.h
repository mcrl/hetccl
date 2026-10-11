#pragma once

#include "collectives.h"

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ncclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ncclFunction_OneRankReduce_##devredop##_##type

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  ncclKernel_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_KERN_NAME_DEBUG(func, algo, proto, devredop, type) \
  ncclKernelDebug_##func##_##algo##_##proto##_##devredop##_##type

#define NCCL_IMPL_NAME(func, algo, proto) \
  nccl##func##algo##proto

/* Declare all collective operations */
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead); \
  extern __global__ void NCCL_KERN_NAME_DEBUG(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead);
#else
#define DECL5(func, algo, proto, devredop, type) \
  extern __device__ __attribute__((noinline)) void NCCL_FUNC_NAME(func, algo, proto, devredop, type)(); \
  extern __global__ void NCCL_KERN_NAME(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead); \
  extern __global__ void NCCL_KERN_NAME_DEBUG(func, algo, proto, devredop, type)(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead);
#endif

#define DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, DECL5(func, algo, LL128,  devredop, type))

#define DECL3(func, devredop, type, undef) \
  DECL4(func, RING,           devredop, type, undef) \
  DECL4(func, TREE,           devredop, type, undef) \
  DECL4(func, COLLNET_DIRECT, devredop, type, undef) \
  DECL4(func, COLLNET_CHAIN,  devredop, type, undef) \
  DECL4(func, NVLS,           devredop, type, undef) \
  DECL4(func, NVLS_TREE,      devredop, type, undef)

#if defined(RCCL_BFLOAT16)
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  DECL3(func, devredop, rccl_bfloat16, /*undef=*/undefForFloat)
#else
#define DECL2(func, devredop, undefForFloat) \
  DECL3(func, devredop, int8_t, /*undef=*/0) \
  DECL3(func, devredop, uint8_t, /*undef=*/0) \
  DECL3(func, devredop, int32_t, /*undef=*/0) \
  DECL3(func, devredop, uint32_t, /*undef=*/0) \
  DECL3(func, devredop, int64_t, /*undef=*/0) \
  DECL3(func, devredop, uint64_t, /*undef=*/0) \
  DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define DECL(func) \
  DECL2(func, Sum, /*undefForFloat=*/0) \
  DECL2(func, Prod, /*undefForFloat=*/0) \
  DECL2(func, Min, /*undefForFloat=*/0) \
  DECL2(func, Max, /*undefForFloat=*/0) \
  DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  DECL2(func, SumPostDiv, /*undefForFloat=*/1)

DECL2(Broadcast, Sum, /*undefForFloat=*/0)
DECL(Reduce)
DECL2(AllGather, Sum, /*undefForFloat=*/0)
DECL(ReduceScatter)
DECL(AllReduce)
DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)
DECL5(AllToAllPivot, RING, SIMPLE, Sum, int8_t)

extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, half)();
#if defined(RCCL_BFLOAT16)
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, rccl_bfloat16)();
#endif
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void NCCL_ONERANK_REDUCE_NAME(PreMulSum, double)();