#pragma once

#include "cuda/collectives_cuda.h"

#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

/* Declare all collective operations */
#define CUDA_DECL5(func, algo, proto, devredop, type) \
  extern __device__ void HETCCL_CUDA_FUNC_NAME(func, algo, proto, devredop, type)(); \
  extern __global__ void HETCCL_CUDA_KERN_NAME(func, algo, proto, devredop, type)(struct hetcclCudaDevComm* comm, uint64_t channelMask, struct hetcclCudaWork* workHead); \

#define CUDA_DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, CUDA_DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, CUDA_DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, CUDA_DECL5(func, algo, LL128,  devredop, type))

#define CUDA_DECL3(func, devredop, type, undef) \
  CUDA_DECL4(func, RING,           devredop, type, undef) \
  CUDA_DECL4(func, TREE,           devredop, type, undef) \
  CUDA_DECL4(func, COLLNET_DIRECT, devredop, type, undef) \
  CUDA_DECL4(func, COLLNET_CHAIN,  devredop, type, undef) \
  CUDA_DECL4(func, NVLS,           devredop, type, undef) \
  CUDA_DECL4(func, NVLS_TREE,      devredop, type, undef)

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define CUDA_DECL2(func, devredop, undefForFloat) \
  CUDA_DECL3(func, devredop, int8_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, int32_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, int64_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  CUDA_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  CUDA_DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  CUDA_DECL3(func, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#else
#define CUDA_DECL2(func, devredop, undefForFloat) \
  CUDA_DECL3(func, devredop, int8_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, int32_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, int64_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  CUDA_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  CUDA_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  CUDA_DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define CUDA_DECL(func) \
  CUDA_DECL2(func, Sum, /*undefForFloat=*/0) \
  CUDA_DECL2(func, Prod, /*undefForFloat=*/0) \
  CUDA_DECL2(func, Min, /*undefForFloat=*/0) \
  CUDA_DECL2(func, Max, /*undefForFloat=*/0) \
  CUDA_DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  CUDA_DECL2(func, SumPostDiv, /*undefForFloat=*/1)

CUDA_DECL2(Broadcast, Sum, /*undefForFloat=*/0)
CUDA_DECL(Reduce)
CUDA_DECL2(AllGather, Sum, /*undefForFloat=*/0)
CUDA_DECL(ReduceScatter)
CUDA_DECL(AllReduce)
CUDA_DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)

extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, half)();
#if defined(__CUDA_BF16_TYPES_EXIST__)
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, __nv_bfloat16)();
#endif
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void HETCCL_CUDA_ONERANK_REDUCE_NAME(PreMulSum, double)();