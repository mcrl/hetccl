#include "cuda/devcomm_cuda.h"
#include "collectives_private.h"

// Interfaces from HETCCL
#include "ext_plugin.h"
#include "tacc.h"

// Only generate inline kernels for LL
#define NCCL_FUNC5(func, algo, devredop, dtype, specialized) \
  /*LL    */{(void*)HETCCL_CUDA_KERN_NAME(func, algo, LL, devredop, dtype), true && specialized}, \
  /*LL128 */{(void*)HETCCL_CUDA_KERN_NAME(func, algo, LL, devredop, dtype), false && specialized}, \
  /*SIMPLE*/{(void*)HETCCL_CUDA_KERN_NAME(func, algo, LL, devredop, dtype), false && specialized}

#define NCCL_FUNC4(func, devredop, type, specialized) \
  NCCL_FUNC5(func, TREE,           devredop, type, specialized), \
  NCCL_FUNC5(func, RING,           devredop, type, specialized), \
  NCCL_FUNC5(func, COLLNET_DIRECT, devredop, type, specialized), \
  NCCL_FUNC5(func, COLLNET_CHAIN,  devredop, type, specialized), \
  NCCL_FUNC5(func, NVLS,           devredop, type, specialized), \
  NCCL_FUNC5(func, NVLS_TREE,      devredop, type, specialized)

#ifdef __CUDA_BF16_TYPES_EXIST__
  #define HAVE_BFLOAT16 1
#else
  #define HAVE_BFLOAT16 0
#endif

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3(func, devredop, reduction, specialized) \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int8_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint8_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int32_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint32_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, int64_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, uint64_t, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, half, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, float, int8_t), specialized), \
  NCCL_FUNC4(func, devredop, MACRO_IF(reduction, double, int8_t), specialized) \
  MACRO_IF(HAVE_BFLOAT16, \
    SINGLE_ARG(, NCCL_FUNC4(func, devredop, MACRO_IF(reduction, __nv_bfloat16, int8_t), specialized)), \
    /*nothing*/ \
  )

// Must be consistent with ncclDevRedOp_t -- but we only generate kernel for sums.
#define NCCL_FUNCS2(func, reduction) \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/1), /*Sum*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Prod*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Max*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*Min*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0), /*PreMulSum*/ \
  NCCL_FUNCS3(func, Sum, reduction, /*specialized=*/0)  /*SumPostDiv*/

// Must be consistent with the ncclFuncSet enum
#if defined(BUILD_ALLREDUCE_ONLY)
static const ncclKernelMatch cudaKerns[] = {
  NCCL_FUNC4(AllReduce, Sum, MACRO_IF(/*reduction=*/1, float, int8_t), /*specialized=*/1)
};

static const int cudaKernsCount = NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS;
#else
static const ncclKernelMatch cudaKerns[] = {
  {(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), true},
  // We don't bake special kernels for the one-rank reductions
  {/*int8*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint8*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*int32*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint32*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*int64*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*uint64*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*half*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*float*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  {/*double*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  #if HAVE_BFLOAT16
    {/*bfloat16*/(void*)HETCCL_CUDA_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), false},
  #endif
  NCCL_FUNCS2(Broadcast, /*reduction=*/0),
  NCCL_FUNCS2(Reduce, /*reduction=*/1),
  NCCL_FUNCS2(AllGather, /*reduction=*/0),
  NCCL_FUNCS2(ReduceScatter, /*reduction=*/1),
  NCCL_FUNCS2(AllReduce, /*reduction=*/1)
};

static const int cudaKernsCount = 1+ncclNumTypes+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS;
#endif

// Helper functions to prevent cross-compilation-unit kernel function references
// Note: Somehow cuda runtime APIs that accept kernel function pointer does not work across shared library boundary
//                so we need to wrap them with these functions. (copied from tacc)
ncclResult_t hetcclCudaFuncGetLocalSizeBytesByIndex(size_t* localSizeBytes, int index) {
  cudaFuncAttributes attr;
  taccError_t err = (taccError_t)cudaFuncGetAttributes(&attr, cudaKerns[index].kernelFn);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  *localSizeBytes = attr.localSizeBytes;
  return ncclSuccess;
}

ncclResult_t hetcclCudaFuncSetAttributeByIndex(int index, taccFuncAttribute attr, int value) {
  taccError_t err = (taccError_t)cudaFuncSetAttribute(cudaKerns[index].kernelFn, (cudaFuncAttribute)attr, value);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t hetcclCudaGetSpecializedByIndex(int index, bool* specialized) {
  *specialized = cudaKerns[index].specialized;
  return ncclSuccess;
}

ncclResult_t hetcclCudaLaunchKernelByIndex(int index, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream) {
  dim3 gridDim_;
  dim3 blockDim_;
  memcpy(&gridDim_, &gridDim, sizeof(gridDim_));
  memcpy(&blockDim_, &blockDim, sizeof(blockDim_));
  taccError_t err = (taccError_t)cudaLaunchKernel(cudaKerns[index].kernelFn, gridDim_, blockDim_, args, sharedMem, (cudaStream_t) stream);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

hetcclPlugin_t hetcclPluginCuda = {
  .kernsCount = cudaKernsCount,
  .funcGetLocalSizeBytesByIndex = hetcclCudaFuncGetLocalSizeBytesByIndex,
  .funcSetAttributeByIndex = hetcclCudaFuncSetAttributeByIndex,
  .getSpecializedByIndex = hetcclCudaGetSpecializedByIndex,
  .launchKernelByIndex = hetcclCudaLaunchKernelByIndex
};
