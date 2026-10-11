#include "devcomm.h"
#include "collectives_private.h"

// Interfaces from SCCL
#include "ext_plugin.h"
#include "tacc.h"

static const ncclKernelMatch hipKerns[] = {
  {(void *)NCCL_KERN_NAME(SendRecv, RING, SIMPLE, Sum, int8_t), true},
  {(void *)NCCL_KERN_NAME_DEBUG(SendRecv, RING, SIMPLE, Sum, int8_t), true},
};
static const int hipKernsCount = 2;

static ncclResult_t scclHipFuncGetAttributesByIndex(taccFuncAttributes* attr, int index) {
  taccError_t err = taccFuncGetAttributes(attr, hipKerns[index].kernelFn);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t scclHipFuncSetAttributeByIndex(int index, taccFuncAttribute attr, int value) {
  taccError_t err = taccFuncSetAttribute(hipKerns[index].kernelFn, attr, value);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t scclHipGetSpecializedByIndex(int index, bool* specialized) {
  *specialized = hipKerns[index].specialized;
  return ncclSuccess;
}

static ncclResult_t scclHipLaunchKernelByIndex(int index, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream) {
  taccError_t err = taccLaunchKernel(hipKerns[index].kernelFn, gridDim, blockDim, args, sharedMem, stream);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t scclHipExtLaunchKernelByIndex(int index, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags) {
  taccError_t err = taccExtLaunchKernel(hipKerns[index].kernelFn, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

__attribute__ ((visibility("default"))) scclPlugin_t scclPluginHip = {
  .kernsCount = hipKernsCount,
  .funcGetAttributesByIndex = scclHipFuncGetAttributesByIndex,
  .funcSetAttributeByIndex = scclHipFuncSetAttributeByIndex,
  .getSpecializedByIndex = scclHipGetSpecializedByIndex,
  .launchKernelByIndex = scclHipLaunchKernelByIndex,
  .extLaunchKernelByIndex = scclHipExtLaunchKernelByIndex
};