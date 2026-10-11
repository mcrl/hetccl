#include "devcomm.h"
#include "collectives_private.h"

// Interfaces from HETCCL
#include "ext_plugin.h"
#include "tacc.h"
#include "msccl/msccl_kernel.h"

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NULL() \
  nullptr, \
  nullptr, \
  nullptr

#define MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, type, fullOps) \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL, fullOps), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, LL128, fullOps), \
  (void *)MSCCL_KERNEL_ENTRY_NAME(devredop, type, Simple, fullOps)

#define MSCCL_KERNEL_ENTRY_DEVREDOP(devredop, fullOps) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, half, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, float, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, double, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, rccl_bfloat16, fullOps)

#define MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(devredop, fullOps) \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint8_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint32_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, int64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_TYPE(devredop, uint64_t, fullOps), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL(), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NULL()

#define MSCCL_KERNEL_ENTRY() \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Sum, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Prod, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Min, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(Max, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP(PreMulSum, false), \
  MSCCL_KERNEL_ENTRY_DEVREDOP_NOFLOAT(SumPostDiv, false)

void* mscclKernelEntries[ncclNumDevRedOps * ncclNumTypes * NCCL_NUM_PROTOCOLS] = {
#ifdef COMPILE_MSCCL_KERNEL
  MSCCL_KERNEL_ENTRY()
#endif
};

static ncclResult_t hetcclHipFuncGetAttributesByIndex(taccFuncAttributes* attr, int index) {
  taccError_t err = taccFuncGetAttributes(attr, mscclKernelEntries[index]);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t hetcclHipFuncSetAttributeByIndex(int index, taccFuncAttribute attr, int value) {
  taccError_t err = taccFuncSetAttribute(mscclKernelEntries[index], attr, value);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t hetcclHipLaunchKernelByIndex(int index, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream) {
  taccError_t err = taccLaunchKernel(mscclKernelEntries[index], gridDim, blockDim, args, sharedMem, stream);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t hetcclHipExtLaunchKernelByIndex(int index, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags) {
  taccError_t err = taccExtLaunchKernel(mscclKernelEntries[index], numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
  if (err != taccSuccess) {
    return ncclInternalError;
  }
  return ncclSuccess;
}

hetcclPlugin_t hetcclPluginHipMsccl = {
  .kernsCount = sizeof(mscclKernelEntries)/sizeof(void *),
  .funcGetAttributesByIndex = hetcclHipFuncGetAttributesByIndex,
  .funcSetAttributeByIndex = hetcclHipFuncSetAttributeByIndex,
  .launchKernelByIndex = hetcclHipLaunchKernelByIndex,
  .extLaunchKernelByIndex = hetcclHipExtLaunchKernelByIndex
};