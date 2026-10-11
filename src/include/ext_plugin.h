#pragma once

#include <cstddef>
#include "tacc.h"
#include "sccl.h"

struct ncclKernelMatch {
  void* kernelFn;
  bool specialized;
};

typedef struct {
  // common
  int kernsCount;
  ncclResult_t (*funcGetLocalSizeBytesByIndex)(size_t* localSizeBytes, int index);
  ncclResult_t (*funcGetAttributesByIndex)(taccFuncAttributes* attr, int index);
  ncclResult_t (*funcSetAttributeByIndex)(int index, taccFuncAttribute attr, int value);
  ncclResult_t (*getSpecializedByIndex)(int index, bool* specialized);
  ncclResult_t (*launchKernelByIndex)(int index, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream);
  ncclResult_t (*extLaunchKernelByIndex)(int index, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags);
} scclPlugin_t;