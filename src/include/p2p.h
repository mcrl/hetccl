/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <hip/hip_runtime.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#define NCCL_P2P_HANDLE_TYPE taccMemHandleTypePosixFileDescriptor

typedef struct {
  int data; // Currently only support an fd based descriptor
} ncclCuDesc;

typedef union {
  // Legacy CUDA IPC
  taccIpcMemHandle_t devIpc;
  // cuMem API support
  ncclCuDesc cuDesc;
} ncclIpcDesc;

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, ncclIpcDesc *ipcDesc, void **ptr);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int tpPeer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr);

#endif
