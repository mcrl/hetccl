/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);

ncclResult_t
ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle)
{
	return ncclSuccess;
}

ncclResult_t
ncclCommDeregister(const ncclComm_t comm, void* handle)
{
	return ncclSuccess;
}
