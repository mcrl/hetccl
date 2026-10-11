/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "sccl.h"
#include "checks.h"
#include "align.h"
#include "utils.h"
#include "p2p.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "rccl_vars.h"

uint64_t clockNano(); // from utils.h with which we have a circular dependency

template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(taccHostAlloc((void**)ptr, nelem*sizeof(T), taccHostMallocMapped), result, finish);
  memset(*ptr, 0, nelem*sizeof(T));
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) WARN("Failed to CUDA host alloc %ld bytes", nelem*sizeof(T));
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(taccFreeHost(ptr));
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  //INFO(NCCL_ALLOC, "%s:%d malloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), p);
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}
#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__)

template <typename T>
ncclResult_t ncclRealloc(T** ptr, size_t oldNelem, size_t nelem) {
  if (nelem < oldNelem) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* oldp = *ptr;
  T* p = (T*)malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memcpy(p, oldp, oldNelem*sizeof(T));
  free(oldp);
  memset(p+oldNelem, 0, (nelem-oldNelem)*sizeof(T));
  *ptr = (T*)p;
  INFO(NCCL_ALLOC, "Mem Realloc old size %ld, new size %ld pointer %p", oldNelem*sizeof(T), nelem*sizeof(T), *ptr);
  return ncclSuccess;
}

struct __attribute__ ((aligned(64))) allocationTracker {
  union {
    struct {
      uint64_t totalAlloc;
      uint64_t totalAllocSize;
    };
    char align[64];
  };
};
static_assert(sizeof(struct allocationTracker) == 64, "allocationTracker must be size of 64 bytes");
#define MAX_ALLOC_TRACK_NGPU 32
extern struct allocationTracker allocTracker[];

#if CUDART_VERSION >= 11030

#include "cudawrap.h"

static inline ncclResult_t ncclCuMemAlloc(void **ptr, hipMemGenericAllocationHandle_t *handlep, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  hipDevice_t currentDev;
  hipMemAllocationProp prop = {};
  hipMemAccessDesc accessDesc = {};
  hipMemGenericAllocationHandle_t handle;
  int cudaDev;
  int flag = 0;
  CUDACHECK(hipGetDevice(&cudaDev));
  CUCHECK(hipDeviceGet(&currentDev, cudaDev));
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.requestedHandleTypes = NCCL_P2P_HANDLE_TYPE; // So it can be exported
  prop.location.id = currentDev;
  // Query device to see if RDMA support is available
  CUCHECK(hipDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
  CUCHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(hipMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(hipMemAddressReserve((hipDeviceptr_t *)ptr, size, 0, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(hipMemMap((hipDeviceptr_t)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory */
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = currentDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  CUCHECK(hipMemSetAccess((hipDeviceptr_t)*ptr, size, &accessDesc, 1));
  if (handlep) *handlep = handle;
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zi pointer %p handle %llx", size, *ptr, handle);
  return result;
}

static inline ncclResult_t ncclCuMemFree(void *ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  hipMemGenericAllocationHandle_t handle;
  size_t size = 0;
  CUCHECK(hipMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(hipMemRelease(handle));
  CUCHECK(hipMemGetAddressRange(NULL, &size, (hipDeviceptr_t)ptr));
  TRACE(NCCL_ALLOC, "CuMem Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(hipMemUnmap((hipDeviceptr_t)ptr, size));
  CUCHECK(hipMemRelease(handle));
  CUCHECK(hipMemAddressFree((hipDeviceptr_t)ptr, size));
  return result;
}

#else

extern int ncclCuMemEnable();

static inline ncclResult_t ncclCuMemAlloc(void **ptr, void *handlep, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void *ptr) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

#endif

template <typename T>
ncclResult_t ncclCudaMallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, bool isFineGrain = false) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  taccPlatform platform;
  TACCCHECK(taccGetPlatform(&platform));
  if (platform == taccPlatformHip) {
    if (isFineGrain) {
      #if defined(HIP_UNCACHED_MEMORY)
        CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocUncached), result, finish);
      #else
        CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocFinegrained), result, finish);
      #endif
    } else {
      CUDACHECKGOTO(taccMalloc((void**)ptr, nelem*sizeof(T)), result, finish);
    }
  } else if (platform == taccPlatformCuda) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, nelem*sizeof(T)), result, finish);
    } else {
      CUDACHECKGOTO(taccMalloc((void**)ptr, nelem*sizeof(T)), result, finish);
    }
  } else {
    WARN("Unknown backend");
    return ncclInternalError;
  }
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) WARN("Failed to CUDA malloc %ld bytes", nelem*sizeof(T));
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define ncclCudaMalloc(...) ncclCudaMallocDebug( __FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocDebug(const char *filefunc, int line, T** ptr, size_t nelem, taccStream_t sideStream = nullptr, bool isFineGrain = false) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  taccStream_t stream = sideStream;
  if (stream == nullptr)
    CUDACHECK(taccStreamCreateWithFlags(&stream, taccStreamNonBlocking));
  taccPlatform platform;
  TACCCHECK(taccGetPlatform(&platform));
  if (platform == taccPlatformHip) {
    if (isFineGrain) {
      #if defined(HIP_UNCACHED_MEMORY)
        CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocUncached), result, finish);
      #else
        CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocFinegrained), result, finish);
      #endif
    } else {
      CUDACHECKGOTO(taccMalloc((void **)ptr, nelem*sizeof(T)), result, finish);
    }
  } else if (platform == taccPlatformCuda) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void **)ptr, NULL, nelem*sizeof(T)), result, finish);
    } else {
      CUDACHECKGOTO(taccMalloc((void**)ptr, nelem*sizeof(T)), result, finish);
    }
  } else {
    WARN("Unknown backend");
    return ncclInternalError;
  }
  CUDACHECKGOTO(taccMemsetAsync(*ptr, 0, nelem*sizeof(T), stream), result, finish);
  CUDACHECKGOTO(taccStreamSynchronize(stream), result, finish);
  if (sideStream == nullptr)
    CUDACHECKGOTO(taccStreamDestroy(stream), result, finish);
  int dev;
  CUDACHECK(taccGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
    __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*sizeof(T), __ATOMIC_RELAXED);
  }
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) WARN("Failed to CUDA calloc %ld bytes", nelem*sizeof(T));
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define ncclCudaCalloc(...) ncclCudaCallocDebug(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(const char *filefunc, int line, T** ptr, size_t nelem, taccStream_t stream, bool isFineGrain = false) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (isFineGrain) {
#if defined(HIP_UNCACHED_MEMORY)
    CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocUncached), result, finish);
#else
    CUDACHECKGOTO(taccExtMallocWithFlags((void**)ptr, nelem*sizeof(T), taccDeviceMallocFinegrained), result, finish);
#endif
  } else
    CUDACHECKGOTO(taccMalloc((void**)ptr, nelem*sizeof(T)), result, finish);
  CUDACHECKGOTO(taccMemsetAsync(*ptr, 0, nelem*sizeof(T), stream), result, finish);
  int dev;
  CUDACHECK(taccGetDevice(&dev));
  if (dev < MAX_ALLOC_TRACK_NGPU) {
    __atomic_fetch_add(&allocTracker[dev].totalAlloc, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&allocTracker[dev].totalAllocSize, nelem*sizeof(T), __ATOMIC_RELAXED);
  }
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr) WARN("Failed to CUDA calloc async %ld bytes", nelem*sizeof(T));
  INFO(NCCL_ALLOC, "%s:%d Cuda Alloc Size %ld pointer %p", filefunc, line, nelem*sizeof(T), *ptr);
  return result;
}
#define ncclCudaCallocAsync(...) ncclCudaCallocAsyncDebug(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  taccStream_t stream;
  CUDACHECKGOTO(taccStreamCreateWithFlags(&stream, taccStreamNonBlocking), result, finish);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  CUDACHECKGOTO(taccStreamSynchronize(stream), result, finish);
  CUDACHECKGOTO(taccStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, taccStream_t stream) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(taccMemcpyAsync(dst, src, nelem*sizeof(T), taccMemcpyDefault, stream), result, finish);
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaFree(T* ptr) {
  ncclResult_t result = ncclSuccess;
  taccStreamCaptureMode mode = taccStreamCaptureModeRelaxed;
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree((void *)ptr), result, finish);
  } else {
    CUDACHECKGOTO(taccFree(ptr), result, finish);
  }
finish:
  CUDACHECK(taccThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char *filefunc, int line) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  INFO(NCCL_ALLOC, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__)

#endif
