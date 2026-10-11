#include "tacc.h"
#include "backend_hip.h"
#include "backend_cuda.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "util.h"

static taccPlatform curPlatform = taccPlatformInvalid;

taccError_t taccGetAvailablePlatforms(int num_entries, taccPlatform* platforms, int* num_platforms) {
  if (platforms) {
    int n = 0;
    if (n < num_entries && IsCudaAvailable()) platforms[n++] = taccPlatformCuda;
    if (n < num_entries && IsHipAvailable()) platforms[n++] = taccPlatformHip;
    if (num_platforms) *num_platforms = n;
    return taccSuccess;
  }
  if (!num_platforms) return taccErrorInvalidValue;
  if (IsCudaAvailable()) ++*num_platforms;
  if (IsHipAvailable()) ++*num_platforms;
  return taccSuccess;
}

taccError_t taccSetPlatform(taccPlatform platform) {
  taccError_t res = taccSuccess;

  if (curPlatform == platform) {
    return res;
  }

  switch (curPlatform) {
    case taccPlatformCuda: res = FinalizeCuda(); break;
    case taccPlatformHip: res = FinalizeHip(); break;
    default:;
  }
  if (res != taccSuccess) return res;

  curPlatform = taccPlatformInvalid;

  switch (platform) {
    case taccPlatformCuda: res = InitializeCuda(); break;
    case taccPlatformHip: res = InitializeHip(); break;
    default: return taccErrorInvalidValue;
  }
  if (res != taccSuccess) return res;

  curPlatform = platform;

  return taccSuccess;
}

taccError_t taccSetPlatformAuto() {
  int num_platforms = 0;
  taccError_t res = taccSuccess;
  res = taccGetAvailablePlatforms(0, NULL, &num_platforms);
  if (res != taccSuccess) return res;
  if (num_platforms == 0) {
    WARN("No available platforms");
    return taccErrorNotInitialized;
  }
  if (num_platforms == 1) {
    taccPlatform platform;
    res = taccGetAvailablePlatforms(1, &platform, NULL);
    if (res != taccSuccess) return res;
    res = taccSetPlatform(platform);
    if (res != taccSuccess) return res;
    return taccSuccess;
  }
  // num_platforms >= 2
  char* name = getenv("TACC_PLATFORM");
  if (!name) {
    WARN("Multiple platforms available and TACC_PLATFORM is not set");
    return taccErrorNotInitialized;
  }
  if (strcmp(name, "CUDA") == 0) {
    res = taccSetPlatform(taccPlatformCuda);
    if (res != taccSuccess) return res;
  } else if (strcmp(name, "HIP") == 0) {
    res = taccSetPlatform(taccPlatformHip);
    if (res != taccSuccess) return res;
  } else {
    WARN("Invalid TACC_PLATFORM(%s). Should be CUDA or HIP.", name);
    return taccErrorNotInitialized;
  }
  return taccSuccess;
}

taccError_t taccGetPlatform(taccPlatform* platform) {
  if (curPlatform == taccPlatformInvalid) return taccErrorNotInitialized;
  if (!platform) return taccErrorInvalidValue;
  *platform = curPlatform;
  return taccSuccess;
}

taccError_t taccThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode) {
  if (curPlatform == taccPlatformHip) {
    return _hipThreadExchangeStreamCaptureMode(mode);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaThreadExchangeStreamCaptureMode(mode);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMalloc(void** devPtr, size_t size) {
  if (curPlatform == taccPlatformHip) {
    return _hipMalloc(devPtr, size);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMalloc(devPtr, size);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMallocHost(void** ptr, size_t size) {
  if (curPlatform == taccPlatformHip) {
    return _hipMallocHost(ptr, size);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMallocHost(ptr, size);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMallocManaged(void** ptr, size_t size, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipMallocManaged(ptr, size, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMallocManaged(ptr, size, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccHostAlloc(void** ptr, size_t size, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostAlloc(ptr, size, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaHostAlloc(ptr, size, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGetDevice(int* deviceId) {
  if (curPlatform == taccPlatformHip) {
    return _hipGetDevice(deviceId);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGetDevice(deviceId);
  }
  return taccErrorNotInitialized;
}

taccError_t taccSetDevice(int deviceId) {
  if (curPlatform == taccPlatformHip) {
    return _hipSetDevice(deviceId);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaSetDevice(deviceId);
  }
  return taccErrorNotInitialized;
}

const char* taccGetErrorString(taccError_t taccError) {
  if (curPlatform == taccPlatformHip) {
    return _hipGetErrorString(taccError);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGetErrorString(taccError);
  }
  return "taccErrorNotInitialized";
}

taccError_t taccGetLastError(void) {
  if (curPlatform == taccPlatformHip) {
    return _hipGetLastError();
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGetLastError();
  }
  return taccErrorNotInitialized;
}

taccError_t taccExtMallocWithFlags(void** ptr, size_t size, unsigned int flags) {
  if (curPlatform == taccPlatformHip)
    return _hipExtMallocWithFlags(ptr, size, flags);
  return taccErrorNotInitialized;
}

taccError_t taccDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
  }
  return taccErrorNotInitialized;
}

taccError_t taccFreeHost(void* ptr) {
  if (curPlatform == taccPlatformHip) {
    return _hipFreeHost(ptr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaFreeHost(ptr);
  }
  return taccErrorNotInitialized;
}

/*
taccError_t taccDeviceGet(taccdevice_t* device, int ordinal) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceGet(device, ordinal);
  } else if (curPlatform == taccPlatformCuda) {
    return _cuDeviceGet(device, ordinal);
  }
}
*/

taccError_t taccDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceGetAttribute(flag, attrib, device);
  }
  else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceGetAttribute(flag, attrib, device);
  }
  return taccErrorNotInitialized;
}

/*
taccError_t taccMemGetAllocationGranularity(size_t* granularity, const void* prop, taccMemAllocationGranularity_flags option) {
  if (curPlatform == taccPlatformHip) {
    return _hipMemGetAllocationGranularity(granularity, prop, option);
  } else if (curPlatform == taccPlatformCuda) {
    return _cuMemGetAllocationGranularity(granularity, prop, option);
  }
}
*/

taccError_t taccStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamCreateWithFlags(pStream, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamCreateWithFlags(pStream, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind) {
  if (curPlatform == taccPlatformHip) {
    return _hipMemcpy(dst, src, count, kind);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMemcpy(dst, src, count, kind);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipMemcpyAsync(dst, src, count, kind, stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMemcpyAsync(dst, src, count, kind, stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccFree(void* ptr) {
  if (curPlatform == taccPlatformHip) {
    return _hipFree(ptr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaFree(ptr);
  }
  return taccErrorNotInitialized;
}

taccError_t taccMemset(void *dst, int value, size_t sizeBytes) {
  if (curPlatform == taccPlatformHip) {
    return _hipMemset(dst, value, sizeBytes);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMemset(dst, value, sizeBytes);
  }
  return taccErrorNotInitialized;
}


taccError_t taccMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipMemsetAsync(dst, value, sizeBytes, stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaMemsetAsync(dst, value, sizeBytes, stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamQuery(taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamQuery(stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamQuery(stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamSynchronize(taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamSynchronize(stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamSynchronize(stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamDestroy(taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamDestroy(stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamDestroy(stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccEventCreate(taccEvent_t* event) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventCreate(event);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventCreate(event);
  }
  return taccErrorNotInitialized;
}

taccError_t taccEventCreateWithFlags(taccEvent_t* event, unsigned flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventCreateWithFlags(event, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventCreateWithFlags(event, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccEventQuery(taccEvent_t event) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventQuery(event);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventQuery(event);
  }
  return taccErrorNotInitialized;
}

taccError_t taccEventRecord(taccEvent_t event, taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventRecord(event, stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventRecord(event, stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamWaitEvent(stream, event, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamWaitEvent(stream, event, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccEventDestroy(taccEvent_t event) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventDestroy(event);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventDestroy(event);
  }
  return taccErrorNotInitialized;
}

taccError_t taccLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData) {
  if (curPlatform == taccPlatformHip) {
    return _hipLaunchHostFunc(stream, fn, userData);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaLaunchHostFunc(stream, fn, userData);
  }
  return taccErrorNotInitialized;
}

taccError_t taccIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipIpcOpenMemHandle(devPtr, handle, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaIpcOpenMemHandle(devPtr, handle, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccIpcCloseMemHandle(void* devPtr) {
  if (curPlatform == taccPlatformHip) {
    return _hipIpcCloseMemHandle(devPtr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaIpcCloseMemHandle(devPtr);
  }
  return taccErrorNotInitialized;
}

taccError_t taccIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr) {
  if (curPlatform == taccPlatformHip) {
    return _hipIpcGetMemHandle(handle, devPtr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaIpcGetMemHandle(handle, devPtr);
  }
  return taccErrorNotInitialized;
}

taccError_t taccHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostRegister(hostPtr, sizeBytes, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaHostRegister(hostPtr, sizeBytes, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccHostUnregister(void* hostPtr) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostUnregister(hostPtr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaHostUnregister(hostPtr);
  }
  return taccErrorNotInitialized;
}

taccError_t taccHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostGetDevicePointer(devPtr, hstPtr, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaHostGetDevicePointer(devPtr, hstPtr, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceGetPCIBusId(pciBusId, len, device);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceGetPCIBusId(pciBusId, len, device);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGetDeviceProperties(taccDeviceProp_t* prop, int deviceId) {
  if (curPlatform == taccPlatformHip) {
    return _hipGetDeviceProperties(prop, deviceId);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGetDeviceProperties(prop, deviceId);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceGetByPCIBusId(device, pciBusId);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceGetByPCIBusId(device, pciBusId);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceEnablePeerAccess(peerDeviceId, flags);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceEnablePeerAccess(peerDeviceId, flags);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGetDeviceCount(int* count) {
  if (curPlatform == taccPlatformHip) {
    return _hipGetDeviceCount(count);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGetDeviceCount(count);
  }
  return taccErrorNotInitialized;
}

taccError_t taccFuncSetAttribute(const void* func, taccFuncAttribute attr, int value) {
  if (curPlatform == taccPlatformHip) {
    return _hipFuncSetAttribute(func, attr, value);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaFuncSetAttribute(func, attr, value);
  }
  return taccErrorNotInitialized;
}

taccError_t taccFuncGetAttributes(taccFuncAttributes* attr, const void* func) {
  if (curPlatform == taccPlatformHip) {
    return _hipFuncGetAttributes(attr, func);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaFuncGetAttributes(attr, func);
  }
  return taccErrorNotInitialized;
}

taccError_t taccExtLaunchKernel(const void* function_address, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipExtLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream, startEvent, stopEvent, flags);
  }
  return taccErrorNotSupported;
}

taccError_t taccLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceSetLimit(taccLimit limit, size_t value) {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceSetLimit(limit, value);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceSetLimit(limit, value);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                       unsigned long long* id_out, taccGraph_t* graph_out,
                                       const taccGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams) {
  if (curPlatform == taccPlatformHip) {
    return _hipGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  if (curPlatform == taccPlatformHip) {
    return _hipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream) {
  if (curPlatform == taccPlatformHip) {
    return _hipGraphLaunch(graphExec, stream);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGraphLaunch(graphExec, stream);
  }
  return taccErrorNotInitialized;
}

taccError_t taccGraphExecDestroy(taccGraphExec_t graphExec) {
  if (curPlatform == taccPlatformHip) {
    return _hipGraphExecDestroy(graphExec);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGraphExecDestroy(graphExec);
  }
  return taccErrorNotInitialized;
}
taccError_t taccGraphDestroy(taccGraph_t graph) {
  if (curPlatform == taccPlatformHip) {
    return _hipGraphDestroy(graph);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaGraphDestroy(graph);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceSynchronize() {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceSynchronize();
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceSynchronize();
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamBeginCapture(stream, mode);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamBeginCapture(stream, mode);
  }
  return taccErrorNotInitialized;
}

taccError_t taccStreamEndCapture(taccStream_t stream, taccGraph_t* graph) {
  if (curPlatform == taccPlatformHip) {
    return _hipStreamEndCapture(stream, graph);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaStreamEndCapture(stream, graph);
  }
  return taccErrorNotInitialized;
}

taccError_t taccDeviceReset() {
  if (curPlatform == taccPlatformHip) {
    return _hipDeviceReset();
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaDeviceReset();
  }
  return taccErrorNotInitialized;
}

taccError_t taccPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr) {
  if (curPlatform == taccPlatformHip) {
    return _hipPointerGetAttributes(attributes, ptr);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaPointerGetAttributes(attributes, ptr);
  }
  return taccErrorNotInitialized;
}

taccError_t taccExtStreamCreateWithCUMask(taccStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask) {
  if (curPlatform == taccPlatformHip) {
    return _hipExtStreamCreateWithCUMask(stream, cuMaskSize, cuMask);
  }
  return taccErrorNotSupported;
}

taccError_t taccStreamCreate(taccStream_t* stream) {
  if (curPlatform == taccPlatformCuda) {
    return _cudaStreamCreate(stream);
  } else if (curPlatform == taccPlatformHip) {
    return _hipStreamCreate(stream);
  }
  return taccErrorNotSupported;
}

taccError_t taccExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount) {
  if (curPlatform == taccPlatformHip) {
    return _hipExtGetLinkTypeAndHopCount(device1, device2, linktype, hopcount);
  }
  return taccErrorNotSupported;
}

taccError_t taccHostMalloc(void** ptr, size_t size, unsigned int flags) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostMalloc(ptr, size, flags);
  }
  return taccErrorNotSupported;
}

taccError_t taccHostFree(void* ptr) {
  if (curPlatform == taccPlatformHip) {
    return _hipHostFree(ptr);
  }
  return taccErrorNotSupported;
}

taccError_t taccEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop) {
  if (curPlatform == taccPlatformHip) {
    return _hipEventElapsedTime(ms, start, stop);
  } else if (curPlatform == taccPlatformCuda) {
    return _cudaEventElapsedTime(ms, start, stop);
  }
  return taccErrorNotSupported;
}
