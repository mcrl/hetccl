#pragma once

#include "tacc.h"

bool IsHipAvailable();
taccError_t InitializeHip();
taccError_t FinalizeHip();

taccError_t _hipThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode);
taccError_t _hipMalloc(void** devPtr, size_t size);
taccError_t _hipMallocHost(void** devPtr, size_t size);
taccError_t _hipMallocManaged(void** ptr, size_t size, unsigned int flags);
taccError_t _hipHostAlloc(void** ptr, size_t size, unsigned int flags);
taccError_t _hipGetDevice(int* deviceId);
taccError_t _hipSetDevice(int deviceId);

const char* _hipGetErrorString(taccError_t taccError);
taccError_t _hipGetLastError(void);

taccError_t _hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags);
taccError_t _hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
taccError_t _hipFreeHost(void* ptr);

// taccError_t _hipDeviceGet(taccdevice_t* device, int ordinal);
taccError_t _hipDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device);
// taccError_t _hipMemGetAllocationGranularity(size_t* granularity, const void* prop, taccMemAllocationGranularity_flags option);

taccError_t _hipStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags);
taccError_t _hipMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind);
taccError_t _hipMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream);
taccError_t _hipFree(void* ptr);
taccError_t _hipMemset(void* dst, int value, size_t sizeBytes);
taccError_t _hipMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream);
taccError_t _hipStreamQuery(taccStream_t stream);
taccError_t _hipStreamSynchronize(taccStream_t stream);
taccError_t _hipStreamDestroy(taccStream_t stream);
taccError_t _hipEventCreate(taccEvent_t* event);
taccError_t _hipEventCreateWithFlags(taccEvent_t* event, unsigned flags);
taccError_t _hipEventQuery(taccEvent_t event);
taccError_t _hipEventRecord(taccEvent_t event, taccStream_t stream);
taccError_t _hipStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags);
taccError_t _hipEventDestroy(taccEvent_t event);

taccError_t _hipLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData);
// taccError_t _hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, taccStream_t stream);

taccError_t _hipIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags);
taccError_t _hipIpcCloseMemHandle(void* devPtr);
taccError_t _hipIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr);

taccError_t _hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
taccError_t _hipHostUnregister(void* hostPtr);
taccError_t _hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
taccError_t _hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
taccError_t _hipGetDeviceProperties(taccDeviceProp_t* prop, int deviceId);
taccError_t _hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
taccError_t _hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
taccError_t _hipGetDeviceCount(int* count);

taccError_t _hipFuncSetAttribute(const void* func, taccFuncAttribute attr, int value);
taccError_t _hipFuncGetAttributes(taccFuncAttributes* attr, const void* func);

taccError_t _hipExtLaunchKernel(const void* function_address, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags);

taccError_t _hipLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args,
                             size_t sharedMem, taccStream_t stream);

taccError_t _hipDeviceSetLimit(taccLimit limit, size_t value);

taccError_t _hipStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                         unsigned long long* id_out, taccGraph_t* graph_out,
                                         const taccGraphNode_t** dependencies_out, size_t* numDependencies_out);

taccError_t _hipGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams);
taccError_t _hipGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
taccError_t _hipGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream);
taccError_t _hipGraphExecDestroy(taccGraphExec_t graphExec);
taccError_t _hipGraphDestroy(taccGraph_t graph);
taccError_t _hipDeviceSynchronize();
taccError_t _hipStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode);
taccError_t _hipStreamEndCapture(taccStream_t stream, taccGraph_t* graph);
taccError_t _hipDeviceReset();
taccError_t _hipPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr);
taccError_t _hipExtStreamCreateWithCUMask(taccStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask);
taccError_t _hipStreamCreate(taccStream_t* stream);
taccError_t _hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount);
taccError_t _hipHostMalloc(void** ptr, size_t size, unsigned int flags);
taccError_t _hipHostFree(void* ptr);
taccError_t _hipEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop);