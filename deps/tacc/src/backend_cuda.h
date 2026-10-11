#pragma once

#include "tacc.h"

bool IsCudaAvailable();
taccError_t InitializeCuda();
taccError_t FinalizeCuda();

taccError_t _cudaThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode);
taccError_t _cudaMalloc(void** devPtr, size_t size);
taccError_t _cudaMallocHost(void** devPtr, size_t size);
taccError_t _cudaMallocManaged(void** ptr, size_t size, unsigned int flags);
taccError_t _cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
taccError_t _cudaGetDevice(int* deviceId);
taccError_t _cudaSetDevice(int deviceId);

const char* _cudaGetErrorString(taccError_t taccError);
taccError_t _cudaGetLastError(void);

taccError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
taccError_t _cudaFreeHost(void* ptr);

// taccError_t _cuDeviceGet(taccdevice_t* device, int ordinal);
taccError_t _cudaDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device);
// taccError_t _cuMemGetAllocationGranularity(size_t* granularity, const void* prop, taccMemAllocationGranularity_flags option);

taccError_t _cudaStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags);
taccError_t _cudaMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind);
taccError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream);
taccError_t _cudaFree(void* ptr);
taccError_t _cudaMemset(void* dst, int value, size_t sizeBytes);
taccError_t _cudaMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream);
taccError_t _cudaStreamQuery(taccStream_t stream);
taccError_t _cudaStreamSynchronize(taccStream_t stream);
taccError_t _cudaStreamDestroy(taccStream_t stream);
taccError_t _cudaEventCreate(taccEvent_t* event);
taccError_t _cudaEventCreateWithFlags(taccEvent_t* event, unsigned flags);
taccError_t _cudaEventQuery(taccEvent_t event);
taccError_t _cudaEventRecord(taccEvent_t event, taccStream_t stream);
taccError_t _cudaStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags);
taccError_t _cudaEventDestroy(taccEvent_t event);

taccError_t _cudaLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData);
// taccError_t _cudaLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, taccStream_t stream);

taccError_t _cudaIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags);
taccError_t _cudaIpcCloseMemHandle(void* devPtr);
taccError_t _cudaIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr);

taccError_t _cudaHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
taccError_t _cudaHostUnregister(void* hostPtr);
taccError_t _cudaHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
taccError_t _cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
taccError_t _cudaGetDeviceProperties(taccDeviceProp_t* prop, int deviceId);
taccError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId);
taccError_t _cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
taccError_t _cudaGetDeviceCount(int* count);

taccError_t _cudaFuncSetAttribute(const void* func, taccFuncAttribute attr, int value);
taccError_t _cudaFuncGetAttributes(taccFuncAttributes* attr, const void* func);

taccError_t _cudaLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args,
                             size_t sharedMem, taccStream_t stream);

taccError_t _cudaDeviceSetLimit(taccLimit limit, size_t value);

taccError_t _cudaStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                         unsigned long long* id_out, taccGraph_t* graph_out,
                                         const taccGraphNode_t** dependencies_out, size_t* numDependencies_out);

taccError_t _cudaGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams);
taccError_t _cudaGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
taccError_t _cudaGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream);
taccError_t _cudaGraphExecDestroy(taccGraphExec_t graphExec);
taccError_t _cudaGraphDestroy(taccGraph_t graph);
taccError_t _cudaDeviceSynchronize();
taccError_t _cudaStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode);
taccError_t _cudaStreamEndCapture(taccStream_t stream, taccGraph_t* graph);
taccError_t _cudaDeviceReset();
taccError_t _cudaPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr);
taccError_t _cudaStreamCreate(taccStream_t* stream);
taccError_t _cudaEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop);