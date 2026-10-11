#include "backend_cuda.h"
#include "tacc.h"
#include <cuda_runtime_api.h>
#include <string.h>
#include <cuda.h>
#include <dlfcn.h>
#include "util.h"

static struct {
  void* handle = nullptr;
  #define CMD(name, ret_type, ...) \
    ret_type (*name)(__VA_ARGS__) = nullptr
  #include "backend_cuda_func_list.h"
  #undef CMD
} funcList;

bool IsCudaAvailable() {
  bool wasInitialized = funcList.handle != nullptr;
  if (!wasInitialized) {
    if (InitializeCuda() != taccSuccess) return false;
  }
  int count = 0;
  if (funcList.cudaGetDeviceCount(&count) != cudaSuccess) {
    //WARN("CUDA is available but no devices are found");
  }
  if (!wasInitialized) {
    if (FinalizeCuda() != taccSuccess) return false;
  }
  return count > 0;
}

taccError_t InitializeCuda() {
  if (funcList.handle) {
    // Already initialized
    return taccSuccess;
  }
  
  const char* fn = "libcudart.so";
  funcList.handle = dlopen(fn, RTLD_LAZY);
  if (!funcList.handle) {
    //WARN("Failed to open %s", fn);
    return taccErrorNotInitialized;
  }

  #define CMD(name, ret_type, ...) \
    do { \
      funcList.name = (ret_type(*)(__VA_ARGS__))dlsym(funcList.handle, #name); \
      if (!funcList.name) { \
        WARN("Failed to load %s", #name); \
      } \
    } while (0)
  #include "backend_cuda_func_list.h"
  #undef CMD

  return taccSuccess;
}

taccError_t FinalizeCuda() {
  if (!funcList.handle) {
    // Already Finalized
    return taccSuccess;
  }
  int res = dlclose(funcList.handle);
  if (res != 0) return taccErrorNotInitialized;
  funcList.handle = NULL;
  return taccSuccess;
}

taccError_t _cudaThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode) {
  return (taccError_t)funcList.cudaThreadExchangeStreamCaptureMode((cudaStreamCaptureMode*)mode);
}

taccError_t _cudaMalloc(void** devPtr, size_t size) {
  return (taccError_t)funcList.cudaMalloc(devPtr, size);
}

taccError_t _cudaMallocHost(void** devPtr, size_t size) {
  return (taccError_t)funcList.cudaMallocHost(devPtr, size);
}

taccError_t _cudaMallocManaged(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.cudaMallocManaged(ptr, size, flags);
}

taccError_t _cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.cudaHostAlloc(ptr, size, flags);
}

taccError_t _cudaGetDevice(int* deviceId) {
  return (taccError_t)funcList.cudaGetDevice(deviceId);
}

taccError_t _cudaSetDevice(int deviceId) {
  return (taccError_t)funcList.cudaSetDevice(deviceId);
}

const char* _cudaGetErrorString(taccError_t taccError) {
  return funcList.cudaGetErrorString((cudaError_t)taccError);
}

taccError_t _cudaGetLastError(void) {
  return (taccError_t)funcList.cudaGetLastError();
}

taccError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  return (taccError_t)funcList.cudaDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
}

taccError_t _cudaFreeHost(void* ptr) {
  return (taccError_t)funcList.cudaFreeHost(ptr);
}

static taccError_t ConvertTaccDeviceAttrToCudaDeviceAttr(taccDevice_attribute in, cudaDeviceAttr* out) {
  switch (in) {
    case taccDeviceAttributeWarpSize: *out = cudaDevAttrWarpSize; return taccSuccess;
    case taccDeviceAttributeComputeCapabilityMajor: *out = cudaDevAttrComputeCapabilityMajor; return taccSuccess;
    case taccDeviceAttributeComputeCapabilityMinor: *out = cudaDevAttrComputeCapabilityMinor; return taccSuccess;
    case taccDeviceAttributeMaxSharedMemoryPerMultiprocessor: *out = cudaDevAttrMaxSharedMemoryPerMultiprocessor; return taccSuccess;
    case taccDeviceAttributeMultiprocessorCount: *out = cudaDevAttrMultiProcessorCount; return taccSuccess;
    default:
      WARN("Unknown device attribute: %d", in);
      return taccErrorInvalidValue;
  }
}

taccError_t _cudaDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device) {
  cudaDeviceAttr cudaAttrib;
  taccError_t err = ConvertTaccDeviceAttrToCudaDeviceAttr(attrib, &cudaAttrib);
  if (err != taccSuccess) {
    return err;
  }
  return (taccError_t)funcList.cudaDeviceGetAttribute(flag, cudaAttrib, (int) device);
}

taccError_t _cudaStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags) {
  return (taccError_t)funcList.cudaStreamCreateWithFlags((cudaStream_t*) pStream, flags);
}

taccError_t _cudaMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind) {
  return (taccError_t)funcList.cudaMemcpy(dst, src, count, (cudaMemcpyKind) kind);
}

taccError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream) {
  return (taccError_t)funcList.cudaMemcpyAsync(dst, src, count, (cudaMemcpyKind) kind, (cudaStream_t) stream);
}

taccError_t _cudaFree(void* ptr) {
  return (taccError_t)funcList.cudaFree(ptr);
}

taccError_t _cudaMemset(void* dst, int value, size_t sizeBytes) {
  return (taccError_t)funcList.cudaMemset(dst, value, sizeBytes);
}

taccError_t _cudaMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream) {
  return (taccError_t)funcList.cudaMemsetAsync(dst, value, sizeBytes, (cudaStream_t) stream);
}

taccError_t _cudaStreamQuery(taccStream_t stream) {
  return (taccError_t)funcList.cudaStreamQuery((cudaStream_t) stream);
}

taccError_t _cudaStreamSynchronize(taccStream_t stream) {
  return (taccError_t)funcList.cudaStreamSynchronize((cudaStream_t) stream);
}

taccError_t _cudaStreamDestroy(taccStream_t stream) {
  return (taccError_t)funcList.cudaStreamDestroy((cudaStream_t) stream);
}

taccError_t _cudaEventCreate(taccEvent_t* event) {
  return (taccError_t)funcList.cudaEventCreate((cudaEvent_t*) event);
}

taccError_t _cudaEventCreateWithFlags(taccEvent_t* event, unsigned flags) {
  return (taccError_t)funcList.cudaEventCreateWithFlags((cudaEvent_t*) event, flags);
}

taccError_t _cudaEventQuery(taccEvent_t event) {
  return (taccError_t)funcList.cudaEventQuery((cudaEvent_t) event);
}

taccError_t _cudaEventRecord(taccEvent_t event, taccStream_t stream) {
  return (taccError_t)funcList.cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
}

taccError_t _cudaStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags) {
  return (taccError_t)funcList.cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) event, flags);
}

taccError_t _cudaEventDestroy(taccEvent_t event) {
  return (taccError_t)funcList.cudaEventDestroy((cudaEvent_t) event);
}

taccError_t _cudaLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData) {
  return (taccError_t)funcList.cudaLaunchHostFunc((cudaStream_t) stream, (cudaHostFn_t) fn, userData);
}

cudaIpcMemHandle_t taccHandleToCuda(taccIpcMemHandle_t src) {
  cudaIpcMemHandle_t dest;
  memcpy(dest.reserved, src.reserved, sizeof(src.reserved));
  return dest;
}

taccError_t _cudaIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags) {
  return (taccError_t)funcList.cudaIpcOpenMemHandle(devPtr, taccHandleToCuda(handle), flags);
}

taccError_t _cudaIpcCloseMemHandle(void* devPtr) {
  return (taccError_t)funcList.cudaIpcCloseMemHandle(devPtr);
}

void cudaHandleToTacc(cudaIpcMemHandle_t* src, taccIpcMemHandle_t* dest) {
  memcpy(dest->reserved, src->reserved, TACC_IPC_HANDLE_SIZE);
}

taccError_t _cudaIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr) {
  cudaIpcMemHandle_t cudaHandle;
  taccError_t ret = (taccError_t)funcList.cudaIpcGetMemHandle(&cudaHandle, devPtr);
  cudaHandleToTacc(&cudaHandle, handle);
  return ret;
}

taccError_t _cudaHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  return (taccError_t)funcList.cudaHostRegister(hostPtr, sizeBytes, flags);
}

taccError_t _cudaHostUnregister(void* hostPtr) {
  return (taccError_t)funcList.cudaHostUnregister(hostPtr);
}

taccError_t _cudaHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  return (taccError_t)funcList.cudaHostGetDevicePointer(devPtr, hstPtr, flags);
}

taccError_t _cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  return (taccError_t)funcList.cudaDeviceGetPCIBusId(pciBusId, len, device);
}

void cudaDevicePropToTacc(cudaDeviceProp* src, taccDeviceProp_t* dest) {
    memcpy(dest->name, src->name, sizeof(dest->name));
    memcpy(dest->uuid.bytes, src->uuid.bytes, sizeof(src->uuid.bytes));
    memcpy(dest->luid, src->luid, sizeof(dest->luid));

    dest->luidDeviceNodeMask = src->luidDeviceNodeMask;
    dest->totalGlobalMem = src->totalGlobalMem;
    dest->sharedMemPerBlock = src->sharedMemPerBlock;
    dest->regsPerBlock = src->regsPerBlock;
    dest->warpSize = src->warpSize;
    dest->memPitch = src->memPitch;
    dest->maxThreadsPerBlock = src->maxThreadsPerBlock;

    memcpy(dest->maxThreadsDim, src->maxThreadsDim, sizeof(dest->maxThreadsDim));
    memcpy(dest->maxGridSize, src->maxGridSize, sizeof(dest->maxGridSize));

    dest->clockRate = src->clockRate;
    dest->totalConstMem = src->totalConstMem;
    dest->major = src->major;
    dest->minor = src->minor;
    dest->textureAlignment = src->textureAlignment;
    dest->texturePitchAlignment = src->texturePitchAlignment;
    dest->deviceOverlap = src->deviceOverlap;
    dest->multiProcessorCount = src->multiProcessorCount;
    dest->kernelExecTimeoutEnabled = src->kernelExecTimeoutEnabled;
    dest->integrated = src->integrated;
    dest->canMapHostMemory = src->canMapHostMemory;
    dest->computeMode = src->computeMode;
    dest->maxTexture1D = src->maxTexture1D;
    dest->maxTexture1DMipmap = src->maxTexture1DMipmap;
    dest->maxTexture1DLinear = src->maxTexture1DLinear;

    memcpy(dest->maxTexture2D, src->maxTexture2D, sizeof(dest->maxTexture2D));
    memcpy(dest->maxTexture2DMipmap, src->maxTexture2DMipmap, sizeof(dest->maxTexture2DMipmap));
    memcpy(dest->maxTexture2DLinear, src->maxTexture2DLinear, sizeof(dest->maxTexture2DLinear));
    memcpy(dest->maxTexture2DGather, src->maxTexture2DGather, sizeof(dest->maxTexture2DGather));
    memcpy(dest->maxTexture3D, src->maxTexture3D, sizeof(dest->maxTexture3D));
    memcpy(dest->maxTexture3DAlt, src->maxTexture3DAlt, sizeof(dest->maxTexture3DAlt));
    
    dest->maxTextureCubemap = src->maxTextureCubemap;
    memcpy(dest->maxTexture1DLayered, src->maxTexture1DLayered, sizeof(dest->maxTexture1DLayered));
    memcpy(dest->maxTexture2DLayered, src->maxTexture2DLayered, sizeof(dest->maxTexture2DLayered));
    memcpy(dest->maxTextureCubemapLayered, src->maxTextureCubemapLayered, sizeof(dest->maxTextureCubemapLayered));
    
    dest->maxSurface1D = src->maxSurface1D;
    memcpy(dest->maxSurface2D, src->maxSurface2D, sizeof(dest->maxSurface2D));
    memcpy(dest->maxSurface3D, src->maxSurface3D, sizeof(dest->maxSurface3D));
    memcpy(dest->maxSurface1DLayered, src->maxSurface1DLayered, sizeof(dest->maxSurface1DLayered));
    memcpy(dest->maxSurface2DLayered, src->maxSurface2DLayered, sizeof(dest->maxSurface2DLayered));
    dest->maxSurfaceCubemap = src->maxSurfaceCubemap;
    memcpy(dest->maxSurfaceCubemapLayered, src->maxSurfaceCubemapLayered, sizeof(dest->maxSurfaceCubemapLayered));
    
    dest->surfaceAlignment = src->surfaceAlignment;
    dest->concurrentKernels = src->concurrentKernels;
    dest->ECCEnabled = src->ECCEnabled;
    dest->pciBusID = src->pciBusID;
    dest->pciDeviceID = src->pciDeviceID;
    dest->pciDomainID = src->pciDomainID;
    dest->tccDriver = src->tccDriver;
    dest->asyncEngineCount = src->asyncEngineCount;
    dest->unifiedAddressing = src->unifiedAddressing;
    dest->memoryClockRate = src->memoryClockRate;
    dest->memoryBusWidth = src->memoryBusWidth;
    dest->l2CacheSize = src->l2CacheSize;
    dest->persistingL2CacheMaxSize = src->persistingL2CacheMaxSize;
    dest->maxThreadsPerMultiProcessor = src->maxThreadsPerMultiProcessor;
    dest->streamPrioritiesSupported = src->streamPrioritiesSupported;
    dest->globalL1CacheSupported = src->globalL1CacheSupported;
    dest->localL1CacheSupported = src->localL1CacheSupported;
    dest->sharedMemPerMultiprocessor = src->sharedMemPerMultiprocessor;
    dest->regsPerMultiprocessor = src->regsPerMultiprocessor;
    dest->managedMemory = src->managedMemory;
    dest->isMultiGpuBoard = src->isMultiGpuBoard;
    dest->multiGpuBoardGroupID = src->multiGpuBoardGroupID;
    dest->hostNativeAtomicSupported = src->hostNativeAtomicSupported;
    dest->singleToDoublePrecisionPerfRatio = src->singleToDoublePrecisionPerfRatio;
    dest->pageableMemoryAccess = src->pageableMemoryAccess;
    dest->concurrentManagedAccess = src->concurrentManagedAccess;
    dest->computePreemptionSupported = src->computePreemptionSupported;
    dest->canUseHostPointerForRegisteredMem = src->canUseHostPointerForRegisteredMem;
    dest->cooperativeLaunch = src->cooperativeLaunch;
    dest->cooperativeMultiDeviceLaunch = src->cooperativeMultiDeviceLaunch;
    dest->sharedMemPerBlockOptin = src->sharedMemPerBlockOptin;
    dest->pageableMemoryAccessUsesHostPageTables = src->pageableMemoryAccessUsesHostPageTables;
    dest->directManagedMemAccessFromHost = src->directManagedMemAccessFromHost;
    dest->maxBlocksPerMultiProcessor = src->maxBlocksPerMultiProcessor;
    dest->accessPolicyMaxWindowSize = src->accessPolicyMaxWindowSize;
    dest->reservedSharedMemPerBlock = src->reservedSharedMemPerBlock;
    dest->hostRegisterSupported = src->hostRegisterSupported;
    dest->sparseTaccArraySupported = src->sparseCudaArraySupported;
    dest->hostRegisterReadOnlySupported = src->hostRegisterReadOnlySupported;
    dest->timelineSemaphoreInteropSupported = src->timelineSemaphoreInteropSupported;
    dest->memoryPoolsSupported = src->memoryPoolsSupported;
    dest->gpuDirectRDMASupported = src->gpuDirectRDMASupported;
    dest->gpuDirectRDMAFlushWritesOptions = src->gpuDirectRDMAFlushWritesOptions;
    dest->gpuDirectRDMAWritesOrdering = src->gpuDirectRDMAWritesOrdering;
    dest->memoryPoolSupportedHandleTypes = src->memoryPoolSupportedHandleTypes;
    dest->deferredMappingTaccArraySupported = src->deferredMappingCudaArraySupported;
    dest->ipcEventSupported = src->ipcEventSupported;
    dest->clusterLaunch = src->clusterLaunch;
    dest->unifiedFunctionPointers = src->unifiedFunctionPointers;
}

taccError_t _cudaGetDeviceProperties(taccDeviceProp_t* prop, int deviceId) {
  cudaDeviceProp cudaProp;
  taccError_t ret;

  ret = (taccError_t)funcList.cudaGetDeviceProperties(&cudaProp, deviceId);
  cudaDevicePropToTacc(&cudaProp, prop);
  return ret;
}

taccError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  return (taccError_t)funcList.cudaDeviceGetByPCIBusId(device, pciBusId);
}

taccError_t _cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  return (taccError_t)funcList.cudaDeviceEnablePeerAccess(peerDeviceId, flags);
}

taccError_t _cudaGetDeviceCount(int* count) {
  return (taccError_t)funcList.cudaGetDeviceCount(count);
}

taccError_t _cudaFuncSetAttribute(const void* func, taccFuncAttribute attr, int value) {
  return (taccError_t)funcList.cudaFuncSetAttribute(func, (cudaFuncAttribute)attr, value);
}

void cudaFuncAttrToTacc(cudaFuncAttributes* src, taccFuncAttributes* dest){
  dest->binaryVersion = src->binaryVersion;
  dest->cacheModeCA = src->cacheModeCA;
  dest->clusterDimMustBeSet = src->clusterDimMustBeSet;
  dest->clusterSchedulingPolicyPreference = src->clusterSchedulingPolicyPreference;
  dest->constSizeBytes = src->constSizeBytes;
  dest->localSizeBytes = src->localSizeBytes;
  dest->maxDynamicSharedSizeBytes = src->maxDynamicSharedSizeBytes;
  dest->maxThreadsPerBlock = src->maxThreadsPerBlock;
  dest->nonPortableClusterSizeAllowed = src->nonPortableClusterSizeAllowed;
  dest->numRegs = src->numRegs;
  dest->preferredShmemCarveout = src->preferredShmemCarveout;
  dest->ptxVersion = src->ptxVersion;
  dest->requiredClusterWidth = src->requiredClusterWidth;
  dest->sharedSizeBytes = src->sharedSizeBytes;
}

taccError_t _cudaFuncGetAttributes(taccFuncAttributes* attr, const void* func) {
  cudaFuncAttributes cudaAttr;
  taccError_t ret;

  ret = (taccError_t)funcList.cudaFuncGetAttributes(&cudaAttr, func);
  cudaFuncAttrToTacc(&cudaAttr, attr);
  return ret;
}

taccError_t _cudaLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args,
                             size_t sharedMem, taccStream_t stream) {
  dim3 gridDim_;
  dim3 blockDim_;

  memcpy(&gridDim_, &gridDim, sizeof(gridDim_));
  memcpy(&blockDim_, &blockDim, sizeof(blockDim_));

  return (taccError_t)funcList.cudaLaunchKernel(func, gridDim_, blockDim_, args, sharedMem, (cudaStream_t) stream);
}

taccError_t _cudaDeviceSetLimit(taccLimit limit, size_t value) {
  return (taccError_t)funcList.cudaDeviceSetLimit((cudaLimit)limit, value);
}

taccError_t _cudaStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                         unsigned long long* id_out, taccGraph_t* graph_out,
                                         const taccGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  return (taccError_t)funcList.cudaStreamGetCaptureInfo_v2((cudaStream_t)stream, (cudaStreamCaptureStatus*)captureStatus_out, id_out, 
                                                  (cudaGraph_t*)graph_out, (const cudaGraphNode_t**)dependencies_out, numDependencies_out);
}

taccError_t _cudaGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams) {
  return (taccError_t)funcList.cudaGraphAddHostNode((cudaGraphNode_t*)pGraphNode, (cudaGraph_t)graph, (const cudaGraphNode_t*)pDependencies, numDependencies, (const cudaHostNodeParams*)pNodeParams);
}

taccError_t _cudaGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  return (taccError_t)funcList.cudaGraphInstantiate((cudaGraphExec_t*)pGraphExec, (cudaGraph_t)graph);
}

taccError_t _cudaGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream) {
  return (taccError_t)funcList.cudaGraphLaunch((cudaGraphExec_t) graphExec, (cudaStream_t) stream);
}

taccError_t _cudaGraphExecDestroy(taccGraphExec_t graphExec) {
  return (taccError_t)funcList.cudaGraphExecDestroy((cudaGraphExec_t) graphExec);
}

taccError_t _cudaGraphDestroy(taccGraph_t graph) {
  return (taccError_t)funcList.cudaGraphDestroy((cudaGraph_t) graph);
}

taccError_t _cudaDeviceSynchronize() {
  return (taccError_t)funcList.cudaDeviceSynchronize();
}

taccError_t _cudaStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode) {
  return (taccError_t)funcList.cudaStreamBeginCapture((cudaStream_t)stream, (cudaStreamCaptureMode)mode);
}

taccError_t _cudaStreamEndCapture(taccStream_t stream, taccGraph_t* graph) {
  return (taccError_t)funcList.cudaStreamEndCapture((cudaStream_t)stream, (cudaGraph_t*) graph);
}

taccError_t _cudaDeviceReset() {
  return (taccError_t)funcList.cudaDeviceReset();
}

static void cudaMemoryTypeToTacc(cudaMemoryType* src, taccMemoryType* dest) {
  switch (*src) {
    case cudaMemoryTypeUnregistered: *dest = taccMemoryTypeUnregistered; break;
    case cudaMemoryTypeHost: *dest = taccMemoryTypeHost; break;
    case cudaMemoryTypeDevice: *dest = taccMemoryTypeDevice; break;
    case cudaMemoryTypeManaged: *dest = taccMemoryTypeManaged; break;
  }
}

static void cudaPointerAttributesToTacc(cudaPointerAttributes* src, taccPointerAttribute_t* dest) {
  dest->device = src->device;
  dest->devicePointer = src->devicePointer;
  dest->hostPointer = src->hostPointer;
  cudaMemoryTypeToTacc(&src->type, &dest->type);
}

taccError_t _cudaPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr) {
  cudaPointerAttributes attr;
  taccError_t ret = (taccError_t)funcList.cudaPointerGetAttributes(&attr, ptr);
  cudaPointerAttributesToTacc(&attr, attributes);
  return ret;
}

taccError_t _cudaStreamCreate(taccStream_t* stream) {
  return (taccError_t)funcList.cudaStreamCreate((cudaStream_t*)stream);
}

taccError_t _cudaEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop) {
  return (taccError_t)funcList.cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)stop);
}
