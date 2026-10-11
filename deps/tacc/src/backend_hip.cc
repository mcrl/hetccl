#include "backend_hip.h"
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include "util.h"

static struct {
  void* handle = nullptr;
  #define CMD(name, ret_type, ...) \
    ret_type (*name)(__VA_ARGS__) = nullptr
  #include "backend_hip_func_list.h"
  #undef CMD
} funcList;

bool IsHipAvailable() {
  bool wasInitialized = funcList.handle != nullptr;
  if (!wasInitialized) {
    if (InitializeHip() != taccSuccess) return false;
  }
  int count = 0;
  if (funcList.hipGetDeviceCount(&count) != hipSuccess) {
    WARN("HIP is available but no devices are found");
  }
  if (!wasInitialized) {
    if (FinalizeHip() != taccSuccess) return false;
  }
  return count > 0;
}

taccError_t InitializeHip() {
  if (funcList.handle) {
    // Already initialized
    return taccSuccess;
  }
  
  const char* fn = "libamdhip64.so";
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
  #include "backend_hip_func_list.h"
  #undef CMD

  return taccSuccess;
}

taccError_t FinalizeHip() {
  if (!funcList.handle) {
    // Already Finalized
    return taccSuccess;
  }
  int res = dlclose(funcList.handle);
  if (res != 0) return taccErrorNotInitialized;
  funcList.handle = NULL;
  return taccSuccess;
}

taccError_t _hipThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode) {
  return (taccError_t)funcList.hipThreadExchangeStreamCaptureMode((hipStreamCaptureMode*)mode);
}

taccError_t _hipMalloc(void** devPtr, size_t size) {
  return (taccError_t)funcList.hipMalloc(devPtr, size);
}

taccError_t _hipMallocHost(void** devPtr, size_t size) {
  return (taccError_t)funcList.hipMallocHost(devPtr, size);
}

taccError_t _hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.hipMallocManaged(ptr, size, flags);
}

taccError_t _hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.hipHostMalloc(ptr, size, flags);
}

taccError_t _hipGetDevice(int* deviceId) {
  return (taccError_t)funcList.hipGetDevice(deviceId);
}

taccError_t _hipSetDevice(int deviceId) {
  return (taccError_t)funcList.hipSetDevice(deviceId);
}

const char* _hipGetErrorString(taccError_t taccError) {
  return funcList.hipGetErrorString((hipError_t)taccError);
}

taccError_t _hipGetLastError(void) {
  return (taccError_t)funcList.hipGetLastError();
}

taccError_t _hipExtMallocWithFlags(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.hipExtMallocWithFlags(ptr, size, flags);
}

taccError_t _hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
  return (taccError_t)funcList.hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
}

taccError_t _hipFreeHost(void* ptr) {
  return (taccError_t)funcList.hipHostFree(ptr);
}

/*
taccError_t _hipDeviceGet(taccdevice_t* device, int ordinal) {
  return (taccError_t)funcList.hipDeviceGet((hipdevice_t*) device, ordinal);
}
*/

taccError_t _hipDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device) {
  return (taccError_t)funcList.hipDeviceGetAttribute(flag, (hipDeviceAttribute_t) attrib, (int) device);
}

/*
taccError_t _hipMemGetAllocationGranularity(size_t* granularity, const void* prop, taccMemAllocationGranularity_flags option) {
  return (taccError_t)funcList.hipMemGetAllocationGranularity(granularity, (const hipMemAllocationProp*) prop, (hipMemAllocationGranularity_flags) option);
}
*/

taccError_t _hipStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags) {
  return (taccError_t)funcList.hipStreamCreateWithFlags((hipStream_t*) pStream, flags);
}

taccError_t _hipMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind) {
  return (taccError_t)funcList.hipMemcpy(dst, src, count, (hipMemcpyKind) kind);
}

taccError_t _hipMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream) {
  return (taccError_t)funcList.hipMemcpyAsync(dst, src, count, (hipMemcpyKind) kind, (hipStream_t) stream);
}

taccError_t _hipFree(void* ptr) {
  return (taccError_t)funcList.hipFree(ptr);
}

taccError_t _hipMemset(void* dst, int value, size_t sizeBytes) {
  return (taccError_t)funcList.hipMemset(dst, value, sizeBytes);
}

taccError_t _hipMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream) {
  return (taccError_t)funcList.hipMemsetAsync(dst, value, sizeBytes, (hipStream_t) stream);
}

taccError_t _hipStreamQuery(taccStream_t stream) {
  return (taccError_t)funcList.hipStreamQuery((hipStream_t) stream);
}

taccError_t _hipStreamSynchronize(taccStream_t stream) {
  return (taccError_t)funcList.hipStreamSynchronize((hipStream_t) stream);
}

taccError_t _hipStreamDestroy(taccStream_t stream) {
  return (taccError_t)funcList.hipStreamDestroy((hipStream_t) stream);
}

taccError_t _hipEventCreate(taccEvent_t* event) {
  return (taccError_t)funcList.hipEventCreate((hipEvent_t*) event);
}

taccError_t _hipEventCreateWithFlags(taccEvent_t* event, unsigned flags) {
  return (taccError_t)funcList.hipEventCreateWithFlags((hipEvent_t*) event, flags);
}

taccError_t _hipEventQuery(taccEvent_t event) {
  return (taccError_t)funcList.hipEventQuery((hipEvent_t) event);
}

taccError_t _hipEventRecord(taccEvent_t event, taccStream_t stream) {
  return (taccError_t)funcList.hipEventRecord((hipEvent_t) event, (hipStream_t) stream);
}

taccError_t _hipStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags) {
  return (taccError_t)funcList.hipStreamWaitEvent((hipStream_t) stream, (hipEvent_t) event, flags);
}

taccError_t _hipEventDestroy(taccEvent_t event) {
  return (taccError_t)funcList.hipEventDestroy((hipEvent_t) event);
}

taccError_t _hipLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData) {
  return (taccError_t)funcList.hipLaunchHostFunc((hipStream_t) stream, (hipHostFn_t) fn, userData);
}

/*
taccError_t _hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args, size_t sharedMemBytes, taccStream_t stream) {
  return (taccError_t)funcList.hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, (hipStream_t) stream);
}
*/

hipIpcMemHandle_t taccHandleToHip(taccIpcMemHandle_t src) {
  hipIpcMemHandle_t dest;
  memcpy(dest.reserved, src.reserved, TACC_IPC_HANDLE_SIZE);
  return dest;
}

taccError_t _hipIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags) {
  return (taccError_t)funcList.hipIpcOpenMemHandle(devPtr, taccHandleToHip(handle), flags);
}

taccError_t _hipIpcCloseMemHandle(void* devPtr) {
  return (taccError_t)funcList.hipIpcCloseMemHandle(devPtr);
}

void hipHandleToTacc(hipIpcMemHandle_t* src, taccIpcMemHandle_t* dest) {
  memcpy(dest->reserved, src->reserved, TACC_IPC_HANDLE_SIZE);
}

taccError_t _hipIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr) {
  hipIpcMemHandle_t hipHandle;
  taccError_t ret = (taccError_t)funcList.hipIpcGetMemHandle(&hipHandle, devPtr);
  hipHandleToTacc(&hipHandle, handle);
  return ret;
}

taccError_t _hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  return (taccError_t)funcList.hipHostRegister(hostPtr, sizeBytes, flags);
}

taccError_t _hipHostUnregister(void* hostPtr) {
  return (taccError_t)funcList.hipHostUnregister(hostPtr);
}

taccError_t _hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
  return (taccError_t)funcList.hipHostGetDevicePointer(devPtr, hstPtr, flags);
}

taccError_t _hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  return (taccError_t)funcList.hipDeviceGetPCIBusId(pciBusId, len, device);
}

void convertHipArch(hipDeviceArch_t* src, hipDeviceArch* dest) {
  dest->hasGlobalInt32Atomics = src->hasGlobalInt32Atomics;
  dest->hasGlobalFloatAtomicExch = src->hasGlobalFloatAtomicExch;
  dest->hasSharedInt32Atomics = src->hasSharedInt32Atomics;
  dest->hasSharedFloatAtomicExch = src->hasSharedFloatAtomicExch;
  dest->hasFloatAtomicAdd = src->hasFloatAtomicAdd;

  dest->hasGlobalInt64Atomics = src->hasGlobalInt64Atomics;
  dest->hasSharedInt64Atomics = src->hasSharedInt64Atomics;

  dest->hasDoubles = src->hasDoubles;

  dest->hasWarpVote = src->hasWarpVote;
  dest->hasWarpBallot = src->hasWarpBallot;
  dest->hasWarpShuffle = src->hasWarpShuffle;
  dest->hasFunnelShift = src->hasFunnelShift;

  dest->hasThreadFenceSystem = src->hasThreadFenceSystem;
  dest->hasSyncThreadsExt = src->hasSyncThreadsExt;

  dest->hasSurfaceFuncs = src->hasSurfaceFuncs;
  dest->has3dGrid = src->has3dGrid;
  dest->hasDynamicParallelism = src->hasDynamicParallelism;
}

void hipDevicePropToTacc(hipDeviceProp_t* src, taccDeviceProp_t* dest) {
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
    dest->sparseTaccArraySupported = src->sparseHipArraySupported;
    dest->hostRegisterReadOnlySupported = src->hostRegisterReadOnlySupported;
    dest->timelineSemaphoreInteropSupported = src->timelineSemaphoreInteropSupported;
    dest->memoryPoolsSupported = src->memoryPoolsSupported;
    dest->gpuDirectRDMASupported = src->gpuDirectRDMASupported;
    dest->gpuDirectRDMAFlushWritesOptions = src->gpuDirectRDMAFlushWritesOptions;
    dest->gpuDirectRDMAWritesOrdering = src->gpuDirectRDMAWritesOrdering;
    dest->memoryPoolSupportedHandleTypes = src->memoryPoolSupportedHandleTypes;
    dest->deferredMappingTaccArraySupported = src->deferredMappingHipArraySupported;
    dest->ipcEventSupported = src->ipcEventSupported;
    dest->clusterLaunch = src->clusterLaunch;
    dest->unifiedFunctionPointers = src->unifiedFunctionPointers;

    // HIP Only struct members
    memcpy(dest->gcnArchName, src->gcnArchName, sizeof(dest->gcnArchName));

    dest->maxSharedMemoryPerMultiProcessor = src->maxSharedMemoryPerMultiProcessor;
    dest->clockInstructionRate = src->clockInstructionRate;

    convertHipArch(&src->arch, &dest->arch);

    dest->hdpMemFlushCntl = src->hdpMemFlushCntl;
    dest->hdpRegFlushCntl = src->hdpRegFlushCntl;
    dest->cooperativeMultiDeviceUnmatchedFunc = src->cooperativeMultiDeviceUnmatchedFunc;
    dest->cooperativeMultiDeviceUnmatchedGridDim = src->cooperativeMultiDeviceUnmatchedGridDim;
    dest->cooperativeMultiDeviceUnmatchedBlockDim = src->cooperativeMultiDeviceUnmatchedBlockDim;
    dest->cooperativeMultiDeviceUnmatchedSharedMem = src->cooperativeMultiDeviceUnmatchedSharedMem;
    dest->isLargeBar = src->isLargeBar;
    dest->asicRevision = src->asicRevision;
}

taccError_t _hipGetDeviceProperties(taccDeviceProp_t* prop, int deviceId) {
  hipDeviceProp_t hipProp;
  taccError_t ret;

  // Note: This actually calls hipGetDevicePropertiesR0600, because hip_runtime_api.h has the following line:
  // #define hipGetDeviceProperties hipGetDevicePropertiesR0600
  // Also, You should write hipGetDevicePropertiesR0600 in backend_hip_func_list.h,
  // because when dlsym("hipGetDeviceProperties"), the string will not be substituted by preprocessor.
  ret = (taccError_t)funcList.hipGetDeviceProperties(&hipProp, deviceId);
  hipDevicePropToTacc(&hipProp, prop);
  return ret;
}

taccError_t _hipDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  return (taccError_t)funcList.hipDeviceGetByPCIBusId(device, pciBusId);
}

taccError_t _hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  return (taccError_t)funcList.hipDeviceEnablePeerAccess(peerDeviceId, flags);
}

taccError_t _hipGetDeviceCount(int* count) {
  return (taccError_t)funcList.hipGetDeviceCount(count);
}

taccError_t _hipFuncSetAttribute(const void* func, taccFuncAttribute attr, int value) {
  return (taccError_t)funcList.hipFuncSetAttribute(func, (hipFuncAttribute)attr, value);
}

void hipFuncAttrToTacc(hipFuncAttributes* src, taccFuncAttributes* dest){
  dest->binaryVersion = src->binaryVersion;
  dest->cacheModeCA = src->cacheModeCA;
  dest->constSizeBytes = src->constSizeBytes;
  dest->localSizeBytes = src->localSizeBytes;
  dest->maxDynamicSharedSizeBytes = src->maxDynamicSharedSizeBytes;
  dest->maxThreadsPerBlock = src->maxThreadsPerBlock;
  dest->numRegs = src->numRegs;
  dest->preferredShmemCarveout = src->preferredShmemCarveout;
  dest->ptxVersion = src->ptxVersion;
  dest->sharedSizeBytes = src->sharedSizeBytes;
}

taccError_t _hipFuncGetAttributes(taccFuncAttributes* attr, const void* func) {
  hipFuncAttributes hipAttr;
  taccError_t ret;

  ret = (taccError_t)funcList.hipFuncGetAttributes(&hipAttr, func);
  hipFuncAttrToTacc(&hipAttr, attr);
  return ret;
}

taccError_t _hipExtLaunchKernel(const void* function_address, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags) {
  dim3 numBlocks_;
  dim3 dimBlocks_;

  memcpy(&numBlocks_, &numBlocks, sizeof(numBlocks_));
  memcpy(&dimBlocks_, &dimBlocks, sizeof(dimBlocks_));

  return (taccError_t)funcList.hipExtLaunchKernel(function_address, numBlocks_, dimBlocks_, args, 
                                         sharedMemBytes, (hipStream_t) stream, 
                                         (hipEvent_t) startEvent, (hipEvent_t) stopEvent, flags);
}

taccError_t _hipLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args,
                             size_t sharedMem, taccStream_t stream) {
  dim3 gridDim_;
  dim3 blockDim_;

  memcpy(&gridDim_, &gridDim, sizeof(gridDim_));
  memcpy(&blockDim_, &blockDim, sizeof(blockDim_));

  return (taccError_t)funcList.hipLaunchKernel(func, gridDim_, blockDim_, args, sharedMem, (hipStream_t) stream);
}

taccError_t _hipDeviceSetLimit(taccLimit limit, size_t value) {
  return (taccError_t)funcList.hipDeviceSetLimit((hipLimit_t) limit, value);
}

taccError_t _hipStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                         unsigned long long* id_out, taccGraph_t* graph_out,
                                         const taccGraphNode_t** dependencies_out, size_t* numDependencies_out) {
  return (taccError_t)funcList.hipStreamGetCaptureInfo_v2((hipStream_t)stream, (hipStreamCaptureStatus*)captureStatus_out, id_out, 
                                                 (hipGraph_t*)graph_out, (const hipGraphNode_t**)dependencies_out, numDependencies_out);
}

taccError_t _hipGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams) {
  return (taccError_t)funcList.hipGraphAddHostNode((hipGraphNode_t*)pGraphNode, (hipGraph_t)graph, (const hipGraphNode_t*)pDependencies, numDependencies, (const hipHostNodeParams*)pNodeParams);
}

taccError_t _hipGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  return (taccError_t)funcList.hipGraphInstantiate((hipGraphExec_t*)pGraphExec, (hipGraph_t)graph, (hipGraphNode_t*)pErrorNode, pLogBuffer, bufferSize);
}

taccError_t _hipGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream) {
  return (taccError_t)funcList.hipGraphLaunch((hipGraphExec_t) graphExec, (hipStream_t) stream);
}

taccError_t _hipGraphExecDestroy(taccGraphExec_t graphExec) {
  return (taccError_t)funcList.hipGraphExecDestroy((hipGraphExec_t) graphExec);
}

taccError_t _hipGraphDestroy(taccGraph_t graph) {
  return (taccError_t)funcList.hipGraphDestroy((hipGraph_t) graph);
}

taccError_t _hipDeviceSynchronize() {
  return (taccError_t)funcList.hipDeviceSynchronize();
}

taccError_t _hipStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode) {
  return (taccError_t)funcList.hipStreamBeginCapture((hipStream_t)stream, (hipStreamCaptureMode)mode);
}

taccError_t _hipStreamEndCapture(taccStream_t stream, taccGraph_t* graph) {
  return (taccError_t)funcList.hipStreamEndCapture((hipStream_t)stream, (hipGraph_t*) graph);
}

taccError_t _hipDeviceReset() {
  return (taccError_t)funcList.hipDeviceReset();
}

taccError_t _hipPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr) {
  return (taccError_t)funcList.hipPointerGetAttributes((hipPointerAttribute_t*)attributes, ptr);
}

taccError_t _hipExtStreamCreateWithCUMask(taccStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask) {
  return (taccError_t)funcList.hipExtStreamCreateWithCUMask((hipStream_t*)stream, cuMaskSize, cuMask);
}

taccError_t _hipStreamCreate(taccStream_t* stream) {
  return (taccError_t)funcList.hipStreamCreate((hipStream_t*)stream);
}

taccError_t _hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount) {
  return (taccError_t)funcList.hipExtGetLinkTypeAndHopCount(device1, device2, linktype, hopcount);
}

taccError_t _hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  return (taccError_t)funcList.hipHostMalloc(ptr, size, flags);
}

taccError_t _hipHostFree(void* ptr) {
  return (taccError_t)funcList.hipHostFree(ptr);
}

taccError_t _hipEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop) {
  return (taccError_t)funcList.hipEventElapsedTime(ms, (hipEvent_t)start, (hipEvent_t)stop);
}