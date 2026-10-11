#pragma once

#include <cstddef>
#include <cstdint>

enum taccPlatform {
  taccPlatformInvalid = -1,
  taccPlatformCuda = 0,
  taccPlatformHip = 1
};

#define taccStreamDefault 0x00
#define taccStreamNonBlocking 0x01

#define taccEventDefault 0x0
#define taccEventBlockingSync 0x1
#define taccEventDisableTiming 0x2
#define taccEventInterprocess 0x4

#define taccHostMallocPortable 0x1
#define taccHostMallocMapped 0x2
#define taccHostRegisterMapped 0x2

#define taccMemAttachGlobal   0x01

#define taccEventDisableSystemFence 0x20000000

#define taccDeviceMallocDefault 0x0
#define taccDeviceMallocFinegrained 0x1
#define taccDeviceMallocUncached 0x3

#define taccIpcMemLazyEnablePeerAccess 0x01

#ifdef _WIN32
#define TACCRT_CB __stdcall
#else
#define TACCRT_CB
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef void(* taccHostFn_t) (void *userData);
typedef struct taccHostNodeParams {
   taccHostFn_t fn;
   void* userData;
} taccHostNodeParams;

typedef enum taccError_t {
    taccSuccess = 0,  ///< Successful completion.
    taccErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    taccErrorOutOfMemory = 2,   ///< out of memory range.
    // Deprecated
    taccErrorMemoryAllocation = 2,  ///< Memory allocation error.
    taccErrorNotInitialized = 3,    ///< Invalid not initialized
    // Deprecated
    taccErrorInitializationError = 3,
    taccErrorDeinitialized = 4,      ///< Deinitialized
    taccErrorProfilerDisabled = 5,
    taccErrorProfilerNotInitialized = 6,
    taccErrorProfilerAlreadyStarted = 7,
    taccErrorProfilerAlreadyStopped = 8,
    taccErrorInvalidConfiguration = 9,  ///< Invalide configuration
    taccErrorInvalidPitchValue = 12,   ///< Invalid pitch value
    taccErrorInvalidSymbol = 13,   ///< Invalid symbol
    taccErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    taccErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    taccErrorInsufficientDriver = 35,
    taccErrorMissingConfiguration = 52,
    taccErrorPriorLaunchFailure = 53,
    taccErrorInvalidDeviceFunction = 98,  ///< Invalid device function
    taccErrorNoDevice = 100,  ///< Call to taccGetDeviceCount returned 0 devices
    taccErrorInvalidDevice = 101,  ///< DeviceID must be in range from 0 to compute-devices.
    taccErrorInvalidImage = 200,   ///< Invalid image
    taccErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    taccErrorContextAlreadyCurrent = 202,
    taccErrorMapFailed = 205,
    // Deprecated
    taccErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    taccErrorUnmapFailed = 206,
    taccErrorArrayIsMapped = 207,
    taccErrorAlreadyMapped = 208,
    taccErrorNoBinaryForGpu = 209,
    taccErrorAlreadyAcquired = 210,
    taccErrorNotMapped = 211,
    taccErrorNotMappedAsArray = 212,
    taccErrorNotMappedAsPointer = 213,
    taccErrorECCNotCorrectable = 214,
    taccErrorUnsupportedLimit = 215,   ///< Unsupported limit
    taccErrorContextAlreadyInUse = 216,   ///< The context is already in use
    taccErrorPeerAccessUnsupported = 217,
    taccErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    taccErrorInvalidGraphicsContext = 219,
    taccErrorInvalidSource = 300,   ///< Invalid source.
    taccErrorFileNotFound = 301,   ///< the file is not found.
    taccErrorSharedObjectSymbolNotFound = 302,
    taccErrorSharedObjectInitFailed = 303,   ///< Failed to initialize shared object.
    taccErrorOperatingSystem = 304,   ///< Not the correct operating system
    taccErrorInvalidHandle = 400,  ///< Invalide handle
    // Deprecated
    taccErrorInvalidResourceHandle = 400,  ///< Resource handle (taccEvent_t or taccStream_t) invalid.
    taccErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
    taccErrorNotFound = 500,   ///< Not found
    taccErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from taccSuccess (which indicates completion).  APIs that return
                             ///< this error include taccEventQuery and taccStreamQuery.
    taccErrorIllegalAddress = 700,
    taccErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    taccErrorLaunchTimeOut = 702,   ///< Timeout for the launch.
    taccErrorPeerAccessAlreadyEnabled =
        704,  ///< Peer access was already enabled from the current device.
    taccErrorPeerAccessNotEnabled =
        705,  ///< Peer access was never enabled from the current device.
    taccErrorSetOnActiveProcess = 708,   ///< The process is active.
    taccErrorContextIsDestroyed = 709,   ///< The context is already destroyed
    taccErrorAssert = 710,  ///< Produced when the kernel calls assert.
    taccErrorHostMemoryAlreadyRegistered =
        712,  ///< Produced when trying to lock a page-locked memory.
    taccErrorHostMemoryNotRegistered =
        713,  ///< Produced when trying to unlock a non-page-locked memory.
    taccErrorLaunchFailure =
        719,  ///< An exception occurred on the device while executing a kernel.
    taccErrorCooperativeLaunchTooLarge =
        720,  ///< This error indicates that the number of blocks launched per grid for a kernel
              ///< that was launched via cooperative launch APIs exceeds the maximum number of
              ///< allowed blocks for the current device
    taccErrorNotSupported = 801,  ///< Produced when the tacc API is not supported/implemented
    taccErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    taccErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    taccErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    taccErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    taccErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    taccErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies  are allowed
                                           ///< to cross the boundary
    taccErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from taccStreamLegacy.
    taccErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    taccErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the taccStreamCaptureModeRelaxed argument to
                                             ///< taccStreamBeginCapture was passed to
                                             ///< taccStreamEndCapture in a different thread.
    taccErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraintsspecific to instantiated graph
                                           ///< update.
    taccErrorUnknown = 999,  ///< Unknown error.
    // HSA Runtime Error Codes start here.
    taccErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    taccErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    taccErrorTbd  ///< Marker that more error codes are needed.
} taccError_t;

typedef enum taccStreamCaptureMode {
  taccStreamCaptureModeGlobal = 0,
  taccStreamCaptureModeThreadLocal,
  taccStreamCaptureModeRelaxed
} taccStreamCaptureMode;

taccError_t taccThreadExchangeStreamCaptureMode(taccStreamCaptureMode* mode);
taccError_t taccMalloc(void** devPtr, size_t size);
taccError_t taccMallocHost(void** ptr, size_t size);
taccError_t taccMallocManaged(void** ptr, size_t size, unsigned int flags=taccMemAttachGlobal);
taccError_t taccHostAlloc(void** ptr, size_t size, unsigned int flags);
taccError_t taccGetDevice(int* deviceId);
taccError_t taccSetDevice(int deviceId);

const char* taccGetErrorString(taccError_t taccError);
taccError_t taccGetLastError(void);

taccError_t taccExtMallocWithFlags(void** ptr, size_t size, unsigned int flags); // Only exists on HIP
taccError_t taccDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
taccError_t taccFreeHost(void* ptr);


// CUdevice is integer
typedef int taccdevice_t;
//taccError_t taccDeviceGet(taccdevice_t* device, int ordinal);

// From https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3
// different with https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___global_defs.html#gacc0acd7b9bda126c6bb3dfd6e2796d7c
/*
enum taccDevice_attribute {
    TACC_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    TACC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
    TACC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
    TACC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
    TACC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
    TACC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    TACC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
    TACC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    TACC_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
    TACC_DEVICE_ATTRIBUTE_WARP_SIZE,
    TACC_DEVICE_ATTRIBUTE_MAX_PITCH,
    TACC_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    TACC_DEVICE_ATTRIBUTE_CLOCK_RATE,
    TACC_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
    TACC_DEVICE_ATTRIBUTE_GPU_OVERLAP,
    TACC_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    TACC_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
    TACC_DEVICE_ATTRIBUTE_INTEGRATED,
    TACC_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
    TACC_DEVICE_ATTRIBUTE_COMPUTE_MODE,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
    TACC_DEVICE_ATTRIBUTE_CONTACCRRENT_KERNELS,
    TACC_DEVICE_ATTRIBUTE_ECC_ENABLED,
    TACC_DEVICE_ATTRIBUTE_PCI_BUS_ID,
    TACC_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
    TACC_DEVICE_ATTRIBUTE_TCC_DRIVER,
    TACC_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    TACC_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    TACC_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    TACC_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    TACC_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
    TACC_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
    TACC_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
    TACC_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURETACCBEMAP_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURETACCBEMAP_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURETACCBEMAP_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACETACCBEMAP_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACETACCBEMAP_LAYERED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_SURFACETACCBEMAP_LAYERED_LAYERS,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT,
    TACC_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    TACC_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    TACC_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH,
    TACC_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    TACC_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
    TACC_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
    TACC_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
    TACC_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
    TACC_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
    TACC_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    TACC_DEVICE_ATTRIBUTE_CONTACCRRENT_MANAGED_ACCESS,
    TACC_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1,
    TACC_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
    TACC_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH,
    TACC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    TACC_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
    TACC_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
    TACC_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
    TACC_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
    TACC_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
    TACC_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
    TACC_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_TACCDA_VMM_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
    TACC_DEVICE_ATTRIBUTE_SPARSE_TACCDA_ARRAY_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
    TACC_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
    TACC_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES,
    TACC_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH,
    TACC_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_TACCDA_ARRAY_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
    TACC_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR,
    TACC_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT,
    TACC_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS,
    TACC_DEVICE_ATTRIBUTE_NUMA_CONFIG,
    TACC_DEVICE_ATTRIBUTE_NUMA_ID,
    TACC_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
    TACC_DEVICE_ATTRIBUTE_MPS_ENABLED,
    TACC_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
    TACC_DEVICE_ATTRIBUTE_MAX
};
*/

// 1-to-1 mapping to hipDeviceAttribute_t, but not cudaDeviceAttr
typedef enum taccDevice_attribute {
    taccDeviceAttributeCudaCompatibleBegin = 0,
    taccDeviceAttributeEccEnabled = taccDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    taccDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    taccDeviceAttributeAsyncEngineCount,                 ///< Asynchronous engines number.
    taccDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    taccDeviceAttributeCanUseHostPointerForRegisteredMem,///< Device can access host registered memory
                                                        ///< at the same virtual address as the CPU
    taccDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    taccDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    taccDeviceAttributeComputePreemptionSupported,       ///< Device supports Compute Preemption.
    taccDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    taccDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    taccDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    taccDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    taccDeviceAttributeDeviceOverlap,                    ///< Device can concurrently copy memory and execute a kernel.
                                                        ///< Deprecated. Use instead asyncEngineCount.
    taccDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
                                                        ///< the device without migration
    taccDeviceAttributeGlobalL1CacheSupported,           ///< Device supports caching globals in L1
    taccDeviceAttributeHostNativeAtomicSupported,        ///< Link between the device and the host supports native atomic operations
    taccDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    taccDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    taccDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    taccDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    taccDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    taccDeviceAttributeLuid,                             ///< 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    taccDeviceAttributeLuidDeviceNodeMask,               ///< Luid device node mask. Undefined on TCC and non-Windows platforms
    taccDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    taccDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    taccDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Max block size per multiprocessor
    taccDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    taccDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    taccDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    taccDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    taccDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    taccDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    taccDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    taccDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    taccDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    taccDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    taccDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    taccDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    taccDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    taccDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    taccDeviceAttributeMaxTexture1DLayered,              ///< Maximum dimensions of 1D layered texture.
    taccDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
                                                        ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    taccDeviceAttributeMaxTexture1DMipmap,               ///< Maximum size of 1D mipmapped texture.
    taccDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    taccDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    taccDeviceAttributeMaxTexture2DGather,               ///< Maximum dimensions of 2D texture if gather operations  performed.
    taccDeviceAttributeMaxTexture2DLayered,              ///< Maximum dimensions of 2D layered texture.
    taccDeviceAttributeMaxTexture2DLinear,               ///< Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    taccDeviceAttributeMaxTexture2DMipmap,               ///< Maximum dimensions of 2D mipmapped texture.
    taccDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    taccDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    taccDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    taccDeviceAttributeMaxTexture3DAlt,                  ///< Maximum dimensions of alternate 3D texture.
    taccDeviceAttributeMaxTextureCubemap,                ///< Maximum dimensions of Cubemap texture
    taccDeviceAttributeMaxTextureCubemapLayered,         ///< Maximum dimensions of Cubemap layered texture.
    taccDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    taccDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    taccDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    taccDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    taccDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    taccDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    taccDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    taccDeviceAttributeMultiGpuBoardGroupID,             ///< Unique ID of device group on the same multi-GPU board
    taccDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    taccDeviceAttributeUnused1,                          ///< Previously taccDeviceAttributeName
    taccDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
                                                        ///< without calling hipHostRegister on it
    taccDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    taccDeviceAttributePciBusId,                         ///< PCI Bus ID.
    taccDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    taccDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    taccDeviceAttributePersistingL2CacheMaxSize,         ///< Maximum l2 persisting lines capacity in bytes
    taccDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
                                                        ///< by all thread blocks simultaneously resident on a multiprocessor.
    taccDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    taccDeviceAttributeReservedSharedMemPerBlock,        ///< Shared memory reserved by CUDA driver per block.
    taccDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    taccDeviceAttributeSharedMemPerBlockOptin,           ///< Maximum shared memory per block usable by special opt in.
    taccDeviceAttributeSharedMemPerMultiprocessor,       ///< Shared memory available per multiprocessor.
    taccDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    taccDeviceAttributeStreamPrioritiesSupported,        ///< Whether to support stream priorities.
    taccDeviceAttributeSurfaceAlignment,                 ///< Alignment requirement for surfaces
    taccDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    taccDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    taccDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    taccDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    taccDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    taccDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    taccDeviceAttributeUnused2,                          ///< Previously taccDeviceAttributeUuid
    taccDeviceAttributeWarpSize,                         ///< Warp size in threads.
    taccDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    taccDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management
    taccDeviceAttributeHostRegisterSupported,            ///< Can device support host memory registration via hipHostRegister
    taccDeviceAttributeCudaCompatibleEnd = 9999,
    taccDeviceAttributeAmdSpecificBegin = 10000,

    taccDeviceAttributeClockInstructionRate = taccDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    taccDeviceAttributeUnused3,                                  ///< Previously taccDeviceAttributeArch
    taccDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    taccDeviceAttributeUnused4,                                  ///< Previously taccDeviceAttributeGcnArch
    taccDeviceAttributeUnused5,                                  ///< Previously taccDeviceAttributeGcnArchName
    taccDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    taccDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    taccDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched functions
    taccDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched grid dimensions
    taccDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched block dimensions
    taccDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
                                                                ///< devices with unmatched shared memories
    taccDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    taccDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    taccDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
                                                                ///< hipStreamWaitValue64(), '0' otherwise.
    taccDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    taccDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
                                                                ///< units for the device
    taccDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    taccDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    taccDeviceAttributeAmdSpecificEnd = 19999,
    taccDeviceAttributeVendorSpecificBegin = 20000
} taccDevice_attribute;
taccError_t taccDeviceGetAttribute(int* flag, taccDevice_attribute attrib, int device);


typedef enum taccMemAllocationHandleType {
    taccMemHandleTypeNone                    = 0x0,  ///< Does not allow any export mechanism
    taccMemHandleTypePosixFileDescriptor     = 0x1,  ///< Allows a file descriptor for exporting. Permitted only on POSIX systems
    taccMemHandleTypeWin32                   = 0x2,  ///< Allows a Win32 NT handle for exporting. (HANDLE)
    taccMemHandleTypeWin32Kmt                 = 0x4,  ///< Allows a Win32 KMT handle for exporting. (D3DKMT_HANDLE)
    taccMemHandleTypeFabric                   = 0x8,  ///< Allows a fabric handle to be used for exporting.
} taccMemAllocationHandleType;


typedef void* taccEvent_t;
typedef void* taccStream_t;

taccError_t taccStreamCreateWithFlags(taccStream_t* pStream, unsigned int flags);

typedef enum taccMemcpyKind {
    taccMemcpyHostToHost = 0,	
    taccMemcpyHostToDevice,	
    taccMemcpyDeviceToHost,
    taccMemcpyDeviceToDevice,
    taccMemcpyDefault
} taccMemcpyKind;

taccError_t taccMemcpy(void* dst, const void* src, size_t count, taccMemcpyKind kind);
taccError_t taccMemcpyAsync(void* dst, const void* src, size_t count, taccMemcpyKind kind, taccStream_t stream = 0);
taccError_t taccFree(void* ptr);
taccError_t taccMemset(void *dst, int value, size_t sizeBytes);
taccError_t taccMemsetAsync(void* dst, int value, size_t sizeBytes, taccStream_t stream = 0);
taccError_t taccStreamQuery(taccStream_t stream);
taccError_t taccStreamSynchronize(taccStream_t stream);
taccError_t taccStreamDestroy(taccStream_t stream);
taccError_t taccEventCreate(taccEvent_t* event);
taccError_t taccEventCreateWithFlags(taccEvent_t* event, unsigned flags);
taccError_t taccEventQuery(taccEvent_t event);
taccError_t taccEventRecord(taccEvent_t event, taccStream_t stream);
taccError_t taccStreamWaitEvent(taccStream_t stream, taccEvent_t event, unsigned int flags);
taccError_t taccEventDestroy(taccEvent_t event);

taccError_t taccLaunchHostFunc(taccStream_t stream, taccHostFn_t fn, void *userData);

/*
    How to def dim3?
    HIP:
    typedef struct dim3 {
        uint32_t x;  ///< x
        uint32_t y;  ///< y
        uint32_t z;  ///< z
    #ifdef __cplusplus
        constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
    #endif
    } dim3;

    CUDA:
    struct __device_builtin__ dim3
    {
        unsigned int x, y, z;
    #if defined(__cplusplus)
    #if __cplusplus >= 201103L
        __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
        __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        __host__ __device__ constexpr operator uint3(void) const { return uint3{x, y, z}; }
    #else
        __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
        __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
        __host__ __device__ operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
    #endif
    #endif 
    };

    typedef __device_builtin__ struct dim3 dim3;
*/

/*
#define HIP_IPC_HANDLE_SIZE 64
typedef struct hipIpcMemHandle_st {
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;

#define CUDA_IPC_HANDLE_SIZE 64
typedef __device_builtin__ struct __device_builtin__ cudaIpcMemHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;
*/

#define TACC_IPC_HANDLE_SIZE 64
typedef struct taccIpcMemHandle_st 
{
    char reserved[TACC_IPC_HANDLE_SIZE];
} taccIpcMemHandle_t;


taccError_t taccIpcOpenMemHandle(void** devPtr, taccIpcMemHandle_t handle, unsigned int flags);
taccError_t taccIpcCloseMemHandle(void* devPtr);
taccError_t taccIpcGetMemHandle(taccIpcMemHandle_t* handle, void* devPtr);

taccError_t taccHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
taccError_t taccHostUnregister(void* hostPtr);

taccError_t taccHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);

typedef struct taccUUID_t {
    char bytes[16];
} taccUUID;

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch;

typedef struct taccDeviceProp_t {
    char name[256];                   ///< Device name.
    taccUUID uuid;                     ///< UUID of a device
    char luid[8];                     ///< 8-byte unique identifier. Only valid on windows
    
    unsigned int luidDeviceNodeMask;  ///< LUID node mask
    size_t totalGlobalMem;            ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;         ///< Size of shared memory region (in bytes).
    int regsPerBlock;                 ///< Registers per block.
    int warpSize;                     ///< Warp size.
    size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
                                      ///< pitched memory
    int maxThreadsPerBlock;           ///< Max work items per work group or workgroup max size.
    
    int maxThreadsDim[3];             ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];               ///< Max grid dimensions (XYZ).
    
    int clockRate;                    ///< Max clock frequency of the multiProcessors in khz.
    size_t totalConstMem;             ///< Size of shared memory region (in bytes).
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
                ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
                ///< feature caps.
    size_t textureAlignment;       ///< Alignment requirement for textures
    size_t texturePitchAlignment;  ///< Pitch alignment requirement for texture references bound to
    int deviceOverlap;             ///< Deprecated. Use asyncEngineCount instead
    int multiProcessorCount;       ///< Number of multi-processors (compute units).
    int kernelExecTimeoutEnabled;  ///< Run time limit for kernels executed on the device
    int integrated;                ///< APU vs dGPU
    int canMapHostMemory;          ///< Check whether TACC can map host memory
    int computeMode;               ///< Compute mode.
    int maxTexture1D;              ///< Maximum number of elements in 1D images
    int maxTexture1DMipmap;        ///< Maximum 1D mipmap texture size
    int maxTexture1DLinear;        ///< Maximum size for 1D textures bound to linear memory
    
    int maxTexture2D[2];  ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture2DMipmap[2];  ///< Maximum number of elements in 2D array mipmap of images
    int maxTexture2DLinear[3];  ///< Maximum 2D tex dimensions if tex are bound to pitched memory
    int maxTexture2DGather[2];  ///< Maximum 2D tex dimensions if gather has to be performed
    int maxTexture3D[3];  ///< Maximum dimensions (width, height, depth) of 3D images, in image
    ///< elements
    int maxTexture3DAlt[3];           ///< Maximum alternate 3D texture dims
    
    int maxTextureCubemap;            ///< Maximum cubemap texture dims
    int maxTexture1DLayered[2];       ///< Maximum number of elements in 1D array images
    int maxTexture2DLayered[3];       ///< Maximum number of elements in 2D array images
    int maxTextureCubemapLayered[2];  ///< Maximum cubemaps layered texture dims
    
    int maxSurface1D;                 ///< Maximum 1D surface size
    int maxSurface2D[2];              ///< Maximum 2D surface size
    int maxSurface3D[3];              ///< Maximum 3D surface size
    int maxSurface1DLayered[2];       ///< Maximum 1D layered surface size
    int maxSurface2DLayered[3];       ///< Maximum 2D layared surface size
    int maxSurfaceCubemap;            ///< Maximum cubemap surface size
    int maxSurfaceCubemapLayered[2];  ///< Maximum cubemap layered surface size
    
    size_t surfaceAlignment;          ///< Alignment requirement for surface
    int concurrentKernels;         ///< Device can possibly execute multiple kernels concurrently.
    int ECCEnabled;                ///< Device has ECC support enabled
    int pciBusID;                  ///< PCI Bus ID.
    int pciDeviceID;               ///< PCI Device ID.
    int pciDomainID;               ///< PCI Domain ID
    int tccDriver;                 ///< 1:If device is Tesla device using TCC driver, else 0
    int asyncEngineCount;          ///< Number of async engines
    int unifiedAddressing;         ///< Does device and host share unified address space
    int memoryClockRate;           ///< Max global memory clock frequency in khz.
    int memoryBusWidth;            ///< Global memory bus width in bits.
    int l2CacheSize;               ///< L2 cache size.
    int persistingL2CacheMaxSize;  ///< Device's max L2 persisting lines in bytes
    int maxThreadsPerMultiProcessor;    ///< Maximum resident threads per multi-processor.
    int streamPrioritiesSupported;      ///< Device supports stream priority
    int globalL1CacheSupported;         ///< Indicates globals are cached in L1
    int localL1CacheSupported;          ///< Locals are cahced in L1
    size_t sharedMemPerMultiprocessor;  ///< Amount of shared memory available per multiprocessor.
    int regsPerMultiprocessor;          ///< registers available per multiprocessor
    int managedMemory;         ///< Device supports allocating managed memory on this system
    int isMultiGpuBoard;       ///< 1 if device is on a multi-GPU board, 0 if not.
    int multiGpuBoardGroupID;  ///< Unique identifier for a group of devices on same multiboard GPU
    int hostNativeAtomicSupported;         ///< Link between host and device supports native atomics
    int singleToDoublePrecisionPerfRatio;  ///< Deprecated. CUDA only.
    int pageableMemoryAccess;              ///< Device supports coherently accessing pageable memory
                                           ///< without calling taccHostRegister on it
    int concurrentManagedAccess;  ///< Device can coherently access managed memory concurrently with
                                  ///< the CPU
    int computePreemptionSupported;         ///< Is compute preemption supported on the device
    int canUseHostPointerForRegisteredMem;  ///< Device can access host registered memory with same
                                            ///< address as the host
    int cooperativeLaunch;                  ///< TACC device supports cooperative launch
    int cooperativeMultiDeviceLaunch;       ///< TACC device supports cooperative launch on multiple
                                            ///< devices
    size_t
        sharedMemPerBlockOptin;  ///< Per device m ax shared mem per block usable by special opt in
    int pageableMemoryAccessUsesHostPageTables;  ///< Device accesses pageable memory via the host's
                                                 ///< page tables
    int directManagedMemAccessFromHost;  ///< Host can directly access managed memory on the device
                                         ///< without migration
    int maxBlocksPerMultiProcessor;      ///< Max number of blocks on CU
    int accessPolicyMaxWindowSize;       ///< Max value of access policy window
    size_t reservedSharedMemPerBlock;    ///< Shared memory reserved by driver per block
    int hostRegisterSupported;           ///< Device supports taccHostRegister
    int sparseTaccArraySupported;         ///< Indicates if device supports sparse tacc arrays
    int hostRegisterReadOnlySupported;   ///< Device supports using the taccHostRegisterReadOnly flag
                                         ///< with taccHostRegistger
    int timelineSemaphoreInteropSupported;  ///< Indicates external timeline semaphore support
    int memoryPoolsSupported;  ///< Indicates if device supports taccMallocAsync and taccMemPool APIs
    int gpuDirectRDMASupported;                    ///< Indicates device support of RDMA APIs
    unsigned int gpuDirectRDMAFlushWritesOptions;  ///< Bitmask to be interpreted according to
                                                   ///< taccFlushGPUDirectRDMAWritesOptions
    int gpuDirectRDMAWritesOrdering;               ///< value of taccGPUDirectRDMAWritesOrdering
    unsigned int
        memoryPoolSupportedHandleTypes;  ///< Bitmask of handle types support with mempool based IPC
    int deferredMappingTaccArraySupported;  ///< Device supports deferred mapping TACC arrays and TACC
                                           ///< mipmapped arrays
    int ipcEventSupported;                 ///< Device supports IPC events
    int clusterLaunch;                     ///< Device supports cluster launch
    int unifiedFunctionPointers;           ///< Indicates device supports unified function pointers
    int reserved[63];                      ///< CUDA Reserved.

    int taccReserved[32];  ///< Reserved for adding new entries for TACC/HIP/CUDA.

    /* HIP Only struct members */
    char gcnArchName[256];                    ///< AMD GCN Arch Name. HIP Only.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per CU. HIP Only.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch arch;      ///< Architectural feature flags.  New for HIP.
    unsigned int* hdpMemFlushCntl;            ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;            ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    int cooperativeMultiDeviceUnmatchedFunc;  ///< HIP device supports cooperative launch on
                                              ///< multiple
                                              /// devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;    ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;   ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;  ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched shared memories
    int isLargeBar;                                ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                              ///< Revision of the GPU in this device
} taccDeviceProp_t;

taccError_t taccGetDeviceProperties(taccDeviceProp_t* prop, int deviceId);
taccError_t taccDeviceGetPCIBusId(char* pciBusId, int len, int device);
taccError_t taccDeviceGetByPCIBusId(int* device, const char* pciBusId);
taccError_t taccDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
taccError_t taccGetDeviceCount(int* count);

/*
typedef enum hipFuncAttribute {
    hipFuncAttributeMaxDynamicSharedMemorySize = 8,
    hipFuncAttributePreferredSharedMemoryCarveout = 9,
    hipFuncAttributeMax
} hipFuncAttribute;

enum __device_builtin__ cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, 
    cudaFuncAttributeClusterDimMustBeSet = 10, 
    cudaFuncAttributeRequiredClusterWidth = 11, 
    cudaFuncAttributeRequiredClusterHeight = 12, 
    cudaFuncAttributeRequiredClusterDepth = 13, 
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
    cudaFuncAttributeMax
};
*/

typedef enum taccFuncAttribute {
    taccFuncAttributeMaxDynamicSharedMemorySize = 8,
    taccFuncAttributePreferredSharedMemoryCarveout = 9,
    taccFuncAttributeMax
} taccFuncAttribute;

typedef struct taccFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    int clusterDimMustBeSet; // CUDA only
    int clusterSchedulingPolicyPreference; // CUDA only
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int nonPortableClusterSizeAllowed; // CUDA only
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    int requiredClusterWidth; // CUDA only
    int reserved[16]; // CUDA only
    size_t sharedSizeBytes;
} taccFuncAttributes;

taccError_t taccFuncSetAttribute(const void* func, taccFuncAttribute attr, int value);
taccError_t taccFuncGetAttributes(taccFuncAttributes* attr, const void* func);

typedef struct dim3_t {
    unsigned int x, y, z;
} dim3_t;

taccError_t taccExtLaunchKernel(const void* function_address, dim3_t numBlocks,
                                dim3_t dimBlocks, void** args, size_t sharedMemBytes,
                                taccStream_t stream, taccEvent_t startEvent,
                                taccEvent_t stopEvent, int flags);

taccError_t taccLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim, void** args, size_t sharedMem, taccStream_t stream);

/*
cudaLimitStackSize = 0x00
GPU thread stack size
cudaLimitPrintfFifoSize = 0x01
GPU printf FIFO size
cudaLimitMallocHeapSize = 0x02
GPU malloc heap size
cudaLimitDevRuntimeSyncDepth = 0x03
GPU device runtime synchronize depth
cudaLimitDevRuntimePendingLaunchCount = 0x04
GPU device runtime pending launch count
cudaLimitMaxL2FetchGranularity = 0x05
A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
cudaLimitPersistingL2CacheSize = 0x06
A size in bytes for L2 persisting lines cache size
*/

typedef enum taccLimit{
  taccLimitStackSize = 0x0,
  taccLimitPrintfFifoSize = 0x01,
  taccLimitMallocHeapSize = 0x02,
  taccLimitRange
} taccLimit;

taccError_t taccDeviceSetLimit(taccLimit limit, size_t value);

typedef enum taccStreamCaptureStatus{
    taccStreamCaptureStatusNone = 0,
    taccStreamCaptureStatusActive,
    taccStreamCaptureStatusInvalidated
} taccStreamCaptureStatus;

typedef void* taccGraph_t;
typedef void* taccGraphNode_t;
typedef void* taccGraphExec_t;

taccError_t taccStreamGetCaptureInfo_v2(taccStream_t stream, taccStreamCaptureStatus* captureStatus_out, 
                                       unsigned long long* id_out = 0, taccGraph_t* graph_out = 0,
                                       const taccGraphNode_t** dependencies_out = 0, size_t* numDependencies_out = 0);

taccError_t taccGraphAddHostNode(taccGraphNode_t* pGraphNode, taccGraph_t graph, const taccGraphNode_t* pDependencies, size_t numDependencies, const void* pNodeParams);
taccError_t taccGraphInstantiate(taccGraphExec_t* pGraphExec, taccGraph_t graph, taccGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
taccError_t taccGraphLaunch(taccGraphExec_t graphExec, taccStream_t stream);
taccError_t taccGraphExecDestroy(taccGraphExec_t graphExec);
taccError_t taccGraphDestroy(taccGraph_t graph);
taccError_t taccDeviceSynchronize();
taccError_t taccStreamBeginCapture(taccStream_t stream, taccStreamCaptureMode mode);
taccError_t taccStreamEndCapture(taccStream_t stream, taccGraph_t* graph);
taccError_t taccDeviceReset();

typedef enum taccMemoryType {
    taccMemoryTypeUnregistered = 0,  ///< Unregistered memory
    taccMemoryTypeHost         = 1,  ///< Memory is physically located on host
    taccMemoryTypeDevice       = 2,  ///< Memory is physically located on device. (see deviceId for
                                    ///< specific device)
    taccMemoryTypeManaged      = 3,  ///< Managed memory, automaticallly managed by the unified
                                    ///< memory system
                                    ///< place holder for new values.
    taccMemoryTypeArray        = 10, ///< Array memory, physically located on device. (see deviceId for
                                    ///< specific device)
    taccMemoryTypeUnified      = 11  ///< unified address space

} taccMemoryType;

typedef struct taccPointerAttribute_t {
    enum taccMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
} taccPointerAttribute_t;

taccError_t taccPointerGetAttributes(taccPointerAttribute_t* attributes, const void* ptr);

/**
 * @brief Returns available platforms.
 * 
 * "Available" means that
 * 1. The platform is installed on the system (e.g., libcudart.so is available for CUDA)
 * 2. At least one device is available on the platform
 *
 * @param num_entries The number of entries that can be added to platforms. If platforms is not NULL, num_enteries must be greater than 0.
 * @param platforms Returns the available platforms. If NULL, the argument is ignored. The number of platforms returned is the minimum of num_entries and the number of available platforms.
 * @param num_platforms Returnes the number of available platforms. If NULL, the argument is ignored.
 * @return taccError_t 
 */
taccError_t taccGetAvailablePlatforms(int num_entries, taccPlatform* platforms, int* num_platforms);

/**
 * @brief Set the current platform. Future tacc calls will be relayed to the current platform.
 * 
 * @param platform 
 * @return taccError_t 
 */
taccError_t taccSetPlatform(taccPlatform platform);

/**
 * @brief Automatically set platform based on the environment
 *
 * If only one of the platforms is available, it will be set as the current platform.
 * If not, return 
 *
 * @return taccError_t 
 */
taccError_t taccSetPlatformAuto();

/**
 * @brief Get current platform.
 * 
 * @param platform 
 * @return taccError_t 
 */
taccError_t taccGetPlatform(taccPlatform* platform);

/**
 * @brief Create an asynchronous stream with the specified CU mask.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] cuMaskSize Size of CU mask bit array passed in.
 * @param[in ] cuMask Bit-vector representing the CU mask. Each active bit represents using one CU.
 * The first 32 bits represent the first 32 CUs, and so on. If its size is greater than physical
 * CU number (i.e., multiProcessorCount member of hipDeviceProp_t), the extra elements are ignored.
 * It is user's responsibility to make sure the input is meaningful.
 * @return #hipSuccess, #hipErrorInvalidHandle, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream with the specified CU mask.  @p stream returns an opaque handle
 * that can be used to reference the newly created stream in subsequent hipStream* commands.  The
 * stream is allocated on the heap and will remain allocated even if the handle goes out-of-scope.
 * To release the memory used by the stream, application must call hipStreamDestroy.
 *
 *
 * @see hipStreamCreate, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
taccError_t taccExtStreamCreateWithCUMask(taccStream_t* stream, uint32_t cuMaskSize, const uint32_t* cuMask);

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Valid pointer to hipStream_t.  This function writes the memory with the
 * newly created stream.
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent hipStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, applicaiton must call hipStreamDestroy.
 *
 * @return #hipSuccess, #hipErrorInvalidValue
 *
 * @see hipStreamCreateWithFlags, hipStreamCreateWithPriority, hipStreamSynchronize, hipStreamWaitEvent, hipStreamDestroy
 */
taccError_t taccStreamCreate(taccStream_t* stream);

/**
 * @brief Returns the link type and hop count between two devices
 *
 * @param [in] device1 Ordinal for device1
 * @param [in] device2 Ordinal for device2
 * @param [out] linktype Returns the link type (See hsa_amd_link_info_type_t) between the two devices
 * @param [out] hopcount Returns the hop count between the two devices
 *
 * Queries and returns the HSA link type and the hop count between the two specified devices.
 *
 * @returns #hipSuccess, #hipInvalidDevice, #hipErrorRuntimeOther
 */
taccError_t taccExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t* hopcount);

/**
 *  @brief Allocates device accessible page locked (pinned) host memory
 *
 *  This API allocates pinned host memory which is mapped into the address space of all GPUs
 *  in the system, the memory can be accessed directly by the GPU device, and can be read or
 *  written with much higher bandwidth than pageable memory obtained with functions such as
 *  malloc().
 *
 *  Using the pinned host memory, applications can implement faster data transfers for HostToDevice
 *  and DeviceToHost. The runtime tracks the hipHostMalloc allocations and can avoid some of the
 *  setup required for regular unpinned memory.
 *
 *  When the memory accesses are infrequent, zero-copy memory can be a good choice, for coherent
 *  allocation. GPU can directly access the host memory over the CPU/GPU interconnect, without need
 *  to copy the data.
 *
 *  Currently the allocation granularity is 4KB for the API.
 *
 *  Developers need to choose proper allocation flag with consideration of synchronization.
 *
 *  @param[out] ptr Pointer to the allocated host pinned memory
 *  @param[in]  size Requested memory size in bytes
 *  If size is 0, no memory is allocated, *ptr returns nullptr, and hipSuccess is returned.
 *  @param[in]  flags Type of host memory allocation
 *
 *  If no input for flags, it will be the default pinned memory allocation on the host.
 *
 *  @return #hipSuccess, #hipErrorOutOfMemory
 *
 *  @see hipSetDeviceFlags, hipHostFree
 */
taccError_t taccHostMalloc(void** ptr, size_t size, unsigned int flags);

/**
* Host memory allocation will follow numa policy set by user.
* @note  This numa allocation flag is applicable on Linux, under development on Windows.
*/
#define taccHostMallocNumaUser  0x20000000

/** Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define taccHostMallocCoherent  0x40000000

/** Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for specific allocation.*/
#define taccHostMallocNonCoherent  0x80000000

/**
 *  @brief Free memory allocated by the hcc hip host memory allocation API
 *  This API performs an implicit hipDeviceSynchronize() call.
 *  If pointer is NULL, the hip runtime is initialized and hipSuccess is returned.
 *
 *  @param[in] ptr Pointer to memory to be freed
 *  @return #hipSuccess,
 *          #hipErrorInvalidValue (if pointer is invalid, including device pointers allocated with
 * hipMalloc)
 *
 *  @see hipMalloc, hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D,
 * hipMalloc3DArray, hipHostMalloc
 */
taccError_t taccHostFree(void* ptr);

/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]   start : Start event.
 * @param[in]   stop  : Stop event.
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotReady, #hipErrorInvalidHandle,
 * #hipErrorNotInitialized, #hipErrorLaunchFailure
 *
 * Computes the elapsed time between two events. Time is computed in ms, with
 * a resolution of approximately 1 us.
 *
 * Events which are recorded in a NULL stream will block until all commands
 * on all other streams complete execution, and then record the timestamp.
 *
 * Events which are recorded in a non-NULL stream will record their timestamp
 * when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.  Thus the time that
 * the event recorded may be significantly after the host calls hipEventRecord().
 *
 * If hipEventRecord() has not been called on either event, then #hipErrorInvalidHandle is
 * returned. If hipEventRecord() has been called on both events, but the timestamp has not yet been
 * recorded on one or both events (that is, hipEventQuery() would return #hipErrorNotReady on at
 * least one of the events), then #hipErrorNotReady is returned.
 *
 * @see hipEventCreate, hipEventCreateWithFlags, hipEventQuery, hipEventDestroy, hipEventRecord,
 * hipEventSynchronize
 */
taccError_t taccEventElapsedTime(float* ms, taccEvent_t start, taccEvent_t stop);

#ifdef __cplusplus
}
#endif
