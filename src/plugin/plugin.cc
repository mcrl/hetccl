#include "plugin.h"
#include "debug.h"
#include <dlfcn.h>
#include "comm.h"

static void* hip_handle = nullptr;
static void* cuda_handle = nullptr;

ncclResult_t scclPluginInit(ncclComm_t comm, taccPlatform backend) {
  if (backend == taccPlatformHip) {
    if (!hip_handle) {
      const char* fn = "libsccl_hip.so";
      hip_handle = dlopen(fn, RTLD_LAZY);
      if (!hip_handle) {
        WARN("Failed to load %s: %s", fn, dlerror());
        return ncclInternalError;
      }
    }
    comm->scclPlugin = (scclPlugin_t*)dlsym(hip_handle, "scclPluginHip");
    if (!comm->scclPlugin) {
      WARN("Failed to find scclPluginHip in libsccl_hip.so: %s", dlerror());
      return ncclInternalError;
    }
    //comm->mscclPlugin = (scclPlugin_t*)dlsym(hip_handle, "scclPluginHipMsccl");
    //if (!comm->mscclPlugin) {
    //  WARN("Failed to find scclPluginHipMsccl in libsccl_hip.so: %s", dlerror());
    //  return ncclInternalError;
    //}
    return ncclSuccess;
  } else if (backend == taccPlatformCuda) {
    if (!cuda_handle) {
      const char* fn = "libsccl_cuda.so";
      cuda_handle = dlopen(fn, RTLD_LAZY);
      if (!cuda_handle) {
        WARN("Failed to load %s: %s", fn, dlerror());
        return ncclInternalError;
      }
    }
    comm->scclPlugin = (scclPlugin_t*)dlsym(cuda_handle, "scclPluginCuda");
    if (!comm->scclPlugin) {
      WARN("Failed to find scclPluginCuda in libsccl_cuda.so: %s", dlerror());
      return ncclInternalError;
    }
    return ncclSuccess;
  }
  WARN("Unsupported backend %d", backend);
  return ncclInternalError;
}

