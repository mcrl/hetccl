#include "plugin.h"
#include "debug.h"
#include <dlfcn.h>
#include "comm.h"

static void* hip_handle = nullptr;
static void* cuda_handle = nullptr;

ncclResult_t hetcclPluginInit(ncclComm_t comm, taccPlatform backend) {
  if (backend == taccPlatformHip) {
    if (!hip_handle) {
      const char* fn = "libhetccl_hip.so";
      hip_handle = dlopen(fn, RTLD_LAZY);
      if (!hip_handle) {
        WARN("Failed to load %s: %s", fn, dlerror());
        return ncclInternalError;
      }
    }
    comm->hetcclPlugin = (hetcclPlugin_t*)dlsym(hip_handle, "hetcclPluginHip");
    if (!comm->hetcclPlugin) {
      WARN("Failed to find hetcclPluginHip in libhetccl_hip.so: %s", dlerror());
      return ncclInternalError;
    }
    //comm->mscclPlugin = (hetcclPlugin_t*)dlsym(hip_handle, "hetcclPluginHipMsccl");
    //if (!comm->mscclPlugin) {
    //  WARN("Failed to find hetcclPluginHipMsccl in libhetccl_hip.so: %s", dlerror());
    //  return ncclInternalError;
    //}
    return ncclSuccess;
  } else if (backend == taccPlatformCuda) {
    if (!cuda_handle) {
      const char* fn = "libhetccl_cuda.so";
      cuda_handle = dlopen(fn, RTLD_LAZY);
      if (!cuda_handle) {
        WARN("Failed to load %s: %s", fn, dlerror());
        return ncclInternalError;
      }
    }
    comm->hetcclPlugin = (hetcclPlugin_t*)dlsym(cuda_handle, "hetcclPluginCuda");
    if (!comm->hetcclPlugin) {
      WARN("Failed to find hetcclPluginCuda in libhetccl_cuda.so: %s", dlerror());
      return ncclInternalError;
    }
    return ncclSuccess;
  }
  WARN("Unsupported backend %d", backend);
  return ncclInternalError;
}

