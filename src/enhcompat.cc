/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* Define weak symbols used to allow libnccl_static.a to work with older libcudart_static.a */

enum hipError_t { cudaErrorStubLibrary = 34 };

extern "C" {

hipError_t cudaStreamGetCaptureInfo_v2(...)         __attribute__((visibility("hidden"))) __attribute((weak));
hipError_t cudaStreamGetCaptureInfo_v2(...)         { return cudaErrorStubLibrary; }

hipError_t hipUserObjectCreate(...)                __attribute__((visibility("hidden"))) __attribute((weak));
hipError_t hipUserObjectCreate(...)                { return cudaErrorStubLibrary; }

hipError_t hipGraphRetainUserObject(...)           __attribute__((visibility("hidden"))) __attribute((weak));
hipError_t hipGraphRetainUserObject(...)           { return cudaErrorStubLibrary; }

hipError_t cudaStreamUpdateCaptureDependencies(...) __attribute__((visibility("hidden"))) __attribute((weak));
hipError_t cudaStreamUpdateCaptureDependencies(...) { return cudaErrorStubLibrary; }

hipError_t cudaGetDriverEntryPoint(...)             __attribute__((visibility("hidden"))) __attribute((weak));
hipError_t cudaGetDriverEntryPoint(...)             { return cudaErrorStubLibrary; }

}
