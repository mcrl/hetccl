/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#include "verifiable.h"
#include "verifiable_cuda.h"
#include "verifiable_hip.h"
#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>
#include "common.h"

/* Routines for launching kernels that verify reduction results. A significant
 * feature of these routines is they carefully craft floating point input
 * to produce exactly predictable output.
 *
 * int elt_ty: actually just a ncclDataType_t
 *
 * int red_op: mostly just a  ncclRedOp_t. Since PreMulSum ops are dynamically
 * created, these are encoded as the value ncclNumOps and their scalar is
 * assumed to be `ncclVerifiablePremulScalar(rank_me)`
 *
 * uint64_t seed: arbitrary 64-bits to use in seeding the random values
 *
 * intptr_t elt_ix0: index of first element pointed to by elts when generating
 * random values. This makes it possible to generate subsequences independently
 * as well as in aggregate.
 *
 * int rank_n: Number of contributions into the reduction. Non-reduction
 * collectives like broadcast, gather, etc will always set this to one.
 *
 * int rank_me: Index of this contribution
 */

static struct {
  void* handle = nullptr;
  #define CMD(name, ret_type, ...) \
    ret_type (*name)(__VA_ARGS__) = nullptr
  #include "verifiable_hip_func_list.h"
  #undef CMD
} hipFuncList;

static struct {
  void* handle = nullptr;
  #define CMD(name, ret_type, ...) \
    ret_type (*name)(__VA_ARGS__) = nullptr
  #include "verifiable_cuda_func_list.h"
  #undef CMD
} cudaFuncList;

static void init() {
  taccPlatform platform;
  taccGetPlatform(&platform);
  if (platform == taccPlatformHip) {
    if (hipFuncList.handle) return;
    const char* fn = "libsccl_tests_hip.so";
    hipFuncList.handle = dlopen(fn, RTLD_LAZY);
    if (!hipFuncList.handle) {
      fprintf(stderr, "Failed to load %s: %s\n", fn, dlerror());
      exit(1);
    }
    #define CMD(name, ret_type, ...) \
      do { \
        hipFuncList.name = (ret_type(*)(__VA_ARGS__))dlsym(hipFuncList.handle, #name); \
        if (!hipFuncList.name) { \
          fprintf(stderr, "Failed to load %s: %s\n", #name, dlerror()); \
        } \
      } while (0)
    #include "verifiable_hip_func_list.h"
    #undef CMD
  } else if (platform == taccPlatformCuda) {
    if (cudaFuncList.handle) return;
    const char* fn = "libsccl_tests_cuda.so";
    cudaFuncList.handle = dlopen(fn, RTLD_LAZY);
    if (!cudaFuncList.handle) {
      fprintf(stderr, "Failed to load %s: %s\n", fn, dlerror());
      exit(1);
    }
    #define CMD(name, ret_type, ...) \
      do { \
        cudaFuncList.name = (ret_type(*)(__VA_ARGS__))dlsym(cudaFuncList.handle, #name); \
        if (!cudaFuncList.name) { \
          fprintf(stderr, "Failed to load %s: %s\n", #name, dlerror()); \
        } \
      } while (0)
    #include "verifiable_cuda_func_list.h"
    #undef CMD
  }
}

// Enqueue kernel to generate data which is to be reduced.
void ncclVerifiablePrepareInput(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n, int rank_me,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream
) {
  init();
  taccPlatform platform;
  taccGetPlatform(&platform);
  if (platform == taccPlatformHip) {
    hipFuncList.hipVerifiablePrepareInput(elts, elt_n, elt_ty, red_op, rank_n, rank_me, seed, elt_ix0, stream);
  } else if (platform == taccPlatformCuda) {
    cudaFuncList.cudaVerifiablePrepareInput(elts, elt_n, elt_ty, red_op, rank_n, rank_me, seed, elt_ix0, stream);
  }
}

// Enqueue kernel to generate expected results of reduction.
void ncclVerifiablePrepareExpected(
  void *elts, intptr_t elt_n, int elt_ty, int red_op, int rank_n,
  uint64_t seed, intptr_t elt_ix0, taccStream_t stream
) {
  init();
  taccPlatform platform;
  taccGetPlatform(&platform);
  if (platform == taccPlatformHip) {
    hipFuncList.hipVerifiablePrepareExpected(elts, elt_n, elt_ty, red_op, rank_n, seed, elt_ix0, stream);
  } else if (platform == taccPlatformCuda) {
    cudaFuncList.cudaVerifiablePrepareExpected(elts, elt_n, elt_ty, red_op, rank_n, seed, elt_ix0, stream);
  }
}

// Enqueue kernel to verify reduced data matches expectation. The number of
// failed elements is written to bad_elt_n which must be in cudaHost memory.
// If `expected == nullptr` then the expected results are generated on-the-fly
// which can be costly. Thus if you plan to run the same reduction multiple
// times it is advantageous to precompute the expected values with
// ncclVerifiablePrepareExpected and pass them as `expected` here.
void ncclVerifiableVerify(
  void const *results, void const *expected, intptr_t elt_n, int elt_ty,
  int red_op, int rank_n, uint64_t seed, intptr_t elt_ix0,
  int64_t *bad_elt_n, taccStream_t stream
) {
  init();
  taccPlatform platform;
  taccGetPlatform(&platform);
  if (platform == taccPlatformHip) {
    hipFuncList.hipVerifiableVerify(results, expected, elt_n, elt_ty, red_op, rank_n, seed, elt_ix0, bad_elt_n, stream);
  } else if (platform == taccPlatformCuda) {
    cudaFuncList.cudaVerifiableVerify(results, expected, elt_n, elt_ty, red_op, rank_n, seed, elt_ix0, bad_elt_n, stream);
  }
}
