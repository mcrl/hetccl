# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 3.5)

# RCCL project
#==================================================================================================
project(sccl CXX)

# Build options
#==================================================================================================
option(BUILD_ADDRESS_SANITIZER                 "Enable address sanitizer"                      OFF)
option(BUILD_ALLREDUCE_ONLY                    "AllReduce(sum,float) kernel only"              OFF)
option(BUILD_BFD                               "Enable custom backtrace (if bfd.h exists)"     OFF)
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "File/folder reorg with backward compatibility" OFF)
option(BUILD_LOCAL_GPU_TARGET_ONLY             "Build only for GPUs detected on this machine"  OFF)
option(BUILD_SHARED_LIBS                       "Build as shared library"                       ON)
option(BUILD_TESTS                             "Build unit test programs"                      OFF)
option(COLLTRACE                               "Collective Trace Option"                       ON)
option(ENABLE_MSCCL_KERNEL                     "Enable MSCCL while compiling"                  OFF)
option(ENABLE_IFC                              "Enable indirect function call"                 ON)
option(INSTALL_DEPENDENCIES                    "Force install dependencies"                    OFF)
option(PROFILE                                 "Enable profiling"                              OFF)
option(TIMETRACE                               "Enable time-trace during compilation"          OFF)
option(TRACE                                   "Enable additional tracing"                     OFF)

# Load CMake modules
#==================================================================================================
include(ExternalProject)

# Determine version from makefiles/version.mk and fill in templates
#==================================================================================================
## parse version from Makefile NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH must exist
## NCCL_SUFFIX is optional
## NCCL_VERSION formatting is ((X) * 1000 + (Y) * 100 + (Z)) so we must first detect one or two digits first
file(READ makefiles/version.mk version_mk_text)
if("${version_mk_text}" MATCHES "NCCL_MAJOR *:= *([0-9]*)")
  set(NCCL_MAJOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MAJOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_MINOR *:= *([0-9]*)")
  set(NCCL_MINOR ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_MINOR")
endif()
if("${version_mk_text}" MATCHES "NCCL_PATCH *:= *([0-9]*)")
  set(NCCL_PATCH ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse NCCL_PATCH")
endif()
if("${version_mk_text}" MATCHES "NCCL_SUFFIX *:= *([0-9]*)")
  set(NCCL_SUFFIX ${CMAKE_MATCH_1})
else()
  set(NCCL_SUFFIX)
endif()
if("${version_mk_text}" MATCHES "PKG_REVISION *:= *([0-9]*)")
  set(PKG_REVISION ${CMAKE_MATCH_1})
else()
  message(FATAL_ERROR "Failed to parse PKG_REVISION")
endif()
if("${NCCL_PATCH}" MATCHES "[0-9][0-9]")
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}${NCCL_PATCH}")
else()
  set(NCCL_VERSION "${NCCL_MAJOR}${NCCL_MINOR}0${NCCL_PATCH}")
endif()

## Setup VERSION
set(VERSION_STRING "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
#rocm_setup_version(VERSION ${VERSION_STRING})
set(VERSION ${VERSION_STRING})

## Fill in version information for main header file
set(STREAM_INCLUDE "#include <hip/hip_runtime.h>")
set(STREAM_TYPE "hipStream_t")
configure_file(src/sccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h) # For linking against HIP programs

set(STREAM_INCLUDE "#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif")
set(STREAM_TYPE "cudaStream_t")
configure_file(src/sccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)      # For linking against CUDA programs

set(STREAM_INCLUDE "")
set(STREAM_TYPE "void*")
configure_file(src/sccl.h.in ${PROJECT_BINARY_DIR}/include/sccl.h)      # Used by some internal files

# Collect list of all source files
#==================================================================================================
# E.g: find src -type f \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
set(SRC_FILES
  src/register.cc
  src/bootstrap.cc
  src/channel.cc
  src/collectives/all_gather.cc
  src/collectives/all_reduce.cc
  src/collectives/all_to_all.cc
  src/collectives/all_to_allv.cc
  src/collectives/broadcast.cc
  src/collectives/msccl.cc
  src/collectives/gather.cc
  src/collectives/reduce.cc
  src/collectives/reduce_scatter.cc
  src/collectives/scatter.cc
  src/collectives/sendrecv.cc
  src/debug.cc
  src/enhcompat.cc
  src/enqueue.cc
  src/graph/connect.cc
  src/graph/paths.cc
  src/graph/rings.cc
  src/graph/rome_models.cc
  src/graph/search.cc
  src/graph/topo.cc
  src/graph/trees.cc
  src/graph/tuning.cc
  src/graph/xml.cc
  src/group.cc
  src/init.cc
  src/misc/archinfo.cc
  src/misc/argcheck.cc
  src/misc/ibvsymbols.cc
  src/misc/ibvwrap.cc
  src/misc/ipcsocket.cc
  src/misc/msccl/msccl_lifecycle.cc
  src/misc/msccl/msccl_parser.cc
  src/misc/msccl/msccl_setup.cc
  src/misc/msccl/msccl_status.cc
  src/misc/npkit.cc
  src/misc/nvmlwrap.cc
  src/misc/param.cc
  src/misc/profiler.cc
  src/misc/rocm_smi_wrap.cc
  src/misc/rocmwrap.cc
  src/misc/shmutils.cc
  src/misc/signals.cc
  src/misc/socket.cc
  src/misc/strongstream.cc
  src/misc/utils.cc
  src/net.cc
  src/proxy.cc
  src/transport.cc
  src/transport/coll_net.cc
  src/transport/net.cc
  src/transport/net_ib.cc
  src/transport/net_socket.cc
  src/transport/nvls.cc
  src/transport/p2p.cc
  src/transport/shm.cc
  src/plugin/plugin.cc
)

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")
list(APPEND SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

# Create a custom target that updates git_version.cpp and executes whenever rccl is built
add_custom_target(git_version_check
  COMMENT "Updating git_version.cpp if necessary"
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  VERBATIM
)

# Set up CUDA parts
#==================================================================================================
ExternalProject_Add(sccl_cuda
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/cuda
  BUILD_ALWAYS true
  CMAKE_ARGS "-DBUILD_ALLREDUCE_ONLY=${BUILD_ALLREDUCE_ONLY}"
  CMAKE_ARGS "-DSCCL_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
  CMAKE_ARGS "-DSCCL_BINARY_DIR=${PROJECT_BINARY_DIR}"
  CMAKE_ARGS "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
)
ExternalProject_Get_Property(sccl_cuda BINARY_DIR)
set(sccl_cuda_binary_dir ${BINARY_DIR})

# Set up HIP parts
#==================================================================================================
ExternalProject_Add(sccl_hip
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/hip
  BUILD_ALWAYS true
  CMAKE_ARGS "-DSCCL_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
  CMAKE_ARGS "-DSCCL_BINARY_DIR=${PROJECT_BINARY_DIR}"
  # User options
  CMAKE_ARGS "-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"
  CMAKE_ARGS "-DENABLE_MSCCL_KERNEL=${ENABLE_MSCCL_KERNEL}"
  # Global options
  CMAKE_ARGS "-DBUILD_ADDRESS_SANITIZER=${BUILD_ADDRESS_SANITIZER}"
  CMAKE_ARGS "-DBUILD_ALLREDUCE_ONLY=${BUILD_ALLREDUCE_ONLY}"
  CMAKE_ARGS "-DBUILD_BFD=${BUILD_BFD}"
  CMAKE_ARGS "-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=${BUILD_FILE_REORG_BACKWARD_COMPATIBILITY}"
  CMAKE_ARGS "-DBUILD_LOCAL_GPU_TARGET_ONLY=${BUILD_LOCAL_GPU_TARGET_ONLY}"
  CMAKE_ARGS "-DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
  CMAKE_ARGS "-DBUILD_TESTS=${BUILD_TESTS}"
  CMAKE_ARGS "-DCOLLTRACE=${COLLTRACE}"
  CMAKE_ARGS "-DENABLE_MSCCL_KERNEL=${ENABLE_MSCCL_KERNEL}"
  CMAKE_ARGS "-DENABLE_IFC=${ENABLE_IFC}"
  CMAKE_ARGS "-DINSTALL_DEPENDENCIES=${INSTALL_DEPENDENCIES}"
  CMAKE_ARGS "-DPROFILE=${PROFILE}"
  CMAKE_ARGS "-DTIMETRACE=${TIMETRACE}"
  CMAKE_ARGS "-DTRACE=${TRACE}"
)
ExternalProject_Get_Property(sccl_hip BINARY_DIR)
set(sccl_hip_binary_dir ${BINARY_DIR})

# Set up RCCL library
#==================================================================================================
## Set RCCL source files
add_library(sccl ${SRC_FILES})
add_dependencies(sccl sccl_hip)
add_dependencies(sccl sccl_cuda)

# Set up TACC
add_subdirectory(deps/tacc)
target_link_libraries(sccl PRIVATE tacc)
target_include_directories(sccl PRIVATE deps/tacc/include)

## Set RCCL dependencies
add_dependencies(sccl git_version_check)                                      # Execute git_version_check during build

## Set RCCL include directories
list(APPEND RCCL_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
list(APPEND RCCL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)              # for hipfied headers
list(APPEND RCCL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/include)
list(APPEND RCCL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/include/ext)
list(APPEND RCCL_INCLUDE_DIRS /usr/local/cuda/include)
list(APPEND RCCL_INCLUDE_DIRS ${HSA_INCLUDE_PATH})
list(APPEND RCCL_INCLUDE_DIRS ${ROCM_SMI_INCLUDE_DIR})
if(DEMANGLE_DIR)
  list(APPEND RCCL_INCLUDE_DIRS ${DEMANGLE_DIR})
endif()

## Set RCCL compile definitions
list(APPEND RCCL_COMPILE_DEFS NVTX_NO_IMPL)                         # NVTX is not supported
if(COLLTRACE)
  list(APPEND RCCL_COMPILE_DEFS ENABLE_COLLTRACE)
endif()
if(ENABLE_MSCCL_KERNEL)
  list(APPEND RCCL_COMPILE_DEFS COMPILE_MSCCL_KERNEL)
endif()
if(HAVE_ROCM_SMI64CONFIG)
  list(APPEND RCCL_COMPILE_DEFS USE_ROCM_SMI64CONFIG)
endif()
if(NPKIT_FLAGS)
  list(APPEND RCCL_COMPILE_DEFS ${NPKIT_FLAGS})
endif()
if(PROFILE)
  list(APPEND RCCL_COMPILE_DEFS ENABLE_PROFILING)
endif()
if(TRACE)
  list(APPEND RCCL_COMPILE_DEFS ENABLE_TRACE)
endif()
if(${HIP_EVENT_DISABLE_FENCE})
  list(APPEND RCCL_COMPILE_DEFS HIP_EVENT_DISABLE_FENCE)
endif()
# Assume uncached memory is supported by default
list(APPEND RCCL_COMPILE_DEFS HIP_UNCACHED_MEMORY)
message(STATUS "HIP_UNCACHED_MEMORY enabled")
if (BUILD_BFD)
  if (HAVE_BFD)
    list(APPEND RCCL_COMPILE_DEFS HAVE_BFD)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_FLAGS)
    list(APPEND RCCL_COMPILE_DEFS HAVE_DECL_BFD_GET_SECTION_FLAGS)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_VMA)
    list(APPEND RCCL_COMPILE_DEFS HAVE_DECL_BFD_GET_SECTION_VMA)
  endif()
  if (HAVE_TWO_ARG_BFD_SECTION_SIZE)
    list(APPEND RCCL_COMPILE_DEFS HAVE_TWO_ARG_BFD_SECTION_SIZE)
  endif()
endif()
if (ENABLE_IFC)
  list(APPEND RCCL_COMPILE_DEFS USE_INDIRECT_FUNCTION_CALL)
endif()
if(DEMANGLE_DIR)
  list(APPEND RCCL_COMPILE_DEFS "HAVE_CPLUS_DEMANGLE=1")
  list(APPEND RCCL_COMPILE_DEFS "HAVE_DECL_BASENAME=1")
endif()

## Set RCCL compile options
list(APPEND RCCL_COMPILE_OPTS -Wno-format-nonliteral)
list(APPEND RCCL_COMPILE_OPTS -fvisibility=default)     # Set symbol visibility to hidden by default
if (HAVE_KERNARG_PRELOAD)
  list(APPEND RCCL_COMPILE_OPTS -mllvm --amdgpu-kernarg-preload-count=16)
endif()

## NOTE: This is currently being handled by rocm-cmake, however may need to be re-enabled in the future
#foreach(target ${GPU_TARGETS})
#  list(APPEND RCCL_COMPILE_OPTS --offload-arch=${target})
#endforeach()

if(BUILD_ADDRESS_SANITIZER)
  list(APPEND RCCL_COMPILE_OPTS -fsanitize=address -shared-libasan)
endif()
if(TIMETRACE)
  list(APPEND RCCL_COMPILE_OPTS -ftime-trace)
endif()

## Set RCCL linked library directories
list(APPEND RCCL_LINK_DIRS ${ROCM_SMI_LIB_DIR})

## Set RCCL linked libraries
if (HAS_BFD)
  list(APPEND RCCL_LINK_LIBS bfd)
  if(HAVE_IBERTY)
    list(APPEND RCCL_LINK_LIBS iberty z)
  endif()
endif()

## Set RCCL link options
if(BUILD_ADDRESS_SANITIZER)
  list(APPEND RCCL_LINK_OPTS -fuse-ld=lld)
endif()
if(TIMETRACE)
  list(APPEND RCCL_LINK_OPTS -ftime-trace)
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
  list(APPEND RCCL_LINK_LIBS --emit-static-lib)
  set(CMAKE_AR "${hipcc_executable}")
  get_property(link_libraries TARGET sccl PROPERTY LINK_LIBRARIES)
  string (REPLACE ";" " " LINK_PROPS "${link_libraries}")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -o <TARGET> ${LINK_PROPS} <LINK_FLAGS> <OBJECTS>")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD)
  list(APPEND RCCL_LINK_OPTS -mllvm --amdgpu-kernarg-preload-count=16)
endif()

list(APPEND RCCL_LINK_LIBS   dl)
list(APPEND RCCL_LINK_LIBS   ${ROCM_SMI_LIBRARIES})
list(APPEND RCCL_LINK_LIBS   pthread)

# Debug
list(APPEND RCCL_COMPILE_OPTS -g)

list(APPEND RCCL_INCLUDE_DIRS /opt/rocm/include)
list(APPEND RCCL_COMPILE_DEFS "__HIP_PLATFORM_AMD__")
if (BUILD_ALLREDUCE_ONLY)
  add_definitions(-DBUILD_ALLREDUCE_ONLY)
endif()

# O0 results in segfault; check github issue
list(APPEND RCCL_COMPILE_OPTS -O3)

# Apply settings to sccl
target_include_directories(sccl PRIVATE ${RCCL_INCLUDE_DIRS})
target_compile_definitions(sccl PRIVATE ${RCCL_COMPILE_DEFS})
target_compile_options(sccl PRIVATE ${RCCL_COMPILE_OPTS})
target_link_directories(sccl PRIVATE ${RCCL_LINK_DIRS})
target_link_libraries(sccl PRIVATE ${RCCL_LINK_LIBS})
target_link_options(sccl PRIVATE ${RCCL_LINK_OPTS})

# Track linking time
set_property(TARGET sccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

# Setup librccl.so version
#rocm_set_soversion(rccl "1.0")

# Install settings
#==================================================================================================
## Specify install targets
#rocm_install_targets(TARGETS rccl)
#rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/rccl/rccl.h src/include/nccl_net.h
#             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
#file(COPY tools/msccl-algorithms DESTINATION ${PROJECT_BINARY_DIR})
#file(COPY tools/msccl-unit-test-algorithms DESTINATION ${PROJECT_BINARY_DIR})
### Install Algorithm files under share folder
#install(DIRECTORY ${PROJECT_BINARY_DIR}/msccl-algorithms DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
#install(DIRECTORY ${PROJECT_BINARY_DIR}/msccl-unit-test-algorithms DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
#
#rocm_export_targets(
#  NAMESPACE roc::
#  TARGETS   rccl
#  DEPENDS   hip)
#
### Build with backwards compatibility if requested
#if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
#  ### Create wrapper files
#  rocm_wrap_header_dir(
#    "${PROJECT_BINARY_DIR}/include/rccl"
#    PATTERNS "rccl.h"
#    GUARDS SYMLINK WRAPPER
#    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} rccl/${CMAKE_INSTALL_INCLUDEDIR})
#
#  ### install the wrapper header file to package
#  rocm_install(
#    FILES ${PROJECT_BINARY_DIR}/rccl/include/rccl.h src/include/nccl_net.h
#    DESTINATION "./rccl/${CMAKE_INSTALL_INCLUDEDIR}/" )
#  rocm_install(
#    FILES ${PROJECT_BINARY_DIR}/include/rccl.h src/include/nccl_net.h
#    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/" )
#endif()
#
### Set package dependencies
#rocm_package_add_dependencies(DEPENDS "hip-rocclr >= 3.5.0" "rocm-smi-lib >= 4.0.0")
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
#set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/opt" "${ROCM_PATH}")
#
#find_file (DEBIAN debian_version debconf.conf PATHS /etc)
#if(DEBIAN)
#  # Write copyright file
#  file(WRITE "${CMAKE_BINARY_DIR}/copyright"
#  "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
#Upstream-Name: rccl
#Source: https://github.com/ROCmSoftwarePlatform/rccl
#
#Files: *
#Copyright: (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
#Modifications Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
#Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#License: See LICENSE.txt for license information\n")
#  install(FILES "${CMAKE_BINARY_DIR}/copyright" DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
#  # Write changelog file
#  find_program( date_executable date )
#  execute_process(COMMAND ${date_executable} -R OUTPUT_VARIABLE TIMESTAMP)
#  file(WRITE "${CMAKE_BINARY_DIR}/changelog"
#  "rccl (${VERSION_STRING}-1) unstable; urgency=medium
#
#  * Initial release.
#
# -- RCCL Maintainer <rccl-maintainer@amd.com>  ${TIMESTAMP}\n")
#  find_program( gzip_executable gzip )
#  execute_process(COMMAND bash "-c" "${gzip_executable} -9 -c ${CMAKE_BINARY_DIR}/changelog"
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR} OUTPUT_FILE "${CMAKE_BINARY_DIR}/changelog.Debian.gz")
#  install(FILES "${CMAKE_BINARY_DIR}/changelog.Debian.gz" DESTINATION ${CMAKE_INSTALL_DATADIR}/rccl)
#  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "ROCm Communication Collectives Library
#  Optimized primitives for collective multi-GPU communication")
#endif()
#
#if(BUILD_TESTS)
#  rocm_package_setup_component(clients)
#  rocm_package_setup_client_component(tests)
#  add_subdirectory(test)
#endif()
#
#rocm_create_package(
#  NAME        rccl
#  DESCRIPTION "ROCm Communication Collectives Library"
#  MAINTAINER  "RCCL Maintainer <rccl-maintainer@amd.com>"
#  LDCONFIG)