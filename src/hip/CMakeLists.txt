cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
project(sccl_hip CXX)

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx803
      gfx900:xnack-
      gfx906:xnack-
      gfx908:xnack-
      gfx90a:xnack-
      gfx90a:xnack+
      gfx940
      gfx941
      gfx942
      gfx1030
      gfx1100
      gfx1101
      gfx1102)

# Load CMake modules
#==================================================================================================
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(ExternalProject)
include(cmake/Dependencies.cmake) # GTest, rocm-cmake, rocm_local_targets

# Determine which GPU architectures to build for
if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets.  Falling back to default GPUs")
  endif()
endif()

# Determine which GPU architectures to build for
set(AMDGPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "Target default GPUs if AMDGPU_TARGETS is not defined.")
if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets:")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${AMDGPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()
set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU targets to compile for.")
message(STATUS "Compiling for ${GPU_TARGETS}")

## NOTE: Reload rocm-cmake in order to update GPU_TARGETS
include(cmake/Dependencies.cmake) # Reloading to use desired GPU_TARGETS instead of defaults

# Try to establish ROCM_PATH (for find_package)
#==================================================================================================
if(NOT DEFINED ROCM_PATH)
  if("${CMAKE_CXX_COMPILER}" MATCHES ".*hipcc$")
    # Guess based on provided compiler location
    get_filename_component(_real_path ${CMAKE_CXX_COMPILER} REALPATH)
    get_filename_component(_new_path "${_real_path}" DIRECTORY)
    get_filename_component(ROCM_PATH "${_new_path}/.." REALPATH)
    message(STATUS "Setting ROCM_PATH based on hipcc location to ${ROCM_PATH}")
  else()
    # Guess default location
    set(ROCM_PATH "/opt/rocm")
    message(WARNING "Unable to find ROCM_PATH: Falling back to ${ROCM_PATH}")
  endif()
else()
  message(STATUS "ROCM_PATH found: ${ROCM_PATH}")
endif()

# Set CMAKE flags
#==================================================================================================
set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "")
set(CMAKE_CXX_STANDARD   14)   # We use C++14 features, this will add compile option: -std=c++14
set(CMAKE_CXX_EXTENSIONS OFF)  # Without this line, it will add -std=gnu++14 instead, which has some issues.
list(APPEND CMAKE_PREFIX_PATH  # Add ROCM_PATH to CMake search paths (for finding HIP / HSA
            ${ROCM_PATH}
            ${ROCM_PATH}/hip
            ${ROCM_PATH}/llvm
            ${ROCM_PATH}/hcc)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check for required dependencies
#==================================================================================================
## Check for Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

## Check for HIP
find_package(hip REQUIRED)
message(STATUS "HIP compiler:     ${HIP_COMPILER}")
message(STATUS "HIP runtime:      ${HIP_RUNTIME}")
if(NOT "${HIP_COMPILER}" MATCHES "clang")
  message(FATAL_ERROR "RCCL requires clang-based compiler (hipcc)")
endif()
find_program(hipcc_executable hipcc)
message(STATUS "hipcc executable: ${hipcc_executable}")
execute_process(
  COMMAND         bash "-c" "${hipcc_executable} --version | grep 'HIP version' | awk -F\" \" '{ printf $3}' | awk -F\"-\" '{ printf $1}'"
  OUTPUT_VARIABLE hipcc_version_string)
message(STATUS "hipcc version:    ${hipcc_version_string}")

## Check for ROCm version
execute_process(
  COMMAND         bash "-c" "cat ${ROCM_PATH}/.info/version"
  OUTPUT_VARIABLE rocm_version_string
)
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" rocm_version_matches ${rocm_version_string})
if (rocm_version_matches)
    set(ROCM_MAJOR_VERSION ${CMAKE_MATCH_1})
    set(ROCM_MINOR_VERSION ${CMAKE_MATCH_2})
    set(ROCM_PATCH_VERSION ${CMAKE_MATCH_3})

    message(STATUS "ROCm version: ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}")

    # Convert the version components to int for comparison
    math(EXPR ROCM_VERSION "(10000 * ${ROCM_MAJOR_VERSION}) + (100 * ${ROCM_MINOR_VERSION}) + ${ROCM_PATCH_VERSION}")
    add_definitions("-DROCM_VERSION=${ROCM_VERSION}")
else()
    message(WARNING "Failed to extract ROCm version.")
endif()

### Check for hipEventDisableSystemFence support
check_symbol_exists("hipEventDisableSystemFence" "hip/hip_runtime_api.h" HIP_EVENT_DISABLE_FENCE)

### Check for hipDeviceMallocUncached support
check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)

### Check for indirect function call support
if(ENABLE_IFC)
  if(${hipcc_version_string} VERSION_GREATER_EQUAL "5.5.30201")
    set(IFC_ENABLED ON)
    message(STATUS "Indirect function call enabled")
  else()
    set(IFC_ENABLED OFF)
    message(WARNING "Indirect function call disabled - requires hipcc version >= 5.5.30201")
  endif()
endif()

## Check for hsa-runtime64
find_package(hsa-runtime64 REQUIRED)
get_target_property(HSA_INCLUDE_PATH hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "HSA runtime: ${HSA_INCLUDE_PATH}")

## Check for ROCM-smi
find_package(rocm_smi PATHS ${ROCM_PATH}/lib/cmake/rocm_smi)
if (rocm_smi_FOUND)
  message(STATUS "Found rocm_smi at ${ROCM_SMI_INCLUDE_DIR}")
else()
  message(STATUS "Checking old include directory structure for rocm_smi")
  set(ROCM_SMI_INCLUDE_DIR "${ROCM_PATH}/rocm_smi/include")
  set(ROCM_SMI_LIB_DIR     "${ROCM_PATH}/rocm_smi/lib")
  set(ROCM_SMI_LIBRARIES   rocm_smi64)
endif()
check_include_file_cxx("${ROCM_SMI_INCLUDE_DIR}/rocm_smi/rocm_smi64Config.h" HAVE_ROCM_SMI64CONFIG)

## Check for BFD library if custom backtrace is requested
if(BUILD_BFD)
  enable_language(C)
  check_include_files(bfd.h HAVE_BFD)
  if (HAVE_BFD)
    message(STATUS "-- Found BFD support")

    # Check for specific BFD feature support
    CHECK_SYMBOL_EXISTS(bfd_get_section_flags "bfd.h" HAVE_DECL_BFD_GET_SECTION_FLAGS)
    CHECK_SYMBOL_EXISTS(bfd_get_section_vma   "bfd.h" HAVE_DECL_BFD_GET_SECTION_VMA)
    CHECK_CXX_SOURCE_COMPILES(
      "#include <bfd.h>

       int main (int argc, char **argv){
           bfd_size_type size;
           bfd abfd;
           asection sec;
           size = bfd_section_size(&abfd, &sec);
           return (int)(size);
       }"
      HAVE_TWO_ARG_BFD_SECTION_SIZE)

    # Check for iberty support
    find_library(HAVE_IBERTY iberty PATHS /usr/lib64 /usr/lib/ PATH_SUFFIXES x86_64-linux-gnu)
    if(HAVE_IBERTY)
      message(STATUS "iberty found @ ${HAVE_IBERTY}")
    endif()

    # Check for demangle support
    find_path(DEMANGLE_DIR demangle.h PATHS /usr/include PATH_SUFFIXES libiberty)
    if(NOT DEMANGLE_DIR)
      message(WARNING "Could not find demangle.h ${DEMANGLE_DIR}")
    else()
      message(STATUS "Found demangle.h in ${DEMANGLE_DIR}")
    endif()
  else()
    message(WARNING "bfd.h header not found - Disabling custom backtrace")
  endif()
endif()

# Check for --amdgpu-kernarg-preload-count
check_cxx_compiler_flag("-mllvm --amdgpu-kernarg-preload-count=16" HAVE_KERNARG_PRELOAD)
if (HAVE_KERNARG_PRELOAD)
  message(STATUS "Kernarg preloading to SGPR enabled")
endif()

# Helper functions

set(DATATYPES_INT
"int8_t"
"uint8_t"
"int32_t"
"uint32_t"
"int64_t"
"uint64_t"
  )
set(DATATYPES_FLOAT
  "half"
  "float"
  "double"
  "rccl_bfloat16"
  )

function(expand_collectives FILE FUNC)
  set(REDOP Sum Prod Min Max PreMulSum SumPostDiv)
  if (FUNC STREQUAL "MscclKernel")
    set(REDOP_FILTERED Sum Prod Min Max PreMulSum SumPostDiv)
  else()
    set(REDOP_FILTERED ${REDOP})
  endif()
  set(DATA_TYPES ${DATATYPES_INT} ${DATATYPES_FLOAT})
  if (BUILD_ALLREDUCE_ONLY)
    set(REDOP_FILTERED Sum)
    set(DATA_TYPES float)
  endif()
  foreach(REDOP_CURRENT IN LISTS REDOP_FILTERED)
    foreach(DATA_TYPE IN LISTS DATA_TYPES)
      if (REDOP_CURRENT STREQUAL "SumPostDiv" AND DATA_TYPE IN_LIST DATATYPES_FLOAT)
        continue()  # Skip the iteration for DATATYPES_FLOAT when REDOP_CURRENT is SumPostDiv
      endif()
      set(SRC_FILE "${CMAKE_SOURCE_DIR}/collectives/device/${FILE}.cpp.in")
      set(DST_FILE "${GENSRC_DIR}/collectives/device/${FILE}_${REDOP_CURRENT}_${DATA_TYPE}.cpp")
      message(STATUS "Generating ${DST_FILE}")
      configure_file(${SRC_FILE} ${DST_FILE})
      list(APPEND SRC_DEV_FILES ${DST_FILE})
    endforeach()
  endforeach()
  set(SRC_DEV_FILES ${SRC_DEV_FILES} PARENT_SCOPE)
endfunction()

# With BUILD_ALLREDUCE_ONLY, we only build (AllReduce, Sum, float) kernel with all algos(Ring, Tree, ...) and protocols(LL, Simple, ...)

## Add kernel files
## E.g: find src -type f \( -name "*.u" \) | sort
if (BUILD_ALLREDUCE_ONLY)
  add_definitions(-DBUILD_ALLREDUCE_ONLY)
  set(DEV_SOURCES
      collectives/device/sendrecv.cpp
      collectives/device/functions.cpp
      )
else()
  set(DEV_SOURCES
      collectives/device/all_gather.cpp
      collectives/device/alltoall_pivot.cpp
      collectives/device/broadcast.cpp
      collectives/device/functions.cpp
      collectives/device/onerank_reduce.cpp
      collectives/device/sendrecv.cpp)
endif()
list(APPEND SRC_DEV_FILES ${DEV_SOURCES})
list(APPEND SRC_DEV_FILES ${CMAKE_SOURCE_DIR}/kernels.cc)
#list(APPEND SRC_DEV_FILES ${CMAKE_SOURCE_DIR}/kernels_msccl.cc)

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
set(GENSRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/gensrc")

# expand_collectives adds the generated files to SRC_DEV_FILES
if (BUILD_ALLREDUCE_ONLY)
  expand_collectives("all_reduce" "AllReduce")
else()
  expand_collectives("all_reduce" "AllReduce")
  expand_collectives("reduce" "Reduce")
  expand_collectives("reduce_scatter" "ReduceScatter")
  if(ENABLE_MSCCL_KERNEL)
    expand_collectives("msccl_kernel" "MscclKernel")
  endif()
endif()

add_library(sccl_hip SHARED ${SRC_DEV_FILES})

## Set RCCL include directories
list(APPEND RCCL_INCLUDE_DIRS ${SCCL_SOURCE_DIR}/src/include)
list(APPEND RCCL_INCLUDE_DIRS ${SCCL_BINARY_DIR}/include)        # for generated rccl.h header
list(APPEND RCCL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/collectives/device)

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
if(${hipcc_version_string} VERSION_GREATER_EQUAL "5.7.31920")
  list(APPEND RCCL_COMPILE_DEFS HIP_UNCACHED_MEMORY)
  message(STATUS "HIP_UNCACHED_MEMORY enabled")
else()
  # keep --hipcc-func-supp on older HIP and compiler
  if(NOT IFC_ENABLED)
    list(APPEND RCCL_COMPILE_OPTS --hipcc-func-supp)
  endif()
endif()
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
if (IFC_ENABLED)
  list(APPEND RCCL_COMPILE_DEFS USE_INDIRECT_FUNCTION_CALL)
endif()
if(DEMANGLE_DIR)
  list(APPEND RCCL_COMPILE_DEFS "HAVE_CPLUS_DEMANGLE=1")
  list(APPEND RCCL_COMPILE_DEFS "HAVE_DECL_BASENAME=1")
endif()

## Set RCCL compile options
list(APPEND RCCL_COMPILE_OPTS -parallel-jobs=12)
list(APPEND RCCL_COMPILE_OPTS -Wno-format-nonliteral)
list(APPEND RCCL_COMPILE_OPTS -fgpu-rdc)               # Generate relocatable device code (required for extern __shared__)
list(APPEND RCCL_COMPILE_OPTS -fvisibility=hidden)     # Set symbol visibility to hidden by default
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
list(APPEND RCCL_LINK_LIBS -fgpu-rdc)               # Required when linking relocatable device code

## Set RCCL link options
list(APPEND RCCL_LINK_OPTS -parallel-jobs=16)       # Use multiple threads to link
if(BUILD_ADDRESS_SANITIZER)
  list(APPEND RCCL_LINK_OPTS -fuse-ld=lld)
endif()
if(TIMETRACE)
  list(APPEND RCCL_LINK_OPTS -ftime-trace)
endif()

if (HAVE_KERNARG_PRELOAD)
  list(APPEND RCCL_LINK_OPTS -mllvm --amdgpu-kernarg-preload-count=16)
endif()

list(APPEND RCCL_LINK_LIBS   Threads::Threads)
target_link_libraries(sccl_hip INTERFACE hip::host)
list(APPEND RCCL_LINK_LIBS   hip::device)
list(APPEND RCCL_LINK_LIBS   dl)
list(APPEND RCCL_LINK_LIBS   ${ROCM_SMI_LIBRARIES})

# CUDA
#list(APPEND RCCL_LINK_DIRS /usr/local/cuda/lib64)
#list(APPEND RCCL_LINK_LIBS cudart)

# Debug
#list(APPEND RCCL_COMPILE_OPTS -g)

# Apply settings to sccl_hip
target_include_directories(sccl_hip PRIVATE ${RCCL_INCLUDE_DIRS})
target_include_directories(sccl_hip PRIVATE ${SCCL_SOURCE_DIR}/deps/tacc/include)
target_compile_definitions(sccl_hip PRIVATE ${RCCL_COMPILE_DEFS})
target_compile_options(sccl_hip PRIVATE ${RCCL_COMPILE_OPTS})
target_link_directories(sccl_hip PRIVATE ${RCCL_LINK_DIRS})
target_link_libraries(sccl_hip PRIVATE ${RCCL_LINK_LIBS})
target_link_options(sccl_hip PRIVATE ${RCCL_LINK_OPTS})

install(TARGETS sccl_hip DESTINATION ${SCCL_BINARY_DIR})
