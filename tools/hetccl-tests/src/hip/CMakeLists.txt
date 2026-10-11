cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
project(sccl_tests_hip CXX)

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

#==================================================================================================
find_package(hip REQUIRED)
add_library(sccl_tests_hip SHARED verifiable_hip.cpp)
target_include_directories(sccl_tests_hip PRIVATE ${GLOBAL_SOURCE_DIR}/src)
target_include_directories(sccl_tests_hip PRIVATE ${SCCL_HOME}/src/tacc)
target_include_directories(sccl_tests_hip PRIVATE ${SCCL_HOME}/build/include)
target_include_directories(sccl_tests_hip PRIVATE ${SCCL_HOME}/deps/tacc/include)
target_link_libraries(sccl_tests_hip PRIVATE hip::device) # this actually adds --offload-arch based on AMDGPU_TARGETS
install(TARGETS sccl_tests_hip DESTINATION ${GLOBAL_BINARY_DIR})