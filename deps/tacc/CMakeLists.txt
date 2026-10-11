cmake_minimum_required(VERSION 3.5)

project(tacc CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRC_FILES
  src/tacc.cc
  src/backend_cuda.cc
  src/backend_hip.cc
)

add_library(tacc SHARED ${SRC_FILES})
target_include_directories(tacc PUBLIC include)
if (CUDA_HOME)
  message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")
  target_include_directories(tacc PRIVATE ${CUDA_HOME}/include)
else()
  message(FATAL_ERROR "CUDA_HOME not set. Pass it via -DCUDA_HOME=...")
endif()
target_include_directories(tacc PRIVATE /opt/rocm/include)
target_compile_definitions(tacc PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(tacc PRIVATE dl)

add_subdirectory(tests)
