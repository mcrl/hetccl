cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_COMPILER nvcc)
project(sccl_tests_cuda CUDA)

list(APPEND CMAKE_CUDA_FLAGS "-gencode=arch=compute_70,code=compute_70")

add_library(sccl_tests_cuda SHARED verifiable_cuda.cu)
target_include_directories(sccl_tests_cuda PRIVATE ${GLOBAL_SOURCE_DIR}/src)
target_include_directories(sccl_tests_cuda PRIVATE ${SCCL_HOME}/src/tacc)
target_include_directories(sccl_tests_cuda PRIVATE ${SCCL_HOME}/build/include)
target_include_directories(sccl_tests_cuda PRIVATE ${SCCL_HOME}/deps/tacc/include)

install(TARGETS sccl_tests_cuda DESTINATION ${GLOBAL_BINARY_DIR})
