cmake_minimum_required(VERSION 3.5)

project(hetccl-tests CXX)

if(NOT HETCCL_HOME)
  set(HETCCL_HOME ${PROJECT_SOURCE_DIR}/../..)
endif()
message(STATUS "HETCCL_HOME: ${HETCCL_HOME}")

find_package(MPI REQUIRED)
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
  message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
else()
  message(FATAL_ERROR "MPI not found")
endif()

include(ExternalProject)

ExternalProject_Add(hetccl_tests_hip
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/hip
  BUILD_ALWAYS true
  CMAKE_ARGS "-DGLOBAL_BINARY_DIR=${PROJECT_BINARY_DIR}"
  CMAKE_ARGS "-DGLOBAL_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
  CMAKE_ARGS "-DHETCCL_HOME=${HETCCL_HOME}"
  CMAKE_ARGS "-DAMDGPU_TARGETS=${AMDGPU_TARGETS}"
)

ExternalProject_Add(hetccl_tests_cuda
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/cuda
  BUILD_ALWAYS true
  CMAKE_ARGS "-DGLOBAL_BINARY_DIR=${PROJECT_BINARY_DIR}"
  CMAKE_ARGS "-DGLOBAL_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
  CMAKE_ARGS "-DHETCCL_HOME=${HETCCL_HOME}"
  CMAKE_ARGS "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")
add_custom_target(git_version_check
  COMMENT "Updating git_version.cpp if necessary"
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/git_version.cmake
  VERBATIM
)

set(COMMON_SRCS
  src/common.cpp
  src/timer.cc
  verifiable/verifiable.cc
  ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp
)

function(gen_exec COLL_NAME)
  add_executable(${COLL_NAME}_perf ${COMMON_SRCS} src/${COLL_NAME}.cpp)
  add_dependencies(${COLL_NAME}_perf hetccl_tests_hip hetccl_tests_cuda git_version_check)
  target_include_directories(${COLL_NAME}_perf PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(${COLL_NAME}_perf PRIVATE ${HETCCL_HOME}/deps/tacc/include)
  target_include_directories(${COLL_NAME}_perf PRIVATE ${HETCCL_HOME}/build/include)
  target_include_directories(${COLL_NAME}_perf PRIVATE /opt/rocm/include)
  target_compile_definitions(${COLL_NAME}_perf PRIVATE __HIP_PLATFORM_AMD__)
  target_compile_definitions(${COLL_NAME}_perf PRIVATE MPI_SUPPORT)
  target_link_libraries(${COLL_NAME}_perf ${HETCCL_HOME}/build/libhetccl.so)
  target_link_libraries(${COLL_NAME}_perf pthread dl ${MPI_LIBRARIES})
  target_link_libraries(${COLL_NAME}_perf ${HETCCL_HOME}/build/deps/tacc/libtacc.so)
endfunction()

#add_subdirectory(deps/tacc)

gen_exec(all_gather)
gen_exec(alltoall)
gen_exec(broadcast)
gen_exec(reduce)
gen_exec(scatter)
gen_exec(all_reduce)
gen_exec(alltoallv)
gen_exec(gather)
gen_exec(reduce_scatter)
gen_exec(sendrecv)