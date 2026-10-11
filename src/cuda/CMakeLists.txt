cmake_minimum_required(VERSION 3.5)

project(hetccl_cuda CXX)

function(expand_collectives_cuda)
  message(STATUS "expanding collectives_cuda")
  if (BUILD_ALLREDUCE_ONLY)
    set(BASE_LIST all_reduce)
    set(OP_LIST sum)
    set(opn_start 0) # sum index
    set(DT_LIST f32)
    set(dtn_start 7) # f32 index
  else()
    set(BASE_LIST sendrecv all_reduce all_gather broadcast reduce reduce_scatter)
    set(OP_LIST sum prod min max premulsum sumpostdiv)
    set(opn_start 0)
    set(DT_LIST i8 u8 i32 u32 i64 u64 f16 f32 f64 bf16)
    set(dtn_start 0)
  endif()
  foreach(BASE IN LISTS BASE_LIST)
    message(STATUS "expanding collectives_cuda ${BASE}")
    set(opn ${opn_start})
    foreach(OP IN LISTS OP_LIST)
      set(dtn ${dtn_start})
      foreach(DT IN LISTS DT_LIST)
        set(SRC_FILE "${CMAKE_SOURCE_DIR}/collectives/device/${BASE}.cu")
        set(DST_FILE "${GENSRC_DIR}/collectives/device/${BASE}_${OP}_${DT}.cu")
        message(STATUS "Generating ${DST_FILE}")
        ADD_CUSTOM_COMMAND(
          OUTPUT ${DST_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE} ${DST_FILE}
          DEPENDS ${SRC_FILE}
          )
        set_property(SOURCE ${DST_FILE} APPEND PROPERTY COMPILE_DEFINITIONS "NCCL_OP=${opn}")
        set_property(SOURCE ${DST_FILE} APPEND PROPERTY COMPILE_DEFINITIONS "NCCL_TYPE=${dtn}")
        list(APPEND SRC_CUDA_DEV_FILES ${DST_FILE})
        math(EXPR dtn "${dtn} + 1")
      endforeach()
      math(EXPR opn "${opn} + 1")
    endforeach()
  endforeach()
  set(SRC_CUDA_DEV_FILES ${SRC_CUDA_DEV_FILES} PARENT_SCOPE)
endfunction()

function(apply_cuda_properties)
  foreach(SRC_FILE IN LISTS SRC_CUDA_DEV_FILES)
    set_property(SOURCE ${SRC_FILE} APPEND PROPERTY LANGUAGE CUDA)
    # relocatable device code generation is necessary to compile code such as the following:
    # // common.h
    # typedef void(*ncclKern_t)();
    # extern __device__ ncclKern_t ncclFuncs[];
    set_property(SOURCE ${SRC_FILE} APPEND PROPERTY COMPILE_OPTIONS "-dc")
    set_property(SOURCE ${SRC_FILE} APPEND PROPERTY COMPILE_OPTIONS "--extended-lambda")
    # Maxrregcount needs to be set accordingly to NCCL_MAX_NTHREADS (otherwise it will cause kernel launch errors)
    # 512 : 120, 640 : 96, 768 : 80, 1024 : 60
    # We would not have to set this if we used __launch_bounds__, but this only works on kernels, not on functions.
    set_property(SOURCE ${SRC_FILE} APPEND PROPERTY COMPILE_OPTIONS "-Xptxas")
    set_property(SOURCE ${SRC_FILE} APPEND PROPERTY COMPILE_OPTIONS "-maxrregcount=96")
  endforeach()
endfunction()

if (BUILD_ALLREDUCE_ONLY)
  add_definitions(-DBUILD_ALLREDUCE_ONLY)
endif()

set(CMAKE_CUDA_COMPILER nvcc)
enable_language(CUDA)
set(GENSRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/gensrc")
expand_collectives_cuda()
list(APPEND SRC_CUDA_DEV_FILES ${CMAKE_SOURCE_DIR}/collectives/device/functions.cu)
list(APPEND SRC_CUDA_DEV_FILES ${CMAKE_SOURCE_DIR}/collectives/device/onerank_reduce.cu)
list(APPEND SRC_CUDA_DEV_FILES ${CMAKE_SOURCE_DIR}/kernels.cc)
apply_cuda_properties()
add_library(hetccl_cuda SHARED ${SRC_CUDA_DEV_FILES})
target_compile_options(hetccl_cuda PRIVATE --compiler-options=-fPIC)
# https://stackoverflow.com/questions/69061997/cuda-separable-compilation-shared-libraries-invalid-device-function-segfa
# https://cmake.org/cmake/help/v3.22/command/target_compile_options.html#option-de-duplication
#target_compile_options(hetccl_cuda PRIVATE "SHELL:-Xcompiler -fvisibility=hidden")
set_target_properties(hetccl_cuda PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hetccl_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Passing flags to device link step is not supported in cmake 3.16.3
# The following is a workaround to pass flags to device link step
# https://gitlab.kitware.com/cmake/cmake/-/issues/18265
# It is also passed to the compile step
# tesla v100 = 70, tesla a100 = 80, rtx a5000 = 86
list(APPEND CMAKE_CUDA_FLAGS "-gencode=arch=compute_70,code=compute_70")
target_include_directories(hetccl_cuda PRIVATE ${CMAKE_SOURCE_DIR}/collectives/device)
target_include_directories(hetccl_cuda PRIVATE ${HETCCL_SOURCE_DIR}/src/include)
target_include_directories(hetccl_cuda PRIVATE ${HETCCL_BINARY_DIR}/include)
target_include_directories(hetccl_cuda PRIVATE ${HETCCL_SOURCE_DIR}/deps/tacc/include)

install(TARGETS hetccl_cuda DESTINATION ${HETCCL_BINARY_DIR})
