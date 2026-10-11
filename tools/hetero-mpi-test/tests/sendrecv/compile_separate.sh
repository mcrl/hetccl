#!/bin/bash
# compile_separate.sh - Build separate binaries for CUDA and HIP

set -e

COMMON_DIR="../../common"

echo "Building separate binaries for heterogeneous communication..."

# CUDA binary for NVIDIA nodes
echo "  [1/2] Building CUDA binary..."
mpicxx -DUSE_CUDA -o sendrecv_cuda main.cpp ${COMMON_DIR}/cuda_backend.cpp \
    -I${COMMON_DIR} \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart

# HIP binary for AMD nodes  
echo "  [2/2] Building HIP binary..."
mpicxx -DUSE_HIP -D__HIP_PLATFORM_AMD__ -o sendrecv_hip main.cpp ${COMMON_DIR}/hip_backend.cpp \
    -I${COMMON_DIR} \
    -I/opt/rocm/include \
    -L/opt/rocm/lib -lamdhip64

echo "Build complete:"
echo "  - sendrecv_cuda (for NVIDIA nodes)"
echo "  - sendrecv_hip  (for AMD nodes)"
