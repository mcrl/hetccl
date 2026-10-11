#!bin/bash

make MPI=1 \
  MPI_HOME=$HOME/openmpi/build \
  ROCM_PATH=/opt/rocm \
  CUSTOM_RCCL_LIB=$HOME/hetccl-ws/hetccl/build/librccl.so