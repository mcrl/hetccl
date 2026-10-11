#!/bin/bash

#SBATCH -p xccl -w em12 --gres=gpu:8 --exclusive --output=sbatch.log
#SBATCH hetjob 
#SBATCH -p xccl -w v01 --gres=gpu:4 --exclusive --output=sbatch.log

export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_hwloc_base_binding_policy=none

export LD_PRELOAD=$HOME/hetccl-ws/hetccl/build/libhetccl.so
export LD_LIBRARY_PATH=$HOME/hetccl-ws/hetccl/build:$HOME/hetccl-ws/hetccl/tools/hetccl-tests/build:$LD_LIBRARY_PATH

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

srun --ntasks-per-node 8 --mpi=pmix ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 1 -n 3 \
  : --ntasks-per-node 4 --mpi=pmix ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 1 -n 3
