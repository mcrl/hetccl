#!/bin/bash

# [Compilation]
# mkdir build && cd build
# cmake -DAMDGPU_TARGETS=gfx908:xnack-  .. |& tee ../cmake.log
# make -j |& tee ../make.log
# ../test_script.sh |& tee ../test.log

# [Execution with MPI]
# mpirun -mca btl ^openib -mca pml ucx --bind-to none -H em02:1,v00:1 \
#   -x LD_PRELOAD=$HOME/hetccl-ws/hetccl/build/libhetccl.so \
#   -x LD_LIBRARY_PATH=$HOME/hetccl-ws/hetccl/build:$HOME/hetccl-ws/hetccl/tools/hetccl-tests/build:$LD_LIBRARY_PATH \
#   -x NCCL_DEBUG=TRACE \
#   -x NCCL_DEBUG_SUBSYS=ALL \
#   ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 1 -n 3

# [Execution with SLURM]
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export OMPI_MCA_hwloc_base_binding_policy=none

export LD_PRELOAD=$HOME/hetccl-ws/hetccl/build/libhetccl.so
export LD_LIBRARY_PATH=$HOME/hetccl-ws/hetccl/build:$HOME/hetccl-ws/hetccl/tools/hetccl-tests/build:$LD_LIBRARY_PATH

# VERSION - Prints the NCCL version at the start of the program.
# WARN - Prints an explicit error message whenever any NCCL call errors out.
# INFO - Prints debug information
# TRACE - Prints replayable trace information on every call.
#export NCCL_DEBUG=TRACE
# Supported subsystem names are INIT (stands for initialization), COLL (stands for collectives), P2P (stands for peer-to-peer), SHM (stands for shared memory), NET (stands for network), GRAPH (stands for topology detection and graph search), TUNING (stands for algorithm/protocol tuning), ENV (stands for environment settings), ALLOC (stands for memory allocations), and ALL (includes every subsystem).
export NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC

salloc \
    -p xccl -w em[11-14] --gres=gpu:8 --exclusive \
  : -p xccl -w v[01-02],v[05-06] --gres=gpu:4 --exclusive \
srun \
    --ntasks-per-node 8 --mpi=pmix ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 1 -n 3 \
  : --ntasks-per-node 4 --mpi=pmix ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -w 1 -n 3

# Available tests:
# all_gather_perf
# alltoall_perf
# broadcast_perf
# reduce_perf
# scatter_perf
# all_reduce_perf
# alltoallv_perf
# gather_perf
# reduce_scatter_perf
# sendrecv_perf

# Available options:
# [-t,--nthreads <num threads>] 
# [-g,--ngpus <gpus per thread>] 
# [-b,--minbytes <min size in bytes>] 
# [-e,--maxbytes <max size in bytes>] 
# [-i,--stepbytes <increment size>] 
# [-f,--stepfactor <increment factor>] 
# [-n,--iters <iteration count>] 
# [-m,--agg_iters <aggregated iteration count>] 
# [-w,--warmup_iters <warmup iteration count>] 
# [-p,--parallel_init <0/1>] 
# [-c,--check <check iteration count>] 
# [-o,--op <sum/prod/min/max/avg/mulsum/all>] 
# [-d,--datatype <nccltype/all>] 
# [-r,--root <root>] 
# [-z,--blocking <0/1>] 
# [-Y,--memory_type <coarse/fine/host/managed>] 
# [-s,--stress_cycles <number of cycles>] 
# [-u,--cumask <d0,d1,d2,d3>] 
# [-y,--stream_null <0/1>] 
# [-T,--timeout <time in seconds>] 
# [-G,--cudagraph <num graph launches>] 
# [-C,--report_cputime <0/1>] 
# [-O,--out_of_place <0/1>] 
# [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
# [-q,--delay <delay between out-of-place and in-place in microseconds>] 
# [-h,--help]
