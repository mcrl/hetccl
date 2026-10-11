#!/bin/bash
# run_hetero.sh - Run heterogeneous GPU communication test
# Rank 0: NVIDIA GPU, Rank 1: AMD GPU

# set -e

# Load MPI and UCX libraries
export LD_LIBRARY_PATH=$HOME/lib/ucx/build_het/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib/openmpi/build_het/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lib/openmpi/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH

# Run heterogeneous test: NVIDIA node runs allreduce_cuda, AMD node runs allreduce_hip
# They communicate via MPI but use different GPU backends

# Option 1: AMD nodes (apa00, apa01)
mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
    --mca mca_base_component_show_load_errors 0 \
    --mca opal_warn_on_missing_libcuda 0 \
    -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
    -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
    -H apa00:1,apa01:1 \
    ./allreduce_hip 

# Option 2: NVIDIA nodes (v05, v06)
mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
    --mca mca_base_component_show_load_errors 0 \
    -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
    -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
    -H v05:1,v06:1 \
    ./allreduce_cuda