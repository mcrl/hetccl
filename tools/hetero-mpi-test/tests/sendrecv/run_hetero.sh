#!/bin/bash
# run_hetero.sh - Run heterogeneous GPU communication test
# Rank 0: NVIDIA GPU, Rank 1: AMD GPU

# set -e

# Load MPI and UCX libraries
export LD_LIBRARY_PATH=$HOME/lib/ucx/build_het/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib/openmpi/build_het/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lib/openmpi/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH

# Run heterogeneous test: NVIDIA node runs sendrecv_cuda, AMD node runs sendrecv_hip
# They communicate via MPI but use different GPU backends

# Define pairs of hosts
PAIRS=(
  "apa00,v05"
  "apa00,v06"
  "apa01,v05"
  "apa01,v06"   
  "v05,v06"  
  "apa00,apa01"
)

# Function to determine GPU backend for each node
get_backend() {
    local node=$1
    if [[ $node == apa* ]]; then
        echo "hip"
    elif [[ $node == v* ]]; then
        echo "cuda"
    else
        echo "unknown"
    fi
}

# Option 3: Heterogeneous (NVIDIA + AMD)
for pair in "${PAIRS[@]}"; do
    echo ">>> Running sendrecv on $pair"
    
    # Split pair into individual nodes
    IFS=',' read -r node1 node2 <<< "$pair"
    
    # Determine backends for each node
    backend1=$(get_backend "$node1")
    backend2=$(get_backend "$node2")
    
    # Set executable names based on backend
    if [[ $backend1 == "hip" ]]; then
        exec1="./sendrecv_hip"
    else
        exec1="./sendrecv_cuda"
    fi
    
    if [[ $backend2 == "hip" ]]; then
        exec2="./sendrecv_hip"
    else
        exec2="./sendrecv_cuda"
    fi
    
    echo "  Node1: $node1 ($backend1) -> $exec1"
    echo "  Node2: $node2 ($backend2) -> $exec2"
    
    mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
        --mca opal_warn_on_missing_libcuda 0 \
        --mca mca_base_component_show_load_errors 0 \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node1:1 $exec1 : \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node2:1 $exec2
        
    echo "  Completed: $pair"
    echo
done