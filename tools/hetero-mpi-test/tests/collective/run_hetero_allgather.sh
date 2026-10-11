#!/bin/bash
# run_hetero.sh - Run heterogeneous GPU communication test
# Rank 0: NVIDIA GPU, Rank 1: AMD GPU

# set -e

# Load MPI and UCX libraries
export LD_LIBRARY_PATH=$HOME/lib/ucx/build_het/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib/openmpi/build_het/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lib/openmpi/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH

# Run heterogeneous test: NVIDIA node runs allgather_cuda, AMD node runs allgather_hip
# They communicate via MPI but use different GPU backends

# Define pairs of hosts (2 nodes)
PAIRS=(
  "apa00,v05"
  "apa00,v06"
  "apa01,v05"
  "apa01,v06"   
  "v05,v06"  
  "apa00,apa01"
)

# Define 3-node combinations (12 GPUs total with 4 GPUs per node)
TRIPLETS=(
  "apa00,apa01,v05"
  "apa00,v05,v06"
)

# Define 4-node combination (16 GPUs total with 4 GPUs per node)
QUARTETS=(
  "apa00,apa01,v05,v06"
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

# # Option 1: Heterogeneous (NVIDIA + AMD)
# for pair in "${PAIRS[@]}"; do
#     echo ">>> Running allgather on $pair"
    
#     # Split pair into individual nodes
#     IFS=',' read -r node1 node2 <<< "$pair"
    
#     # Determine backends for each node
#     backend1=$(get_backend "$node1")
#     backend2=$(get_backend "$node2")
    
#     # Set executable names based on backend
#     if [[ $backend1 == "hip" ]]; then
#         exec1="./allgather_hip"
#     else
#         exec1="./allgather_cuda"
#     fi
    
#     if [[ $backend2 == "hip" ]]; then
#         exec2="./allgather_hip"
#     else
#         exec2="./allgather_cuda"
#     fi
    
#     echo "  Node1: $node1 ($backend1) -> $exec1"
#     echo "  Node2: $node2 ($backend2) -> $exec2"
    
#     # 2 Nodes
#     for num_gpus_per_node in 1 2 3 4 
#     do
#         echo ">>>  Running with $num_gpus_per_node GPUs per node (total: $((num_gpus_per_node * 2)) GPUs)"
#         mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
#             --mca opal_warn_on_missing_libcuda 0 \
#             --mca mca_base_component_show_load_errors 0 \
#             -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
#             -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
#             -H $node1:$num_gpus_per_node $exec1 : \
#             -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
#             -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
#             -H $node2:$num_gpus_per_node $exec2
#     done
#     echo "  Completed: $pair"
#     echo
# done

# # Option 2: 3-node combinations (12 GPUs with 4 GPUs per node)
# for triplet in "${TRIPLETS[@]}"; do
#     echo ">>> Running allgather on $triplet (12 GPUs total)"
    
#     # Split triplet into individual nodes
#     IFS=',' read -r node1 node2 node3 <<< "$triplet"
    
#     # Determine backends and executables for each node
#     backend1=$(get_backend "$node1")
#     backend2=$(get_backend "$node2")
#     backend3=$(get_backend "$node3")
    
#     if [[ $backend1 == "hip" ]]; then
#         exec1="./allgather_hip"
#     else
#         exec1="./allgather_cuda"
#     fi
    
#     if [[ $backend2 == "hip" ]]; then
#         exec2="./allgather_hip"
#     else
#         exec2="./allgather_cuda"
#     fi
    
#     if [[ $backend3 == "hip" ]]; then
#         exec3="./allgather_hip"
#     else
#         exec3="./allgather_cuda"
#     fi
    
#     echo "  Node1: $node1 ($backend1) -> $exec1"
#     echo "  Node2: $node2 ($backend2) -> $exec2"
#     echo "  Node3: $node3 ($backend3) -> $exec3"
    
#     echo ">>>  Running with 4 GPUs per node (total: 12 GPUs)"
#     mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
#         --mca opal_warn_on_missing_libcuda 0 \
#         --mca mca_base_component_show_load_errors 0 \
#         -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
#         -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
#         -H $node1:4 $exec1 : \
#         -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
#         -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
#         -H $node2:4 $exec2 : \
#         -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
#         -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
#         -H $node3:4 $exec3
    
#     echo "  Completed: $triplet"
#     echo
# done

# Option 3: 4-node combination (16 GPUs with 4 GPUs per node)
for quartet in "${QUARTETS[@]}"; do
    echo ">>> Running allgather on $quartet (16 GPUs total)"
    
    # Split quartet into individual nodes
    IFS=',' read -r node1 node2 node3 node4 <<< "$quartet"
    
    # Determine backends and executables for each node
    backend1=$(get_backend "$node1")
    backend2=$(get_backend "$node2")
    backend3=$(get_backend "$node3")
    backend4=$(get_backend "$node4")
    
    if [[ $backend1 == "hip" ]]; then
        exec1="./allgather_hip"
    else
        exec1="./allgather_cuda"
    fi
    
    if [[ $backend2 == "hip" ]]; then
        exec2="./allgather_hip"
    else
        exec2="./allgather_cuda"
    fi
    
    if [[ $backend3 == "hip" ]]; then
        exec3="./allgather_hip"
    else
        exec3="./allgather_cuda"
    fi
    
    if [[ $backend4 == "hip" ]]; then
        exec4="./allgather_hip"
    else
        exec4="./allgather_cuda"
    fi
    
    echo "  Node1: $node1 ($backend1) -> $exec1"
    echo "  Node2: $node2 ($backend2) -> $exec2"
    echo "  Node3: $node3 ($backend3) -> $exec3"
    echo "  Node4: $node4 ($backend4) -> $exec4"
    
    echo ">>>  Running with 4 GPUs per node (total: 16 GPUs)"
    mpirun -mca pml ucx -mca btl ^openib -mca coll ^hcoll \
        --mca opal_warn_on_missing_libcuda 0 \
        --mca mca_base_component_show_load_errors 0 \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node1:4 $exec1 : \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node2:4 $exec2 : \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node3:4 $exec3 : \
        -x LD_LIBRARY_PATH=$HOME/lib/missing_libs:$LD_LIBRARY_PATH \
        -x PATH=$HOME/lib/openmpi/build/bin:$PATH \
        -H $node4:4 $exec4
    
    echo "  Completed: $quartet"
    echo
done