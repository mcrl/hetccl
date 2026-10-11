#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_hetccl_coll.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Base configuration
BEGIN_SIZE=1GB
END_SIZE=1GB
HETCCL_TEST_PATH=$HETCCL_HOME/tools/hetccl-tests/build
HETCCL_LIB_PATH=$HETCCL_HOME/build/libsccl.so
LD_PATH_EXTRA=$HETCCL_HOME/build:$HETCCL_HOME/build/deps/tacc:$HETCCL_HOME/tools/hetccl-tests/build
MPIRUN=${MPIRUN:-mpirun}

PRIMITIVES=(
  all_reduce_perf
  all_gather_perf
  reduce_scatter_perf
  reduce_perf
  broadcast_perf
  alltoall_perf
)

GPU_COUNTS=(2 4 6 8)

APA_NODES=(apa00 apa01)
V_NODES=(v05 v06)

# Fixed configs for 12 and 16 GPUs
declare -A HOST_CONFIGS
HOST_CONFIGS[12_1]="apa00:4,apa01:4,v05:4"
HOST_CONFIGS[12_2]="apa00:4,v05:4,v06:4"
HOST_CONFIGS[16]="apa00:4,apa01:4,v05:4,v06:4"

for primitive in "${PRIMITIVES[@]}"; do
  for gpu_count in "${GPU_COUNTS[@]}"; do
    for apa in "${APA_NODES[@]}"; do
      for v in "${V_NODES[@]}"; do
        # Divide GPUs equally per node (max 4 GPUs)
        gpus_per_node=$((gpu_count / 2))
        if [ "$gpus_per_node" -le 4 ]; then
          hostlist="${apa}:${gpus_per_node},${v}:${gpus_per_node}"
          
          echo ">>> Running primitive: $primitive"
          echo "  - GPU count: $gpu_count GPUs across: $hostlist"

          $MPIRUN --allow-run-as-root \
            -mca opal_warn_on_missing_libcuda 0 \
            -mca pml ucx -mca btl ^openib \
            -H "$hostlist" \
            -x LD_PRELOAD=$HETCCL_LIB_PATH \
            -x LD_LIBRARY_PATH=$LD_PATH_EXTRA:$LD_LIBRARY_PATH \
            $HETCCL_TEST_PATH/$primitive -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
        fi
      done
    done
  done

  # For 12 GPUs, run both configs
  for key in 12_1 12_2; do
    hostlist=${HOST_CONFIGS[$key]}

    echo ">>> Running primitive: $primitive"
    echo "  - GPU count: 12 across: $hostlist"
    
    $MPIRUN --allow-run-as-root \
      -mca opal_warn_on_missing_libcuda 0 \
      -mca pml ucx -mca btl ^openib \
      -H "$hostlist" \
      -x LD_PRELOAD=$HETCCL_LIB_PATH \
      -x LD_LIBRARY_PATH=$LD_PATH_EXTRA:$LD_LIBRARY_PATH \
      $HETCCL_TEST_PATH/$primitive -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
  done

  # For 16 GPUs
  hostlist=${HOST_CONFIGS[16]}

  echo ">>> Running primitive: $primitive"
  echo "  - GPU count: 16 GPUs across: $hostlist"

  $MPIRUN --allow-run-as-root \
    -mca opal_warn_on_missing_libcuda 0 \
    -mca pml ucx -mca btl ^openib \
    -H "$hostlist" \
    -x LD_PRELOAD=$HETCCL_LIB_PATH \
    -x LD_LIBRARY_PATH=$LD_PATH_EXTRA:$LD_LIBRARY_PATH \
    $HETCCL_TEST_PATH/$primitive -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
done
