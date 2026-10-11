#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_nccl_coll.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BEGIN_SIZE=1G
END_SIZE=1G
NCCL_TEST_PATH=${NCCL_TESTS_HOME:?set NCCL_TESTS_HOME}/build
NCCL_LIB_PATH=${NCCL_HOME:?set NCCL_HOME}/build/lib/libnccl.so
MPIRUN=${MPIRUN:-mpirun}
HOST_BASE="v05"
HOST_PEER="v06"

PRIMITIVES=(
  all_reduce_perf
  all_gather_perf
  reduce_scatter_perf
  reduce_perf
  broadcast_perf
  alltoall_perf
)

GPU_COUNTS=(2 4 6 8)

for primitive in "${PRIMITIVES[@]}"; do
  for gpu_count in "${GPU_COUNTS[@]}"; do
    # Calculate how many GPUs per host
    gpus_per_host=$((gpu_count / 2))
    
    echo "Running primitive: $primitive"
    echo "  - GPU count: $gpu_count (per host: $gpus_per_host)"
    
    $MPIRUN --allow-run-as-root \
      -mca pml ucx -mca btl ^openib \
      -H ${HOST_BASE}:${gpus_per_host},${HOST_PEER}:${gpus_per_host} \
      -x NCCL_ALGO=RING \
      -x LD_PRELOAD=$NCCL_LIB_PATH \
      $NCCL_TEST_PATH/$primitive -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
  done
done