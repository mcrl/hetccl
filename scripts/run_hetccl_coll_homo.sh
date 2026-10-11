#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_hetccl_coll_homo.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Base configuration
BEGIN_SIZE=1GB
END_SIZE=1GB
SR_TEST_PATH=$HETCCL_HOME/tools/hetccl-tests/build
SR_LIB_PATH=$HETCCL_HOME/build/libsccl.so
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

run_on_pair() {
  local primitive=$1
  local gpu_count=$2
  shift 2
  local nodes=("$@")

  local gpus_per_node=$((gpu_count / 2))
  if [ "$gpus_per_node" -le 4 ]; then
    hostlist="${nodes[0]}:${gpus_per_node},${nodes[1]}:${gpus_per_node}"

    echo ">>> Running primitive: $primitive"
    echo "  - GPU count: $gpu_count GPUs across: $hostlist"

    $MPIRUN --allow-run-as-root \
      -mca opal_warn_on_missing_libcuda 0 \
      -mca pml ucx -mca btl ^openib \
      -H "$hostlist" \
      -x LD_PRELOAD=$SR_LIB_PATH \
      -x LD_LIBRARY_PATH=$LD_PATH_EXTRA:$LD_LIBRARY_PATH \
      $SR_TEST_PATH/$primitive -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
  fi
}

for primitive in "${PRIMITIVES[@]}"; do
  for gpu_count in "${GPU_COUNTS[@]}"; do
    run_on_pair "$primitive" "$gpu_count" "${APA_NODES[@]}"
    run_on_pair "$primitive" "$gpu_count" "${V_NODES[@]}"
  done
done