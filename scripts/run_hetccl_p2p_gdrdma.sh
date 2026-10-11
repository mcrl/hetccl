#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_hetccl_p2p_gdrdma.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BEGIN_SIZE=1K
END_SIZE=1G
TEST_BIN=$HETCCL_HOME/tools/hetccl-tests/build/sendrecv_perf
LIB_PATH=$HETCCL_HOME/build/libhetccl.so
LD_PATH_EXTRA=$HETCCL_HOME/build:$HETCCL_HOME/build/deps/tacc:$HETCCL_HOME/tools/hetccl-tests/build
MPIRUN=${MPIRUN:-mpirun}


# Define pairs of hosts
PAIRS=(
  "apa00,apa01"
  "v05,v06"  
  "apa00,v05"
  "apa00,v06"
  "apa01,v05"
  "apa01,v06"   
)

GPU_SELECT_SCRIPT=$HETCCL_HOME/scripts/select_gpu.sh

for pair in "${PAIRS[@]}"; do
  echo ">>> Running sendrecv_perf on: $pair"
  
  $MPIRUN --allow-run-as-root \
    -x LD_PRELOAD=$LIB_PATH \
    -x LD_LIBRARY_PATH=$LD_PATH_EXTRA:$LD_LIBRARY_PATH \
    -mca opal_warn_on_missing_libcuda 0 -mca pml ucx -mca btl ^openib \
    -H ${pair//,/":1,"}:1 \
    -x NCCL_ALGO=RING \
    -x NCCL_NET_GDR_LEVEL=SYS \
    $GPU_SELECT_SCRIPT $TEST_BIN -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
done