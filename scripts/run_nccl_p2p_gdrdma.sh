#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_nccl_p2p_gdrdma.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BEGIN_SIZE=1K
END_SIZE=1G
TEST_BIN=${NCCL_TESTS_HOME:?set NCCL_TESTS_HOME}/build/sendrecv_perf
LIB_PATH=${NCCL_HOME:?set NCCL_HOME}/build/lib/libnccl.so
MPIRUN=${MPIRUN:-mpirun}

GPU_SELECT_SCRIPT=$HETCCL_HOME/scripts/select_gpu.sh

$MPIRUN --allow-run-as-root \
  -mca pml ucx -mca btl ^openib \
  -H v05:1,v06:1 \
  -x NCCL_ALGO=RING \
  -x NCCL_NET_GDR_LEVEL=SYS \
  -x LD_PRELOAD=$LIB_PATH \
  $GPU_SELECT_SCRIPT $TEST_BIN -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
  
