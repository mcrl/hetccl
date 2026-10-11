#!/bin/bash

# Override via environment, e.g. MPIRUN=/opt/ompi/bin/mpirun bash scripts/run_rccl_p2p_gdrdma.sh
HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BEGIN_SIZE=1K
END_SIZE=1G
TEST_BIN=${RCCL_TESTS_HOME:?set RCCL_TESTS_HOME}/build/sendrecv_perf
LIB_PATH=${RCCL_HOME:?set RCCL_HOME}/build/librccl.so
MPIRUN=${MPIRUN:-mpirun}

GPU_SELECT_SCRIPT=$HETCCL_HOME/scripts/select_gpu.sh

$MPIRUN --allow-run-as-root \
  -mca opal_warn_on_missing_libcuda 0 -mca pml ucx -mca btl ^openib \
  -H apa00:1,apa01:1 \
  -x NCCL_ALGO=RING \
  -x NCCL_NET_GDR_LEVEL=SYS \
  -x LD_PRELOAD=$LIB_PATH \
  $GPU_SELECT_SCRIPT $TEST_BIN -b $BEGIN_SIZE -e $END_SIZE -f 2 -w 10 -n 100
