#!/bin/bash
# Usage: MODEL=llama3.2-1b DS_STAGE=3 BSZ_PER_GPU=1 ./run_all.sh 

HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MPIRUN=${MPIRUN:-mpirun}

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

MASTER_ADDR=${MASTER_ADDR:-apa00}
MASTER_PORT=${MASTER_PORT:-22555}
: ${MODEL:=llama3.2-1b} ${DS_STAGE:=3} ${BSZ_PER_GPU:=1}

$MPIRUN --allow-run-as-root \
    -mca pml ucx -mca btl ^openib \
    -H ${HOSTS:-apa00:4,apa01:4,v05:4,v06:4} \
    -x LD_PRELOAD=$HETCCL_HOME/build/libsccl.so \
    -x LD_LIBRARY_PATH=$HETCCL_HOME/build:$HETCCL_HOME/build/deps/tacc:$LD_LIBRARY_PATH \
    -x NCCL_ALGO=RING \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x MODEL=${MODEL} \
    -x DS_STAGE=${DS_STAGE} \
    -x BSZ_PER_GPU=${BSZ_PER_GPU} \
    ./driver_all.sh 2>&1 | tee -a logs/hetero_model_${MODEL}_stage${DS_STAGE}_bsz${BSZ_PER_GPU}.log
