#!/bin/bash
# Usage: DS_STAGE=3 BSZ_PER_GPU=1 ./run_convergence_amd.sh 

HETCCL_HOME=${HETCCL_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MPIRUN=${MPIRUN:-mpirun}

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

MASTER_ADDR=${MASTER_ADDR:-apa00}
MASTER_PORT=${MASTER_PORT:-22555}

$MPIRUN --allow-run-as-root \
    -mca pml ucx -mca btl ^openib \
    -H ${HOSTS:-apa00:4,apa01:4} \
    -x NCCL_ALGO=RING \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x DS_STAGE=${DS_STAGE} \
    -x BSZ_PER_GPU=${BSZ_PER_GPU} \
    ./driver_convergence_amd.sh 2>&1 | tee -a logs/amd_llama3-1b_v2_convergence_stage${DS_STAGE}_bsz${BSZ_PER_GPU}.log
