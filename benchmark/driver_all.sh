#!/bin/bash

# Activate the per-vendor conda environment (created by setup_nvidia.sh / setup_amd.sh)
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
if [[ -e /dev/kfd ]]; then  # AMD GPU node (ROCm)
  conda activate ${AMD_ENV:-hetccl-amd}
else                        # NVIDIA GPU node
  conda activate ${NVIDIA_ENV:-hetccl-nvidia}
fi

set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=22555}
: ${OMP_NUM_THREADS=1}
: ${DISTRIBUTED_BACKEND=nccl}
: ${MODEL=llama3.2-1b}
: ${DS_STAGE=3}
: ${BSZ_PER_GPU=1}
set +a

export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK

python -u train.py \
    --model ${MODEL} 