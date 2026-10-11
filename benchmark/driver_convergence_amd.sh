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
set +a

export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export PYTORCH_ROCM_ARCH="gfx1100"

# GPT2 Small Config
# python -u train_convergence.py \
#     --model gpt \
#     --lr 0.0005 \
#     --grad_accum_steps 1 \
#     --num_iter 5000 \
#     --save_loss_curve \
#     --loss_curve_dir ./loss_curves \
#     --loss_curve_filename amd_8N_gpt2-small_lr5e-4_iter5000_wiki103.txt \

    # --save_checkpoint \
    # --checkpoint_dir ./checkpoints/amd_8N_gpt2-small_wiki103/

# LLaMA 3 1B Config
python -u train_convergence.py \
    --model llama \
    --lr 0.0003 \
    --grad_accum_steps 32 \
    --num_iter 32000 \
    --bf16 \
    --save_loss_curve \
    --loss_curve_dir ./loss_curves \
    --loss_curve_filename amd_8N_llama3-1b_v2_lr3e-4_gas32_iter32000_wiki103.txt
    # --save_checkpoint \
    # --checkpoint_dir ./checkpoints/amd_8N_llama3-1b_wiki103/ 