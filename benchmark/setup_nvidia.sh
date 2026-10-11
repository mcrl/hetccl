#!/bin/bash
# Create the conda environment used by driver_*.sh on NVIDIA nodes.
# Usage: bash setup_nvidia.sh   (env name can be overridden with ENV_NAME=...)
set -e
ENV_NAME=${ENV_NAME:-hetccl-nvidia}
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh

conda create -y -n $ENV_NAME --override-channels -c conda-forge python=3.11
conda activate $ENV_NAME

pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# DeepSpeed ops are JIT-compiled on first use
DS_BUILD_OPS=0 pip install deepspeed==0.16.9 transformers==4.52.3 datasets==3.6.0
