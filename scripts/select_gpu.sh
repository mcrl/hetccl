#!/bin/bash

# Get hostname
HOST=$(hostname)

# Select optimal GPU (same NUMA as NIC) per host
if [[ "$HOST" == "apa00" ]]; then
  # apa00: NIC on NUMA 1 -> use GPU 3
  export HIP_VISIBLE_DEVICES=3
  # echo "[Wrapper] $HOST: Selected GPU 3 (NUMA 1)"
elif [[ "$HOST" == "apa01" ]]; then
  # apa01: NIC on NUMA 0 -> use GPU 0
  export HIP_VISIBLE_DEVICES=0
  # echo "[Wrapper] $HOST: Selected GPU 0 (NUMA 0)"
elif [[ "$HOST" == "v05" ]]; then
  # v05: NIC on NUMA 0 -> use GPU 3
  export CUDA_VISIBLE_DEVICES=3
  # echo "[Wrapper] $HOST: Selected GPU 3 (NUMA 0)"
elif [[ "$HOST" == "v06" ]]; then
  # v06: NIC on NUMA 0 -> use GPU 3
  export CUDA_VISIBLE_DEVICES=3
  # echo "[Wrapper] $HOST: Selected GPU 3 (NUMA 0)"
else
  # Default fallback
  export HIP_VISIBLE_DEVICES=0
  export CUDA_VISIBLE_DEVICES=3
fi

# Execute actual command ($@ is the full command passed by mpirun)
exec "$@"