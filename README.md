# HetCCL

HetCCL is a collective communication library for training deep learning models on clusters with **both NVIDIA and AMD GPUs**.
It uses the vendor libraries ([NCCL](https://github.com/NVIDIA/nccl) and [RCCL](https://github.com/ROCm/rccl)) as backends and adds a cross-vendor runtime abstraction (TACC) plus RDMA-based inter-node communication between GPUs of different vendors.
Because HetCCL keeps the NCCL API, existing training code (e.g., PyTorch + DeepSpeed) runs on a mixed NVIDIA/AMD cluster without modification: preload `libhetccl.so` and launch as usual.

This repository is the official implementation of the NeurIPS 2026 paper:

> **HetCCL: Efficient LLM Training on Heterogeneous Vendor GPUs**

## Table of Contents

- [Repository Structure](#repository-structure)
- [Tested System](#tested-system)
- [Setup](#setup)
  - [Dependencies](#dependencies)
  - [Environment Variables](#environment-variables)
  - [Build HetCCL](#build-hetccl)
  - [Build Baselines (NCCL / RCCL)](#build-baselines-nccl--rccl)
- [Evaluation](#evaluation)
  - [Point-to-Point & Collective Communication](#point-to-point--collective-communication)
  - [Training LLMs](#training-llms)
- [Evaluation Results](#evaluation-results)
- [License](#license)
- [Citation](#citation)

## Repository Structure

```
hetccl/
├── src/                    # HetCCL implementation (built on top of NCCL/RCCL)
├── deps/tacc/              # TACC: runtime API abstraction layer across CUDA and HIP
├── tools/
│   ├── hetccl-tests/       # nccl-tests compatible benchmarks for heterogeneous GPUs
│   └── hetero-mpi-test/    # GPU-aware MPI baseline (cross-vendor UCX + OpenMPI)
├── benchmark/              # LLM training benchmarks (PyTorch + DeepSpeed)
├── scripts/                # Scripts to reproduce the communication experiments
├── cmake/, makefiles/      # Build helpers
├── LICENSE.txt
└── NOTICES.txt
```

## Tested System

We evaluate on a multi-vendor GPU cluster with two NVIDIA GPU nodes and two AMD GPU nodes, all connected by InfiniBand.
The host names below (`v05`, `v06`, `apa00`, `apa01`) appear as defaults in the scripts; replace them with your own.

|                  | NVIDIA nodes (v05, v06)            | AMD nodes (apa00, apa01)           |
|:----------------:|:----------------------------------:|:----------------------------------:|
| GPU              | 4 x NVIDIA Tesla V100              | 4 x AMD Radeon PRO W7800           |
| GPU Interconnect | PCIe Gen 3                         | PCIe Gen 4                         |
| Driver (Runtime) | 550.54.15 (CUDA 12.4)              | 6.12.12 (ROCm 6.4.0)               |
| NIC              | Mellanox ConnectX-6 InfiniBand HDR | Mellanox ConnectX-6 InfiniBand HDR |

Each node contains GPUs from a single vendor; heterogeneity is across nodes.

## Setup

### Dependencies

- `CUDA` 12.x (tested: 12.2, 12.4)
- `ROCm` 6.x (tested: 6.4.0)
- `OpenMPI` 5.0.2 and `UCX` 1.19.0
- `CMake` >= 3.16

Each compute node needs only the runtime of its own vendor (CUDA or ROCm). HetCCL loads the vendor runtime at execution time.

Additional dependencies for LLM training (installed by [`benchmark/setup_nvidia.sh`](benchmark/setup_nvidia.sh) and [`benchmark/setup_amd.sh`](benchmark/setup_amd.sh)):

- `torch` 2.7.0 (NVIDIA: `cu126` wheel, AMD: `rocm6.3` wheel)
- `deepspeed` 0.16.9
- `transformers` 4.52.3
- `datasets` 3.6.0

### Environment Variables

```bash
export CUDA_HOME=/path/to/cuda          # default: /usr/local/cuda
export ROCM_PATH=/path/to/rocm          # default: /opt/rocm
export MPI_HOME=/path/to/openmpi
export UCX_HOME=/path/to/ucx

export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$UCX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib:$UCX_HOME/lib:$LD_LIBRARY_PATH"
```

### Build HetCCL

HetCCL compiles device code for both vendors, so the build needs `hipcc` (ROCm) **and** `nvcc` (CUDA).
The examples below target AMD Radeon PRO W7800 (`gfx1100`) and NVIDIA V100 (`sm_70`); change `AMDGPU_TARGETS` and `CMAKE_CUDA_ARCHITECTURES` for your GPUs.
`-DCUDA_HOME` is always required, because TACC includes the CUDA headers.

The build produces `build/libhetccl.so` (the library to preload), `build/libhetccl_cuda.so`, `build/libhetccl_hip.so`, and `build/deps/tacc/libtacc.so`.
The tests build produces `tools/hetccl-tests/build/*_perf`, `libhetccl_tests_cuda.so`, and `libhetccl_tests_hip.so`.

#### Option A: a node with both CUDA and ROCm

```bash
cd hetccl
mkdir build && cd build
cmake -DAMDGPU_TARGETS=gfx1100 -DCMAKE_CUDA_ARCHITECTURES=70 -DCUDA_HOME=$CUDA_HOME ..
make -j

cd ../tools/hetccl-tests
mkdir build && cd build
cmake -DAMDGPU_TARGETS=gfx1100 -DCMAKE_CUDA_ARCHITECTURES=70 ..   # requires mpicxx in PATH
make -j
```

#### Option B: split build across an AMD node and an NVIDIA node

If no single node has both toolkits, build each vendor part on its own node in the same build directory (shared file system).
On the AMD node, `CUDA_HOME` only needs to contain the CUDA headers (`$CUDA_HOME/include`), e.g., a copy of the CUDA `include` directory.

```bash
# 1) AMD node: configure and build the HIP part
cd hetccl && mkdir build && cd build
cmake -DAMDGPU_TARGETS=gfx1100 -DCMAKE_CUDA_ARCHITECTURES=70 -DCUDA_HOME=$CUDA_HOME ..
make -j hetccl_hip tacc git_version_check

# 2) NVIDIA node (nvcc in PATH): build the CUDA part
cd hetccl/build
make -j hetccl_cuda

# 3) AMD node: link libhetccl.so without rebuilding the vendor parts
cd hetccl/build
make -j hetccl/fast
```

```bash
# 1) AMD node (mpicxx in PATH)
cd hetccl/tools/hetccl-tests && mkdir build && cd build
cmake -DAMDGPU_TARGETS=gfx1100 -DCMAKE_CUDA_ARCHITECTURES=70 ..
make -j hetccl_tests_hip git_version_check

# 2) NVIDIA node
cd hetccl/tools/hetccl-tests/build
make -j hetccl_tests_cuda

# 3) AMD node
cd hetccl/tools/hetccl-tests/build
make -j $(for c in all_reduce all_gather reduce_scatter reduce broadcast alltoall alltoallv gather scatter sendrecv; do echo ${c}_perf/fast; done)
```

### Build Baselines (NCCL / RCCL)

HetCCL is built on [NCCL v2.18.3-1](https://github.com/NVIDIA/nccl/tree/v2.18.3-1) and [RCCL rocm-6.0.2](https://github.com/ROCm/rccl/tree/rocm-6.0.2). To compare against them:

```bash
# NCCL
git clone https://github.com/NVIDIA/nccl && cd nccl
git checkout v2.18.3-1
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"     # -> build/lib/libnccl.so

# nccl-tests (latest has a bug: https://github.com/NVIDIA/nccl-tests/issues/355)
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
git checkout da0b547
make -j NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70" MPI=1 MPI_HOME=$MPI_HOME NCCL_HOME=/path/to/nccl/build
```

```bash
# RCCL
git clone https://github.com/ROCm/rccl && cd rccl
git checkout rocm-6.0.2
mkdir build && cd build
CXX=/opt/rocm/bin/hipcc cmake -DBUILD_LOCAL_GPU_TARGET_ONLY=ON -DAMDGPU_TARGETS=gfx1100 ..
make -j                                                                    # -> build/librccl.so

# rccl-tests (latest has a bug with ROCm <= 6.3: https://github.com/ROCm/rccl-tests/pull/128)
git clone https://github.com/ROCm/rccl-tests && cd rccl-tests
git checkout 0abe3c8
mkdir build && cd build
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
    -DGPU_TARGETS=gfx1100 -DCMAKE_PREFIX_PATH=/path/to/rccl/build ..
make -j
```

## Evaluation

### Point-to-Point & Collective Communication

HetCCL is used by preloading `libhetccl.so`. For example, with `hetccl-tests`:

```bash
mpirun -mca pml ucx -mca btl ^openib \
  -H apa00:4,apa01:4,v05:4,v06:4 \
  -x LD_PRELOAD=$HETCCL_HOME/build/libhetccl.so \
  -x LD_LIBRARY_PATH=$HETCCL_HOME/build:$HETCCL_HOME/build/deps/tacc:$HETCCL_HOME/tools/hetccl-tests/build:$LD_LIBRARY_PATH \
  -x NCCL_ALGO=RING \
  $HETCCL_HOME/tools/hetccl-tests/build/all_reduce_perf -b 1K -e 1G -f 2 -w 10 -n 100
```

Available benchmarks: `sendrecv_perf`, `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, `reduce_perf`, `broadcast_perf`, `alltoall_perf`.
Add `-x NCCL_NET_GDR_LEVEL=SYS` to enable GPUDirect RDMA.

The scripts in [`scripts/`](scripts) reproduce the experiments in the paper.
Edit the host lists at the top of each script for your cluster; [`scripts/select_gpu.sh`](scripts/select_gpu.sh) pins the GPU closest to the NIC on each host and also needs to be adapted.
`HETCCL_HOME` defaults to the repository root and `MPIRUN` defaults to `mpirun`.

```bash
# HetCCL
bash scripts/run_hetccl_p2p_gdrdma.sh    # point-to-point with GPUDirect RDMA (1 GPU per node)
bash scripts/run_hetccl_coll.sh          # collectives, heterogeneous GPUs
bash scripts/run_hetccl_coll_homo.sh     # collectives, homogeneous GPUs

# NCCL baseline
NCCL_HOME=/path/to/nccl NCCL_TESTS_HOME=/path/to/nccl-tests bash scripts/run_nccl_p2p_gdrdma.sh
NCCL_HOME=/path/to/nccl NCCL_TESTS_HOME=/path/to/nccl-tests bash scripts/run_nccl_coll.sh

# RCCL baseline
RCCL_HOME=/path/to/rccl RCCL_TESTS_HOME=/path/to/rccl-tests bash scripts/run_rccl_p2p_gdrdma.sh
RCCL_HOME=/path/to/rccl RCCL_TESTS_HOME=/path/to/rccl-tests bash scripts/run_rccl_coll.sh
```

The GPU-aware MPI baseline is in [`tools/hetero-mpi-test`](tools/hetero-mpi-test).

### Training LLMs

Create the conda environments (one per vendor) on the corresponding nodes:

```bash
cd benchmark
bash setup_nvidia.sh   # creates conda env `hetccl-nvidia`
bash setup_amd.sh      # creates conda env `hetccl-amd`
```

`driver_*.sh` activates `hetccl-amd` on AMD nodes (detected by `/dev/kfd`) and `hetccl-nvidia` otherwise.
Set `CONDA_BASE` if conda is not installed in `~/miniconda3`, and `AMD_ENV` / `NVIDIA_ENV` to use different environment names.

```bash
cd benchmark

# HetCCL: 8 NVIDIA + 8 AMD GPUs, with GPU-aware workload balancing
MODEL=llama3.2-1b DS_STAGE=3 BALANCE=1 ./run_all.sh
# NCCL: 8 NVIDIA GPUs
MODEL=llama3.2-1b DS_STAGE=3 ./run_nvidia.sh
# RCCL: 8 AMD GPUs
MODEL=llama3.2-1b DS_STAGE=3 ./run_amd.sh
```

With `BALANCE=1`, each GPU is profiled briefly before training and its micro-batch size is set proportional to its measured speed, with the slowest GPU using `BSZ_PER_GPU` (Section 4.5 of the paper). On our cluster this gives 2 (V100) / 1 (W7800).

Hosts and the rendezvous address can be overridden, e.g. `HOSTS=a0:4,a1:4,n0:4,n1:4 MASTER_ADDR=a0 ./run_all.sh`.
Logs are written to `benchmark/logs/`.

Available models: `gpt2-125m`, `gpt2-355m`, `llama3.2-1b`, `llama3.2-3b`.
Loss-convergence experiments use `run_convergence_{het,nvidia,amd}.sh` (WikiText-103).

## Evaluation Results

Partial results on the system above.

**All-Reduce** (`all_reduce_perf`, 1 GB)

| Library  | # of Nodes (# of GPUs) | Bus Bandwidth (GB/s) |
|:--------:|:----------------------:|:--------------------:|
| NCCL     | 2 (8)                  | 5.77                 |
| RCCL     | 2 (8)                  | 7.18                 |
| HetCCL   | 2 (8)                  | 5.81                 |
| HetCCL   | 4 (16)                 | 5.80                 |

**LLM training** ([Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), DeepSpeed ZeRO-3)

| Library  | # of Nodes (# of GPUs) | Throughput (tokens/s) | Speedup (over RCCL) |
|:--------:|:----------------------:|:---------------------:|:-------------------:|
| NCCL     | 2 (8)                  | 20909                 | 1.92x               |
| RCCL     | 2 (8)                  | 10873                 | 1.00x               |
| HetCCL   | 2 (8)                  | 15526                 | 1.43x               |
| HetCCL   | 4 (16)                 | 30845                 | 2.83x               |

Training configuration: sequence length 8192, batch size per GPU 2 (NVIDIA) / 1 (AMD), ZeRO stage 3, FP16.

## License

HetCCL is derived from NCCL and RCCL. See [LICENSE.txt](LICENSE.txt) and [NOTICES.txt](NOTICES.txt).

## Citation

```bibtex
@inproceedings{hetccl2026,
  title     = {{HetCCL}: Efficient {LLM} Training on Heterogeneous Vendor {GPUs}},
  author    = {Kim, Heehoon and Lee, Jaehwan and Kim, Taejeoung and Park, Jongwon and Kim, Jinpyo and Suh, Pyongwon and Choi, Ryan H. and Lee, Sangwoo and Lee, Jaejin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
