# HETCCL Tests

Tests for evaluating performance and the correctness of HETCCL operations.

## Build

```shell
mkdir build && cd build
cmake -DAMDGPU_TARGETS=gfx908:xnack- .. |& tee ../cmake.log
make -j `nproc` |& tee ../make.log
```

## Usage

### Example

```bash
mpirun -mca btl ^openib -mca pml ucx -H <hosts> \
  -x LD_PRELOAD=$HOME/hetccl-ws/hetccl/build/libhetccl.so \
  -x LD_LIBRARY_PATH=$HOME/hetccl-ws/hetccl/build:$HOME/hetccl-ws/hetccl/tools/hetccl-tests/build:$LD_LIBRARY_PATH \
  /path/to/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```


### Arguments

Identical to the arguments of [rccl-tests](https://github.com/ROCm/rccl-tests?tab=readme-ov-file#arguments).


