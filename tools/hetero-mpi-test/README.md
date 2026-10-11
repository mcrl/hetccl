# Hetero-Vendor GPU MPI Test

This test may work on OpenMPI version >= 5.x. 

However, we have only verified this heterogeneous OpenMPI feature on OpenMPI v5.0.2.


## Build UCX + OpenMPI (v5.0.2) 

### Hetero GPU-aware UCX

```bash
./configure --prefix=$HOME/opt/ucx --enable-mt --with-cuda=/usr/local/cuda --with-rocm=/opt/rocm

make -j $(nproc) && make -j $(nproc) install
```

### Hetero GPU-aware OpenMPI

```bash
./configure --prefix=$HOME/opt/openmpi --with-ucx=$HOME/opt/ucx --with-cuda=/usr/local/cuda --with-cuda-libdir=/usr/local/cuda/lib64/stubs --with-rocm=/opt/rocm

make -j $(nproc) install
```

## Build Hetero-MPI Test Code
```bash
./script/compile_all.sh
```

## Run

```bash
./script/run_all.sh 
```

### Tested MPI Operations

- MPI_Send, MPI_Recv
- MPI_Allreduce, MPI_Allgather