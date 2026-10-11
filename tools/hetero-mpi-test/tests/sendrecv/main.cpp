// For NVIDIA:
// mpicxx -DUSE_CUDA -o sendrecv_cuda main.cpp cuda_backend.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
// For AMD:
// mpicxx -DUSE_HIP -D__HIP_PLATFORM_AMD__ -o sendrecv_hip main.cpp hip_backend.cpp -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64

// main.cpp
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(USE_CUDA)
#include "cuda_backend.h"
#elif defined(USE_HIP)
#include "hip_backend.h"
#else
#error "Define either USE_CUDA or USE_HIP"
#endif

// Set similar to nccl-test defaults
#define WARMUP_ITERS 5
#define MEASURE_ITERS 20

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("This test requires exactly 2 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#if defined(USE_CUDA)
    const char* platform = "CUDA";
#elif defined(USE_HIP)
    const char* platform = "HIP";
#endif
    
    printf("# Rank %d using %s backend\n", rank, platform);
    MPI_Barrier(MPI_COMM_WORLD);

    // nccl-test style header print
    if (rank == 0) {
        printf("# %10s  %12s  %8s  %8s\n", "Size(B)", "Time(us)", "BusBW", "AlgBW");
        printf("# %10s  %12s  %8s  %8s\n", "", "(Avg)", "(GB/s)", "(GB/s)");
    }

    // 1KB to 64MB sizes (17 steps)
    const int num_steps = 17;
    size_t sizes[num_steps];
    for (int s = 0; s < num_steps; s++) {
        sizes[s] = 1024UL << s; 
    }

    for (int s = 0; s < num_steps; s++) {
        size_t count = sizes[s];

        // 1. Allocate and initialize buffers
        void *send_buf = gpu_malloc(count);
        void *recv_buf = gpu_malloc(count);
        gpu_memset(send_buf, rank, count);
        gpu_memset(recv_buf, 0, count);

        MPI_Barrier(MPI_COMM_WORLD);

        double total_time = 0.0;

        // 2. WARMUP + MEASURE
        for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
            
            // Measure time starting after warmup
            double t0 = 0.0;
            if (i >= WARMUP_ITERS) {
                MPI_Barrier(MPI_COMM_WORLD); 
                t0 = MPI_Wtime();
            }

            // Reproduces the nccl-tests SendRecv logic
            // ncclGroupStart() -> Send & Recv -> ncclGroupEnd()
            // For MPI, we use non-blocking calls and waitall to mimic this behavior
            
            MPI_Request requests[2];
            int peer = (rank == 0) ? 1 : 0; // peer rank

            // Post Recv first (common practice)
            MPI_Irecv(recv_buf, count, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &requests[0]);
            // Post Send
            MPI_Isend(send_buf, count, MPI_BYTE, peer, 0, MPI_COMM_WORLD, &requests[1]);
            
            // Wait for both to complete (mimics ncclGroupEnd + Synchronize)
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            // ------------------------------------

            if (i >= WARMUP_ITERS) {
                double t1 = MPI_Wtime();
                total_time += (t1 - t0);
            }
        }

        // 3. Result calculation (nccl-test formula applied)
        if (rank == 0) {
            double avg_time_sec = total_time / MEASURE_ITERS;
            double avg_time_us = avg_time_sec * 1e6;

            // nccl-test SendRecv formula applied
            // AlgBW = count / time (based on one-way data size)
            // BusBW = AlgBW * 1 (for SendRecv, factor is 1)
            double base_bw = (double)(count) / 1.0E9 / avg_time_sec;
            double bus_bw = base_bw; 
            double alg_bw = base_bw;

            printf("  %10zu  %12.2f  %8.2f  %8.2f\n", 
                   count, avg_time_us, bus_bw, alg_bw);
            fflush(stdout);
        }

        gpu_free(send_buf);
        gpu_free(recv_buf);
    }

    MPI_Finalize();
    return 0;
}