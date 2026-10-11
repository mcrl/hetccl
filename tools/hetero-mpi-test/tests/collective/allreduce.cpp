#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Backend implementation headers (Assumed to be same as your SendRecv example)
#if defined(USE_CUDA)
#include "cuda_backend.h"
#elif defined(USE_HIP)
#include "hip_backend.h"
#else
#error "Define either USE_CUDA or USE_HIP"
#endif

// Configuration
#define WARMUP_ITERS 5
#define MEASURE_ITERS 20

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // AllReduce works with N >= 2
    if (size < 2) {
        if (rank == 0) printf("This test requires at least 2 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#if defined(USE_CUDA)
    const char* platform = "CUDA";
#elif defined(USE_HIP)
    const char* platform = "HIP";
#endif
    
    // Simple verification print
    if (rank == 0) {
        printf("# AllReduce Performance Test\n");
        printf("# Backend: %s\n", platform);
        printf("# Ranks: %d\n", size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // nccl-test style header print
    if (rank == 0) {
        printf("# %10s  %12s  %8s  %8s\n", "Size(B)", "Time(us)", "BusBW", "AlgBW");
        printf("# %10s  %12s  %8s  %8s\n", "", "(Avg)", "(GB/s)", "(GB/s)");
    }

    // Only 1KB ~ 64MB for AllReduce (17 steps)
    const int num_steps = 17;
    size_t sizes[num_steps];
    for (int s = 0; s < num_steps; s++) {
        sizes[s] = 1024UL << s;
    }

    // Pre-calculate BusBW correction factor for AllReduce
    // Factor = 2 * (n - 1) / n
    double bus_bw_factor = ((double)(2 * (size - 1))) / ((double)size);

    for (int s = 0; s < num_steps; s++) {
        size_t count = sizes[s];

        // 1. Allocate and initialize buffers
        // Note: We use out-of-place AllReduce (send_buf -> recv_buf) to match standard benchmarks
        void *send_buf = gpu_malloc(count);
        void *recv_buf = gpu_malloc(count);
        
        // Initialize data
        gpu_memset(send_buf, rank, count);
        gpu_memset(recv_buf, 0, count);

        MPI_Barrier(MPI_COMM_WORLD);

        double total_time = 0.0;

        // 2. WARMUP + MEASURE
        for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
            
            // Ensure all ranks are ready before starting the timer
            MPI_Barrier(MPI_COMM_WORLD);

            double t0 = 0.0;
            if (i >= WARMUP_ITERS) {
                t0 = MPI_Wtime();
            }

            // *** Core AllReduce Logic ***
            // Assuming CUDA-aware MPI (passing GPU pointers directly)
            MPI_Allreduce(send_buf, recv_buf, count, MPI_BYTE, MPI_SUM, MPI_COMM_WORLD);

            // ------------------------------------

            if (i >= WARMUP_ITERS) {
                double t1 = MPI_Wtime();
                total_time += (t1 - t0);
            }
        }

        // 3. Result calculation (nccl-test allreduce formula applied)
        if (rank == 0) {
            double avg_time_sec = total_time / MEASURE_ITERS;
            double avg_time_us = avg_time_sec * 1e6;

            // AlgBW = Size / Time
            double base_bw = (double)(count) / 1.0E9 / avg_time_sec;
            
            // BusBW = AlgBW * Factor
            double bus_bw = base_bw * bus_bw_factor; 
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