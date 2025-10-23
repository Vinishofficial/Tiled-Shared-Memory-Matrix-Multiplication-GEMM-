#  Tiled Shared-Memory Matrix Multiplication (CUDA GEMM)

##  Project Overview
This project implements a **GPU-accelerated Matrix Multiplication (GEMM)** using **CUDA shared memory tiling** to achieve high performance and scalability. It compares GPU performance with a CPU baseline and analyzes the effect of tile size and matrix dimension on throughput.

---

##  Key Features
- CUDA-based matrix multiplication using **shared memory tiling**
- Configurable **matrix size (N)** and **tile size (T)**
- **Performance benchmarking** with GFLOPS calculation
- Optional **CPU verification** for correctness
- Scalable design for performance analysis

---

## code
```
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

inline void checkCuda(cudaError_t e, const char *m) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", m, cudaGetErrorString(e));
        exit(1);
    }
}

__global__ void matMulTiled(const float* A, const float* B, float* C, int N, int tile) {
    extern __shared__ float shared[];
    float* As = shared;
    float* Bs = shared + tile * tile;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile + ty;
    int col = blockIdx.x * tile + tx;
    float val = 0.0f;
    for (int m = 0; m < (N + tile - 1) / tile; ++m) {
        int aCol = m * tile + tx;
        int bRow = m * tile + ty;
        if (row < N && aCol < N) As[ty * tile + tx] = A[row * N + aCol];
        else As[ty * tile + tx] = 0.0f;
        if (bRow < N && col < N) Bs[ty * tile + tx] = B[bRow * N + col];
        else Bs[ty * tile + tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < tile; ++k) val += As[ty * tile + k] * Bs[k * tile + tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = val;
}

void cpuMatMul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += (double)A[i*N + k] * (double)B[k*N + j];
            C[i*N + j] = (float)s;
        }
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    int tile = 32;
    int verify = 1;
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) tile = atoi(argv[2]);
    if (argc >= 4) verify = atoi(argv[3]);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_Cref = (float*)malloc(bytes);

    srand(0);
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
        h_C[i] = 0.0f;
        h_Cref[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes), "malloc A");
    checkCuda(cudaMalloc(&d_B, bytes), "malloc B");
    checkCuda(cudaMalloc(&d_C, bytes), "malloc C");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "copy B");

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);
    size_t sharedBytes = 2 * tile * tile * sizeof(float);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start");
    checkCuda(cudaEventCreate(&stop), "create stop");
    checkCuda(cudaEventRecord(start), "record start");
    matMulTiled<<<grid, block, sharedBytes>>>(d_A, d_B, d_C, N, tile);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "sync stop");
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "copy back");

    double gflops = 2.0 * (double)N * N * N / (ms / 1000.0) / 1e9;
    printf("N=%d tile=%d time_ms=%.3f GFLOPS=%.2f\n", N, tile, ms, gflops);

    if (verify) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuMatMul(h_A, h_B, h_Cref, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double maxErr = 0.0;
        for (int i = 0; i < N*N; ++i) {
            double e = fabs((double)h_Cref[i] - (double)h_C[i]);
            if (e > maxErr) maxErr = e;
        }
        printf("CPU_time_ms=%.3f max_abs_error=%.6e\n", cpu_ms, maxErr);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_Cref);
    return 0;
}
```

##  Build & Run
### Compile
```bash
nvcc -O3 -arch=sm_60 -o tiled_gemm tiled_gemm.cu
```

## sample output
```
N=1024 tile=32 time_ms=6.74 GFLOPS=318.50
CPU_time_ms=3845.21 max_abs_error=1.200000e-05
```

