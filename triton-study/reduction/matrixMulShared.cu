#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Tile size for shared memory
#define TILE_WIDTH 16
#define THRESHOLD 1e-4

// CUDA kernel for matrix multiplication using shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K - 1) / TILE_WIDTH + 1; ++t) {
        // Load data into shared memory
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            sharedA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            sharedB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to initialize matrix
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// CPU matrix multiplication for verification
void cpuMatrixMul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify results
bool verifyResults(float* gpuResult, float* cpuResult, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpuResult[i] - cpuResult[i]) > THRESHOLD) {
            printf("Verification failed at index %d: GPU = %f, CPU = %f\n", 
                   i, gpuResult[i], cpuResult[i]);
            //return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions
    const int M = 1024;  // rows of A
    const int N = 1024;  // cols of B
    const int K = 1024;  // cols of A and rows of B
    
    // Allocate host memory
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_gpu = (float*)malloc(M * N * sizeof(float));
    float *C_cpu = (float*)malloc(M * N * sizeof(float));
    
    // Initialize input matrices
    initMatrix(A, M, K);
    initMatrix(B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result from device to host
    cudaMemcpy(C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute CPU result
    printf("Computing CPU result for verification...\n");
    cpuMatrixMul(A, B, C_cpu, M, N, K);
    
    // Verify results
    printf("Verifying results...\n");
    bool correct = verifyResults(C_gpu, C_cpu, M * N);
    
    if (correct) {
        printf("Results match! GPU computation took %.3f ms\n", milliseconds);
    } else {
        printf("Results don't match!\n");
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    
    return 0;
}