#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Kernel for parallel reduction using shared memory
__global__ void reduceSum(float* input, float* output, int size) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    // Global thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localTid = threadIdx.x;
    
    // Load data into shared memory
    sharedData[localTid] = (tid < size) ? input[tid] : 0;
    __syncthreads();
    
    // Perform reduction in shared memory
    // Unrolling warp for efficiency
    if (localTid < 128) { sharedData[localTid] += sharedData[localTid + 128]; } __syncthreads();
    if (localTid < 64)  { sharedData[localTid] += sharedData[localTid + 64];  } __syncthreads();
    if (localTid < 32)  { sharedData[localTid] += sharedData[localTid + 32];  }
    
    // Warp reduction (no sync needed as warp executes in lock-step)
    if (localTid < 16) {
        sharedData[localTid] += sharedData[localTid + 16];
        sharedData[localTid] += sharedData[localTid + 8];
        sharedData[localTid] += sharedData[localTid + 4];
        sharedData[localTid] += sharedData[localTid + 2];
        sharedData[localTid] += sharedData[localTid + 1];
    }
    
    // Write result for this block to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CPU reduction for verification
float cpuReduce(float* data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const int SIZE = 1024 * 1024;  // 1M elements
    const int bytes = SIZE * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1.0f;  // Initialize with 1s for easy verification
    }
    
    // Calculate grid dimensions
    int blockSize = BLOCK_SIZE;
    int numBlocks = (SIZE + blockSize - 1) / blockSize;
    int outputBytes = numBlocks * sizeof(float);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, outputBytes);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    reduceSum<<<numBlocks, blockSize>>>(d_input, d_output, SIZE);
    
    // If necessary, reduce the partial sums with another kernel launch
    float* h_output = (float*)malloc(outputBytes);
    cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);
    
    // Sum up partial results on CPU (for simplicity)
    float gpu_sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        gpu_sum += h_output[i];
    }
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // CPU reduction for verification
    float cpu_sum = cpuReduce(h_input, SIZE);
    
    // Print results
    printf("GPU sum: %.6f\n", gpu_sum);
    printf("CPU sum: %.6f\n", cpu_sum);
    printf("Time taken: %.3f ms\n", milliseconds);
    printf("Relative error: %.10f%%\n", 
           100.0f * fabs(gpu_sum - cpu_sum) / cpu_sum);
    
    // Memory cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
    free(h_output);
    
    return 0;
}

