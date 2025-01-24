#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Tree-based reduction kernel using shared memory
__global__ void reduceTreeBased(float* input, float* output, int size) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    // Load data into shared memory
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localTid = threadIdx.x;
    
    // Initialize shared memory
    sharedData[localTid] = (tid < size) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Perform tree-based reduction in shared memory
    // Each iteration reduces the active threads by half
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// Tree-based reduction kernel with improved bank conflicts handling
__global__ void reduceTreeBasedOptimized(float* input, float* output, int size) {
    __shared__ float sharedData[BLOCK_SIZE];
    
    // Load data into shared memory with sequential addressing
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localTid = threadIdx.x;
    
    sharedData[localTid] = (tid < size) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Perform tree-based reduction with sequential addressing
    for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (localTid < stride) {
            sharedData[localTid] += sharedData[localTid + stride];
        }
        __syncthreads();
    }
    
    // Unrolled warp reduction (last 32 threads)
    if (localTid < 32) {
        volatile float* smem = sharedData;
        smem[localTid] += smem[localTid + 32];
        smem[localTid] += smem[localTid + 16];
        smem[localTid] += smem[localTid + 8];
        smem[localTid] += smem[localTid + 4];
        smem[localTid] += smem[localTid + 2];
        smem[localTid] += smem[localTid + 1];
    }
    
    // Write result
    if (localTid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// Sequential CPU reduction for verification
float cpuReduce(float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

void runBenchmark(const char* name, void (*kernel)(float*, float*, int), 
                 float* d_input, float* d_output, float* h_output,
                 int size, int numBlocks, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    // Timed run
    cudaEventRecord(start);
    kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    cudaEventRecord(stop);
    
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Sum partial results
    float gpu_sum = 0;
    for (int i = 0; i < numBlocks; i++) {
        gpu_sum += h_output[i];
    }
    
    printf("\n%s Results:\n", name);
    printf("Sum: %.6f\n", gpu_sum);
    printf("Time: %.3f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int SIZE = 1024 * 1024;  // 1M elements
    const int bytes = SIZE * sizeof(float);
    
    // Allocate and initialize host memory
    float* h_input = (float*)malloc(bytes);
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1.0f;  // Initialize with 1s for easy verification
    }
    
    // Calculate grid dimensions
    int blockSize = BLOCK_SIZE;
    int numBlocks = (SIZE + blockSize - 1) / blockSize;
    float* h_output = (float*)malloc(numBlocks * sizeof(float));
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Run basic tree-based reduction
    runBenchmark("Basic Tree Reduction", 
                 (void (*)(float*, float*, int))reduceTreeBased,
                 d_input, d_output, h_output, SIZE, numBlocks, blockSize);
    
    // Run optimized tree-based reduction
    runBenchmark("Optimized Tree Reduction", 
                 (void (*)(float*, float*, int))reduceTreeBasedOptimized,
                 d_input, d_output, h_output, SIZE, numBlocks, blockSize);
    
    // CPU reduction for verification
    float cpu_sum = cpuReduce(h_input, SIZE);
    printf("\nCPU Sum: %.6f\n", cpu_sum);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

