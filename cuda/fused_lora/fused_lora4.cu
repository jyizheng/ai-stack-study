#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// 简化参数
#define RANK 128

// warp‐local reduction helper
__inline__ __device__
float warp_reduce_sum(float v) {
    // 全 mask，所有 lane 都参与
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// 每个 warp 计算一个 (m,n) 输出点
// Y[m,n] = sum_k X[m,k]*W[k,n] + sum_k sum_r X[m,k]*A[k,r]*B[r,n]
extern "C" __global__
void fused_lora_warp_shuffle(
    const float* __restrict__ X,   // [M×K]
    const float* __restrict__ W,   // [K×N]
    const float* __restrict__ A,   // [K×RANK]
    const float* __restrict__ B,   // [RANK×N]
          float* __restrict__ Y,   // [M×N]
    int M, int N, int K)
{
    // blockIdx.xy 定义 (m,n)
    int m = blockIdx.y;
    int n = blockIdx.x;
    if (m >= M || n >= N) return;

    // warp 内的 lane id
    int lane = threadIdx.x;  // 0..31

    // 1. 普通 GEMM 累加
    float accWX = 0.f;
    for (int k = 0; k < K; ++k) {
        accWX += X[m * K + k] * W[k * N + n];
    }

    // 2. LoRA 部分：跨线程分担 RANK
    float accLoRA_part = 0.f;
    for (int k = 0; k < K; ++k) {
        float xmk = X[m * K + k];
        // lane 以 stride=WARP_SIZE 扫描 RANK 维度
        for (int r = lane; r < RANK; r += WARP_SIZE) {
            accLoRA_part += xmk
                * A[k * RANK + r]
                * B[r * N   + n];
        }
    }

    // 3. warp‐shuffle 把各 lane 的部分和汇聚到 lane 0
    float accLoRA = warp_reduce_sum(accLoRA_part);

    // 4. lane 0 写回
    if (lane == 0) {
        Y[m * N + n] = accWX + accLoRA;
    }
}

// Host 测试
int main() {
    const int M = 128, K = 256, N = 128;
    printf("Warp‐shuffle fused LoRA: M=%d K=%d N=%d RANK=%d\n",
           M, K, N, RANK);

    size_t sizeX = size_t(M)*K*sizeof(float);
    size_t sizeW = size_t(K)*N*sizeof(float);
    size_t sizeA = size_t(K)*RANK*sizeof(float);
    size_t sizeB = size_t(RANK)*N*sizeof(float);
    size_t sizeY = size_t(M)*N*sizeof(float);

    // 分配并初始化 host 内存
    float *hX = (float*)malloc(sizeX),
          *hW = (float*)malloc(sizeW),
          *hA = (float*)malloc(sizeA),
          *hB = (float*)malloc(sizeB),
          *hY = (float*)malloc(sizeY);
    srand(0);
    auto rnd = [&](){ return (rand()/float(RAND_MAX)-0.5f)*2.0f; };
    for (int i = 0; i < M*K; ++i)  hX[i] = rnd();
    for (int i = 0; i < K*N; ++i)  hW[i] = rnd();
    for (int i = 0; i < K*RANK; ++i) hA[i] = rnd();
    for (int i = 0; i < RANK*N; ++i) hB[i] = rnd();

    // 分配 device 内存
    float *dX, *dW, *dA, *dB, *dY;
    cudaMalloc(&dX, sizeX);
    cudaMalloc(&dW, sizeW);
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dY, sizeY);

    // 拷贝数据到 GPU
    cudaMemcpy(dX, hX, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // launch: 每个 block 一个 warp，grid 大小覆盖 M×N
    dim3 block(WARP_SIZE, 1);
    dim3 grid((N + 0) / 1, (M + 0) / 1);
    fused_lora_warp_shuffle<<<grid, block>>>(
        dX, dW, dA, dB, dY, M, N, K);
    cudaDeviceSynchronize();

    // 拷回并展示首 few 值
    cudaMemcpy(hY, dY, sizeY, cudaMemcpyDeviceToHost);
    printf("Y[0]=%f  Y[M*N-1]=%f\n",
           hY[0], hY[M*N-1]);

    // cleanup
    cudaFree(dX); cudaFree(dW);
    cudaFree(dA); cudaFree(dB);
    cudaFree(dY);
    free(hX); free(hW);
    free(hA); free(hB);
    free(hY);
    return 0;
}


