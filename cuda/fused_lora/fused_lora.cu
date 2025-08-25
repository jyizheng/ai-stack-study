#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief fused_lora_kernel
 *   Compute O[m,n] = sum_k W[m,k]*X[k,n]
 *                  + sum_r B[m,r] * (sum_k A[r,k]*X[k,n])
 *
 * @param X   [D x N]
 * @param W   [M x D]
 * @param A   [R x D]
 * @param B   [M x R]
 * @param O   [M x N]  output
 * @param D   input feature dim
 * @param M   output rows
 * @param N   output cols
 * @param R   adapter rank
 */
__global__
void fused_lora_kernel(const float* __restrict__ X,
                       const float* __restrict__ W,
                       const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ O,
                       int D, int M, int N, int R) {
  // each thread computes one (m,n) element
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  // 1. 计算主分支 W*X
  float sumWX = 0.0f;
  // 2. 计算中间低秩AX，并累积到 accumR[r]
  extern __shared__ float scratch[];      // optional, not used here
  float accumR[16];                       // 假设 R <= 16
  for (int r = 0; r < R; ++r) accumR[r] = 0.0f;

  for (int k = 0; k < D; ++k) {
    float xkn = X[k * N + n];
    sumWX += W[m * D + k] * xkn;
    // low‐rank 部分
    for (int r = 0; r < R; ++r) {
      accumR[r] += A[r * D + k] * xkn;
    }
  }

  // 3. B*(A*X) + 主分支 和并
  float out = sumWX;
  for (int r = 0; r < R; ++r) {
    out += B[m * R + r] * accumR[r];
  }

  O[m * N + n] = out;
}

int main() {
  // 假设 D=128, M=64, N=256, R=4
  const int D = 128, M = 64, N = 256, R = 4;
  size_t sizeX = D * N * sizeof(float);
  size_t sizeW = M * D * sizeof(float);
  size_t sizeA = R * D * sizeof(float);
  size_t sizeB = M * R * sizeof(float);
  size_t sizeO = M * N * sizeof(float);

  // 分配 host & device 内存
  float *h_X = (float*)malloc(sizeX), *h_W = (float*)malloc(sizeW),
        *h_A = (float*)malloc(sizeA), *h_B = (float*)malloc(sizeB),
        *h_O = (float*)malloc(sizeO);

  // 初始化（这里随便填随机值）
  for (int i = 0; i < D * N; ++i) h_X[i] = 1.0f;
  for (int i = 0; i < M * D; ++i) h_W[i] = 0.5f;
  for (int i = 0; i < R * D; ++i) h_A[i] = 0.1f;
  for (int i = 0; i < M * R; ++i) h_B[i] = 0.2f;

  float *d_X, *d_W, *d_A, *d_B, *d_O;
  CHECK_CUDA(cudaMalloc(&d_X, sizeX));
  CHECK_CUDA(cudaMalloc(&d_W, sizeW));
  CHECK_CUDA(cudaMalloc(&d_A, sizeA));
  CHECK_CUDA(cudaMalloc(&d_B, sizeB));
  CHECK_CUDA(cudaMalloc(&d_O, sizeO));

  CHECK_CUDA(cudaMemcpy(d_X, h_X, sizeX, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W, h_W, sizeW, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

  // Kernel 配置
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x,
            (N + block.y - 1) / block.y);

  // 启动 kernel
  fused_lora_kernel<<<grid, block>>>(d_X, d_W, d_A, d_B, d_O, D, M, N, R);
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // 拷回结果
  CHECK_CUDA(cudaMemcpy(h_O, d_O, sizeO, cudaMemcpyDeviceToHost));

  printf("O[0]=%f\n", h_O[0]);

  // 清理
  cudaFree(d_X); cudaFree(d_W);
  cudaFree(d_A); cudaFree(d_B);
  cudaFree(d_O);
  free(h_X); free(h_W);
  free(h_A); free(h_B); free(h_O);
  return 0;
}



