#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE_M 8
#define TILE_N 64
#define TILE_K 64

// 按行平方和
__device__ void row_reduce_add(float v[TILE_M], const float* tile) {
    for (int i = 0; i < TILE_M; ++i) {
        float sumsq = 0.f;
        #pragma unroll
        for (int j = 0; j < TILE_K; ++j) {
            float x = tile[i * TILE_K + j];
            sumsq += x * x;
        }
        v[i] += sumsq;
    }
}

__device__ void apply_gain(float* tile, const __half* gain) {
    for (int i = 0; i < TILE_M; ++i)
        for (int j = 0; j < TILE_K; ++j)
            tile[i * TILE_K + j] *= __half2float(gain[j]);
}

__device__ void scale_row(float* row, float denom) {
    for (int j = 0; j < TILE_N; ++j)
        row[j] *= denom;
}

__global__ void rmsnorm_matmul_fused(
    const __half* __restrict__ X,
    const __half* __restrict__ G,
    const __half* __restrict__ W,
    __half* __restrict__ O,
    int M, int N, int K, float eps)
{
    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    __shared__ float A0[TILE_M * TILE_K];
    __shared__ float A1[TILE_M * TILE_K];
    __shared__ float B0[TILE_K * TILE_N];
    __shared__ float B1[TILE_K * TILE_N];

    float* Acur = A0; float* Bcur = B0;
    float* Anext = A1; float* Bnext = B1;

    float C[TILE_M][TILE_N] = {0};
    float v[TILE_M] = {0};

    // 预取第0块
    int k0 = 0;
    for (int i = 0; i < TILE_M; ++i) {
        int gr = tile_row + i;
        for (int j = 0; j < TILE_K; ++j) {
            int gc = k0 + j;
            Acur[i * TILE_K + j] = __half2float(X[gr * K + gc]);
        }
    }
    for (int i = 0; i < TILE_K; ++i) {
        int gr = k0 + i;
        for (int j = 0; j < TILE_N; ++j) {
            int gc = tile_col + j;
            Bcur[i * TILE_N + j] = __half2float(W[gr * N + gc]);
        }
    }
    __syncthreads();

    for (int k = 0; k < K; k += TILE_K) {
        // 预取下一块
        if (k + TILE_K < K) {
            int k1 = k + TILE_K;
            for (int i = 0; i < TILE_M; ++i) {
                int gr = tile_row + i;
                for (int j = 0; j < TILE_K; ++j) {
                    int gc = k1 + j;
                    Anext[i * TILE_K + j] = __half2float(X[gr * K + gc]);
                }
            }
            for (int i = 0; i < TILE_K; ++i) {
                int gr = k1 + i;
                for (int j = 0; j < TILE_N; ++j) {
                    int gc = tile_col + j;
                    Bnext[i * TILE_N + j] = __half2float(W[gr * N + gc]);
                }
            }
        }

        row_reduce_add(v, Acur);
        apply_gain(Acur, G + k);

        // FMA（可改 Tensor Core）
        for (int i = 0; i < TILE_M; ++i)
            for (int j = 0; j < TILE_N; ++j) {
                float sum = 0.f;
                for (int kk = 0; kk < TILE_K; ++kk)
                    sum += Acur[i * TILE_K + kk] * Bcur[kk * TILE_N + j];
                C[i][j] += sum;
            }

        __syncthreads();
        float* tmpA = Acur; Acur = Anext; Anext = tmpA;
        float* tmpB = Bcur; Bcur = Bnext; Bnext = tmpB;
        __syncthreads();
    }

    // RMS 归一化
    for (int i = 0; i < TILE_M; ++i) {
        float denom = rsqrtf(v[i] / float(K) + eps);
        scale_row(C[i], denom);
    }

    // 写回输出
    for (int i = 0; i < TILE_M; ++i) {
        int gr = tile_row + i;
        for (int j = 0; j < TILE_N; ++j) {
            int gc = tile_col + j;
            O[gr * N + gc] = __float2half(C[i][j]);
        }
    }
}

int main() {
    const int M = 16, K = 4096, N = 4096;
    float eps = 1e-6;

    size_t size_X = M * K * sizeof(__half);
    size_t size_G = K * sizeof(__half);
    size_t size_W = K * N * sizeof(__half);
    size_t size_O = M * N * sizeof(__half);

    __half *hX = (__half*)malloc(size_X);
    __half *hG = (__half*)malloc(size_G);
    __half *hW = (__half*)malloc(size_W);
    __half *hO = (__half*)malloc(size_O);

    srand(42);
    auto rnd_half = [](){ return __float2half(((float)rand()/RAND_MAX - 0.5f) * 2.f); };

    for (int i = 0; i < M*K; ++i) hX[i] = rnd_half();
    for (int i = 0; i < K; ++i)   hG[i] = rnd_half();
    for (int i = 0; i < K*N; ++i) hW[i] = rnd_half();

    __half *dX, *dG, *dW, *dO;
    cudaMalloc(&dX, size_X);
    cudaMalloc(&dG, size_G);
    cudaMalloc(&dW, size_W);
    cudaMalloc(&dO, size_O);

    cudaMemcpy(dX, hX, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(dG, hG, size_G, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, size_W, cudaMemcpyHostToDevice);

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(1, 1); // 每线程块内部使用共享内存+for循环遍历

    rmsnorm_matmul_fused<<<grid, block>>>(dX, dG, dW, dO, M, N, K, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(hO, dO, size_O, cudaMemcpyDeviceToHost);

    // 打印部分输出
    printf("O[0:2,0:5]:\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            printf("%f ", __half2float(hO[i*N + j]));
        }
        printf("\n");
    }

    cudaFree(dX); cudaFree(dG); cudaFree(dW); cudaFree(dO);
    free(hX); free(hG); free(hW); free(hO);
}
