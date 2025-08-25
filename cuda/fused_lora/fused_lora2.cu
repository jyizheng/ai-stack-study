#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// problem sizes
#define TILE_M   64
#define TILE_N   64
#define TILE_K   32
#define RANK      4

// fused LoRA kernel: Y = X·W + X·A·B
extern "C"
__global__ void fused_lora_kernel(
    const float* __restrict__ X,   // [M×K]
    const float* __restrict__ W,   // [K×N]
    const float* __restrict__ A,   // [K×RANK]
    const float* __restrict__ B,   // [RANK×N]
    float*       __restrict__ Y,   // [M×N]
    int M, int N, int K)
{
    int bm = blockIdx.y, bn = blockIdx.x;
    int row0 = bm * TILE_M;
    int col0 = bn * TILE_N;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = row0 + ty;
    int col = col0 + tx;

    // shared‐memory tiles
    __shared__ float sX[TILE_M][TILE_K];
    __shared__ float sW[TILE_K][TILE_N];
    __shared__ float sA[TILE_K][RANK];
    __shared__ float sB[RANK][TILE_N];

    // register accumulators
    float accWX    = 0.0f;
    float accLoRA  = 0.0f;

    // loop over K in chunks
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // load X‐tile
        if (row < M && (k0 + tx) < K) {
            sX[ty][tx] = X[row * K + (k0 + tx)];
        } else {
            sX[ty][tx] = 0.0f;
        }
        // load W‐tile
        if ((k0 + ty) < K && col < N) {
            sW[ty][tx] = W[(k0 + ty) * N + col];
        } else {
            sW[ty][tx] = 0.0f;
        }
        // load A‐tile (small RANK)
        if ((k0 + ty) < K) {
            #pragma unroll
            for (int r = 0; r < RANK; ++r) {
                sA[ty][r] = A[(k0 + ty) * RANK + r];
            }
        }
        // load B‐tile
        if (col < N) {
            #pragma unroll
            for (int r = 0; r < RANK; ++r) {
                sB[r][tx] = B[r * N + col];
            }
        }
        __syncthreads();

        // compute tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float xval = sX[ty][kk];
            float wval = sW[kk][tx];
            accWX += xval * wval;

            // inline small‐rank reduction
            #pragma unroll
            for (int r = 0; r < RANK; ++r) {
                accLoRA += xval * sA[kk][r] * sB[r][tx];
            }
        }
        __syncthreads();
    }

    // write out if in bounds
    if (row < M && col < N) {
        Y[row * N + col] = accWX + accLoRA;
    }
}

// naive CPU version for reference
void fused_lora_cpu(
    const float* X, const float* W,
    const float* A, const float* B,
    float*       Y,
    int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sumWX = 0.f;
            float tmp[RANK] = {0};
            // accumulate X·W and X·A
            for (int k = 0; k < K; ++k) {
                float xkn = X[m * K + k];
                sumWX += xkn * W[k * N + n];
                for (int r = 0; r < RANK; ++r) {
                    tmp[r] += xkn * A[k * RANK + r];
                }
            }
            // finish LoRA term
            float sumLoRA = 0.f;
            for (int r = 0; r < RANK; ++r) {
                sumLoRA += tmp[r] * B[r * N + n];
            }
            Y[m * N + n] = sumWX + sumLoRA;
        }
    }
}

int main() {
    // dimensions (must be multiples of tile or will be guarded)
    const int M = 128, K = 64, N = 128;
    printf("Running fused LoRA: M=%d, K=%d, N=%d, RANK=%d\n",
           M, K, N, RANK);

    size_t sizeX = size_t(M) * K * sizeof(float);
    size_t sizeW = size_t(K) * N * sizeof(float);
    size_t sizeA = size_t(K) * RANK * sizeof(float);
    size_t sizeB = size_t(RANK) * N * sizeof(float);
    size_t sizeY = size_t(M) * N * sizeof(float);

    // host buffers
    float *hX  = (float*)malloc(sizeX),
          *hW  = (float*)malloc(sizeW),
          *hA  = (float*)malloc(sizeA),
          *hB  = (float*)malloc(sizeB),
          *hY  = (float*)malloc(sizeY),
          *hYc = (float*)malloc(sizeY);

    // initialize with random
    srand(0);
    auto rnd = [&](){ return (rand() / float(RAND_MAX) - 0.5f) * 2.0f; };
    for (int i = 0; i < M*K;      ++i) hX[i] = rnd();
    for (int i = 0; i < K*N;      ++i) hW[i] = rnd();
    for (int i = 0; i < K*RANK;   ++i) hA[i] = rnd();
    for (int i = 0; i < RANK*N;   ++i) hB[i] = rnd();

    // device buffers
    float *dX, *dW, *dA, *dB, *dY;
    cudaMalloc(&dX, sizeX);
    cudaMalloc(&dW, sizeW);
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dY, sizeY);

    // copy inputs
    cudaMemcpy(dX, hX, sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 block(32, 8);  // covers TILE_M×TILE_N in loops
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);
    fused_lora_kernel<<<grid, block>>>(dX, dW, dA, dB, dY, M, N, K);
    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(hY, dY, sizeY, cudaMemcpyDeviceToHost);

    // CPU reference
    fused_lora_cpu(hX, hW, hA, hB, hYc, M, N, K);

    // compare & print a small corner
    double maxErr = 0, l1 = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double e = fabs(hY[m*N + n] - hYc[m*N + n]);
            maxErr = fmax(maxErr, e);
            l1    += e;
        }
    }
    l1 /= (M * N);

    printf("max error = %g, avg error = %g\n", maxErr, l1);
    printf("GPU[0..3,0..3] vs CPU[0..3,0..3]:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("% .4f/% .4f  ",
                   hY[i*N + j], hYc[i*N + j]);
        }
        printf("\n");
    }

    // cleanup
    cudaFree(dX); cudaFree(dW);
    cudaFree(dA); cudaFree(dB);
    cudaFree(dY);
    free(hX); free(hW);
    free(hA); free(hB);
    free(hY); free(hYc);

    return 0;
}



