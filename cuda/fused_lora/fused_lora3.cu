#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// problem sizes and tile parameters
#define TILE_M   64
#define TILE_N   64
#define TILE_K   32
#define RANK      8   // example large rank
// register-tile per thread
#define THREAD_TILE_M   2
#define THREAD_TILE_N   2

// warp-shuffle helper (for large RANK)
// float warp_reduce_sum(float val) {
//     for (int offset = 16; offset > 0; offset >>= 1) {
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     }
//     return val;
// }

// fused LoRA kernel: Y = X·W + X·A·B
extern "C"
__global__ void fused_lora_opt_kernel(
    const float* __restrict__ X,   // [M×K]
    const float* __restrict__ W,   // [K×N]
    const float* __restrict__ A,   // [K×RANK]
    const float* __restrict__ B,   // [RANK×N]
    float*       __restrict__ Y,   // [M×N]
    int M, int N, int K)
{
    // block origin
    int blockRow = blockIdx.y * TILE_M;
    int blockCol = blockIdx.x * TILE_N;

    // thread-register tile indices
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // base output coordinate for this thread's tile
    int baseRow = blockRow + ty * THREAD_TILE_M;
    int baseCol = blockCol + tx * THREAD_TILE_N;

    // double-buffered shared memory
    __shared__ float sX[2][TILE_M][TILE_K];
    __shared__ float sW[2][TILE_K][TILE_N];
    __shared__ float sA[2][TILE_K][RANK];
    __shared__ float sB[2][RANK][TILE_N];

    // register accumulators for each (sub-row, sub-col)
    float accWX   [THREAD_TILE_M][THREAD_TILE_N] = {0};
    float accLoRA [THREAD_TILE_M][THREAD_TILE_N] = {0};

    // which buffer we are reading from
    int buf = 0;

    // prefetch first tile into buffer 0
    int k0 = 0;
    for (int m = 0; m < THREAD_TILE_M; ++m) {
        int r = baseRow + m;
        for (int k = 0; k < TILE_K; ++k) {
            int src = (r < M && (k0 + k) < K) ? X[r*K + k0 + k] : 0.f;
            sX[buf][ty*THREAD_TILE_M + m][k] = src;
        }
    }
    for (int k = 0; k < TILE_K; ++k) {
        for (int n = 0; n < THREAD_TILE_N; ++n) {
            int c = baseCol + n;
            int srcW = ((k0 + k) < K && c < N) ? W[(k0 + k)*N + c] : 0.f;
            sW[buf][k][tx*THREAD_TILE_N + n] = srcW;
        }
        for (int r = 0; r < RANK; ++r) {
            sA[buf][k][r] = ((k0 + k) < K) ? A[(k0 + k)*RANK + r] : 0.f;
        }
    }
    for (int r = 0; r < RANK; ++r) {
        for (int n = 0; n < THREAD_TILE_N; ++n) {
            int c = baseCol + n;
            sB[buf][r][tx*THREAD_TILE_N + n] = (c < N) ? B[r*N + c] : 0.f;
        }
    }
    __syncthreads();

    // loop over K in double-buffered fashion
    for (k0 = 0; k0 < K; k0 += TILE_K) {
        int nextBuf = buf ^ 1;
        int kNext  = k0 + TILE_K;
        // prefetch next tile into nextBuf (if in bounds)
        if (kNext < K) {
            for (int m = 0; m < THREAD_TILE_M; ++m) {
                int r = baseRow + m;
                for (int k = 0; k < TILE_K; ++k) {
                    int src = (r < M && (kNext + k) < K)
                            ? X[r*K + kNext + k] : 0.f;
                    sX[nextBuf][ty*THREAD_TILE_M + m][k] = src;
                }
            }
            for (int k = 0; k < TILE_K; ++k) {
                for (int n = 0; n < THREAD_TILE_N; ++n) {
                    int c = baseCol + n;
                    int srcW = ((kNext + k) < K && c < N)
                             ? W[(kNext + k)*N + c] : 0.f;
                    sW[nextBuf][k][tx*THREAD_TILE_N + n] = srcW;
                }
                for (int r = 0; r < RANK; ++r) {
                    sA[nextBuf][k][r] =
                      ((kNext + k) < K) ? A[(kNext + k)*RANK + r] : 0.f;
                }
            }
            for (int r = 0; r < RANK; ++r) {
                for (int n = 0; n < THREAD_TILE_N; ++n) {
                    int c = baseCol + n;
                    sB[nextBuf][r][tx*THREAD_TILE_N + n] =
                        (c < N) ? B[r*N + c] : 0.f;
                }
            }
        }
        __syncthreads();

        // compute on buffer `buf`
        for (int kk = 0; kk < TILE_K; ++kk) {
            for (int m = 0; m < THREAD_TILE_M; ++m) {
                float xVal = sX[buf][ty*THREAD_TILE_M + m][kk];
                for (int n = 0; n < THREAD_TILE_N; ++n) {
                    float wVal = sW[buf][kk][tx*THREAD_TILE_N + n];
                    accWX[m][n] += xVal * wVal;
                }
                // LoRA small-rank inner product
                for (int r = 0; r < RANK; ++r) {
                    float aVal = sA[buf][kk][r];
                    for (int n = 0; n < THREAD_TILE_N; ++n) {
                        float bVal = sB[buf][r][tx*THREAD_TILE_N + n];
                        accLoRA[m][n] += xVal * aVal * bVal;
                    }
                }
            }
        }
        __syncthreads();
        buf = nextBuf;
    }

    // write results
    for (int m = 0; m < THREAD_TILE_M; ++m) {
        int r = baseRow + m;
        if (r >= M) continue;
        for (int n = 0; n < THREAD_TILE_N; ++n) {
            int c = baseCol + n;
            if (c >= N) continue;
            Y[r*N + c] = accWX[m][n] + accLoRA[m][n];
        }
    }
}

// CPU reference (same as before)
void fused_lora_cpu(
    const float* X, const float* W,
    const float* A, const float* B,
    float*       Y, int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sumWX = 0.f, sumLoRA = 0.f;
            float tmp[RANK] = {0};
            for (int k = 0; k < K; ++k) {
                float xkn = X[m*K + k];
                sumWX += xkn * W[k*N + n];
                for (int r = 0; r < RANK; ++r)
                    tmp[r] += xkn * A[k*RANK + r];
            }
            for (int r = 0; r < RANK; ++r)
                sumLoRA += tmp[r] * B[r*N + n];
            Y[m*N + n] = sumWX + sumLoRA;
        }
    }
}

int main() {
    const int M = 128, K = 64, N = 128;
    printf("Optimized fused LoRA: M=%d K=%d N=%d RANK=%d\n",
           M, K, N, RANK);

    size_t sizeX = size_t(M)*K*sizeof(float);
    size_t sizeW = size_t(K)*N*sizeof(float);
    size_t sizeA = size_t(K)*RANK*sizeof(float);
    size_t sizeB = size_t(RANK)*N*sizeof(float);
    size_t sizeY = size_t(M)*N*sizeof(float);

    float *hX=(float*)malloc(sizeX), *hW=(float*)malloc(sizeW),
          *hA=(float*)malloc(sizeA), *hB=(float*)malloc(sizeB),
          *hY=(float*)malloc(sizeY), *hRef=(float*)malloc(sizeY);

    srand(42);
    auto rnd=[&](){ return (rand()/float(RAND_MAX)-0.5f)*2.0f; };
    for(int i=0;i<M*K;i++)      hX[i]=rnd();
    for(int i=0;i<K*N;i++)      hW[i]=rnd();
    for(int i=0;i<K*RANK;i++)   hA[i]=rnd();
    for(int i=0;i<RANK*N;i++)   hB[i]=rnd();

    float *dX, *dW, *dA, *dB, *dY;
    cudaMalloc(&dX,sizeX); cudaMalloc(&dW,sizeW);
    cudaMalloc(&dA,sizeA); cudaMalloc(&dB,sizeB);
    cudaMalloc(&dY,sizeY);

    cudaMemcpy(dX,hX,sizeX,cudaMemcpyHostToDevice);
    cudaMemcpy(dW,hW,sizeW,cudaMemcpyHostToDevice);
    cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice);

    dim3 block(TILE_N/THREAD_TILE_N, TILE_M/THREAD_TILE_M);
    dim3 grid((N+TILE_N-1)/TILE_N, (M+TILE_M-1)/TILE_M);

    fused_lora_opt_kernel<<<grid,block>>>(dX,dW,dA,dB,dY,M,N,K);
    cudaDeviceSynchronize();

    cudaMemcpy(hY,dY,sizeY,cudaMemcpyDeviceToHost);
    fused_lora_cpu(hX,hW,hA,hB,hRef,M,N,K);

    // verify
    double maxE=0, sumE=0;
    for(int i=0;i<M*N;i++){
        double e=fabs(hY[i]-hRef[i]);
        maxE=fmax(maxE,e);
        sumE+=e;
    }
    printf("max error=%g avg error=%g\n", maxE, sumE/(M*N));

    // cleanup
    cudaFree(dX); cudaFree(dW);
    cudaFree(dA); cudaFree(dB);
    cudaFree(dY);
    free(hX); free(hW);
    free(hA); free(hB);
    free(hY); free(hRef);
    return 0;
}


