#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define TILE_M 8
#define TILE_N 64
#define TILE_K 64

// ======== Device helper: accumulate sum of squares per row ========
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

// ======== Device helper: apply per-element gain ========
__device__ void apply_gain(float* tile, const __half* gain) {
    for (int i = 0; i < TILE_M; ++i)
        for (int j = 0; j < TILE_K; ++j)
            tile[i * TILE_K + j] *= __half2float(gain[j]);
}

// ======== Device helper: scale one row by the RMSNorm denominator ========
__device__ void scale_row(float* row, float denom) {
    for (int j = 0; j < TILE_N; ++j)
        row[j] *= denom;
}

// ======== Fused RMSNorm + MatMul kernel ========
__global__ void rmsnorm_matmul_fused(
    const __half* __restrict__ X,   // input matrix X (M?K)
    const __half* __restrict__ G,   // gain vector G (K)
    const __half* __restrict__ W,   // weight matrix W (K?N)
    __half* __restrict__ O,         // output matrix O (M?N)
    int M, int N, int K, float eps)
{
    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    // double-buffer shared memory for X-tiles and W-tiles
    __shared__ float A0[TILE_M * TILE_K], A1[TILE_M * TILE_K];
    __shared__ float B0[TILE_K * TILE_N], B1[TILE_K * TILE_N];

    float* Acur = A0; float* Bcur = B0;
    float* Anext = A1; float* Bnext = B1;
    float C[TILE_M][TILE_N] = {0};   // accumulator for the local output tile
    float v[TILE_M] = {0};           // running sum of squares for RMSNorm

    // preload the first X-tile and W-tile
    int k0 = 0;
    for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
        int row = tile_row + i;
        for (int j = threadIdx.x; j < TILE_K; j += blockDim.x) {
            int col = k0 + j;
            Acur[i * TILE_K + j] = __half2float(X[row * K + col]);
        }
    }
    for (int i = threadIdx.y; i < TILE_K; i += blockDim.y) {
        int row = k0 + i;
        for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
            int col = tile_col + j;
            Bcur[i * TILE_N + j] = __half2float(W[row * N + col]);
        }
    }
    __syncthreads();

    // loop over K in chunks of TILE_K
    for (int k = 0; k < K; k += TILE_K) {
        // if there's another tile ahead, preload it into Anext/Bnext
        if (k + TILE_K < K) {
            int k1 = k + TILE_K;
            for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
                int row = tile_row + i;
                for (int j = threadIdx.x; j < TILE_K; j += blockDim.x) {
                    int col = k1 + j;
                    Anext[i * TILE_K + j] = __half2float(X[row * K + col]);
                }
            }
            for (int i = threadIdx.y; i < TILE_K; i += blockDim.y) {
                int row = k1 + i;
                for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                    int col = tile_col + j;
                    Bnext[i * TILE_N + j] = __half2float(W[row * N + col]);
                }
            }
        }

        __syncthreads();
        // accumulate sum of squares for RMSNorm
        row_reduce_add(v, Acur);
        // apply the gain vector
        apply_gain(Acur, G + k);

        // local matrix multiplication of Acur and Bcur into C
        for (int i = 0; i < TILE_M; ++i) {
            for (int j = 0; j < TILE_N; ++j) {
                float sum = 0.f;
                #pragma unroll
                for (int kk = 0; kk < TILE_K; ++kk)
                    sum += Acur[i * TILE_K + kk] * Bcur[kk * TILE_N + j];
                C[i][j] += sum;
            }
        }

        __syncthreads();
        // swap buffers for next iteration
        //std::swap(Acur, Anext);
        //std::swap(Bcur, Bnext);

        // manually swap the double?buffer pointers
        float* tmpA = Acur;
        Acur      = Anext;
        Anext     = tmpA;

        float* tmpB = Bcur;
        Bcur      = Bnext;
        Bnext     = tmpB;
    }

    // finalize RMSNorm scaling and write back output tile
    for (int i = 0; i < TILE_M; ++i) {
        float denom = rsqrtf(v[i] / float(K) + eps);
        scale_row(C[i], denom);
    }

    for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
        int row = tile_row + i;
        for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
            int col = tile_col + j;
            O[row * N + col] = __float2half(C[i][j]);
        }
    }
}

// ======== CPU reference implementation ========
void rmsnorm_matmul_cpu(const __half* X, const __half* G, const __half* W,
                        __half* O, int M, int N, int K, float eps) {
    for (int m = 0; m < M; ++m) {
        // compute RMSNorm denominator
        float sumsq = 0.f;
        for (int k = 0; k < K; ++k) {
            float v = __half2float(X[m*K + k]);
            sumsq += v * v;
        }
        float denom = 1.f / sqrtf(sumsq / K + eps);

        // perform MatMul with the normalized & gained X
        /*for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                float xn = __half2float(X[m*K + k])
                         * __half2float(G[k]) * denom;
                acc += xn * __half2float(W[k*N + n]);
            }
            O[m*N + n] = __float2half(acc);
        }*/
        // accumulate X*G*W in float, then apply denom once at the end
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                float xg = __half2float(X[m*K + k])
                         * __half2float(G[k]);
                acc += xg * __half2float(W[k*N + n]);
            }
            O[m*N + n] = __float2half(acc * denom);
        }
    }
}

int main() {
    const int M = 16, K = 256, N = 256;
    const float eps = 1e-6f;
    const float tol = 1e-2f;

    size_t size_X = M*K*sizeof(__half);
    size_t size_G = K*sizeof(__half);
    size_t size_W = K*N*sizeof(__half);
    size_t size_O = M*N*sizeof(__half);

    // host buffers
    __half *hX = (__half*)malloc(size_X);
    __half *hG = (__half*)malloc(size_G);
    __half *hW = (__half*)malloc(size_W);
    __half *hO = (__half*)malloc(size_O);
    __half *hO_ref = (__half*)malloc(size_O);

    // random initialization
    srand(42);
    auto rand_half = [](){
        return __float2half((float(rand())/RAND_MAX - 0.5f) * 2.f);
    };
    for (int i = 0; i < M*K; ++i) hX[i] = rand_half();
    for (int i = 0; i < K;   ++i) hG[i] = rand_half();
    for (int i = 0; i < K*N; ++i) hW[i] = rand_half();

    // device buffers
    __half *dX, *dG, *dW, *dO;
    cudaMalloc(&dX, size_X);
    cudaMalloc(&dG, size_G);
    cudaMalloc(&dW, size_W);
    cudaMalloc(&dO, size_O);

    // copy inputs to GPU
    cudaMemcpy(dX, hX, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(dG, hG, size_G, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, size_W, cudaMemcpyHostToDevice);

    // launch fused kernel
    dim3 grid(N/TILE_N, M/TILE_M);
    dim3 block(32, 4);
    rmsnorm_matmul_fused<<<grid, block>>>(dX, dG, dW, dO, M, N, K, eps);
    cudaDeviceSynchronize();

    // retrieve GPU result
    cudaMemcpy(hO, dO, size_O, cudaMemcpyDeviceToHost);

    // compute reference on CPU
    rmsnorm_matmul_cpu(hX, hG, hW, hO_ref, M, N, K, eps);

    // compare errors
    double max_err = 0.0, sum_err = 0.0;
    for (int i = 0; i < M*N; ++i) {
        float a = __half2float(hO[i]);
        float b = __half2float(hO_ref[i]);
        float e = fabsf(a - b);
        max_err = std::max(max_err, double(e));
        sum_err += e;
    }
    double avg_err = sum_err / (M * N);
    bool pass = (max_err < tol);

    // print validation
    printf("Validation: %s\n", pass ? "PASS" : "FAIL");
    printf("Max error = %.6f, Avg error = %.6f, Threshold = %.6f\n",
           max_err, avg_err, tol);

    // cleanup
    cudaFree(dX);  cudaFree(dG);
    cudaFree(dW);  cudaFree(dO);
    free(hX); free(hG);
    free(hW); free(hO);
    free(hO_ref);

    return pass ? 0 : 1;
}
