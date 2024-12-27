#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CALL(func) \
    { \
        cudaError_t err = (func); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(err); \
        } \
    }

#define CUBLAS_CALL(func) \
    { \
        cublasStatus_t err = (func); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __LINE__ << std::endl; \
            exit(err); \
        } \
    }

// Naive convolution kernel
__global__ void conv2d_naive(float* input, float* kernel, float* output, int C, int H, int W, int F, int kH, int kW, int stride, int H_out, int W_out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= F * H_out * W_out) return;

    int f = n / (H_out * W_out);
    int h = (n % (H_out * W_out)) / W_out;
    int w = n % W_out;

    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int h_in = h * stride + kh;
                int w_in = w * stride + kw;
                sum += input[c * H * W + h_in * W + w_in] * kernel[f * C * kH * kW + c * kH * kW + kh * kW + kw];
            }
        }
    }
    output[f * H_out * W_out + h * W_out + w] = sum;
}

// im2col kernel
__global__ void im2col(float* input, float* col, int C, int H, int W, int kH, int kW, int stride, int H_out, int W_out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= C * kH * kW * H_out * W_out) return;

    int c = n / (kH * kW * H_out * W_out);
    int kh = (n % (kH * kW * H_out * W_out)) / (H_out * W_out);
    int kw = (n % (H_out * W_out)) / W_out;
    int h_out = (n % (H_out * W_out)) / W_out;
    int w_out = n % W_out;

    int h_in = h_out * stride + kh;
    int w_in = w_out * stride + kw;

    col[n] = input[c * H * W + h_in * W + w_in];
}

int main() {
    // Increased input dimensions
    const int C = 128, H = 256, W = 256;        // Input channels, height, width
    const int F = 128, kH = 3, kW = 3;         // Filters, kernel height, kernel width
    const int stride = 1;
    const int H_out = (H - kH) / stride + 1;
    const int W_out = (W - kW) / stride + 1;

    // Allocate and initialize input, kernel, and output
    int input_size = C * H * W;
    int kernel_size = F * C * kH * kW;
    int output_size = F * H_out * W_out;
    int col_size = C * kH * kW * H_out * W_out;

    float* h_input = new float[input_size];
    float* h_kernel = new float[kernel_size];
    float* h_output_naive = new float[output_size];
    float* h_output_im2col = new float[output_size];

    // Initialize input and kernel with random values
    for (int i = 0; i < input_size; i++) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernel_size; i++) h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_input, *d_kernel, *d_output_naive, *d_output_im2col, *d_col;
    CUDA_CALL(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_output_naive, output_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_output_im2col, output_size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_col, col_size * sizeof(float)));

    // Copy data to device
    CUDA_CALL(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // Measure naive conv2d
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));
    conv2d_naive<<<(output_size + 255) / 256, 256>>>(d_input, d_kernel, d_output_naive, C, H, W, F, kH, kW, stride, H_out, W_out);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float naive_time;
    CUDA_CALL(cudaEventElapsedTime(&naive_time, start, stop));
    std::cout << "Naive conv2d time: " << naive_time << " ms" << std::endl;

    // Measure im2col + GEMM
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    CUDA_CALL(cudaEventRecord(start));
    im2col<<<(col_size + 255) / 256, 256>>>(d_input, d_col, C, H, W, kH, kW, stride, H_out, W_out);
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H_out * W_out, F, C * kH * kW, &alpha, d_col, H_out * W_out, d_kernel, C * kH * kW, &beta, d_output_im2col, H_out * W_out));
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float im2col_time;
    CUDA_CALL(cudaEventElapsedTime(&im2col_time, start, stop));
    std::cout << "im2col + GEMM time: " << im2col_time << " ms" << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output_naive;
    delete[] h_output_im2col;
    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_kernel));
    CUDA_CALL(cudaFree(d_output_naive));
    CUDA_CALL(cudaFree(d_output_im2col));
    CUDA_CALL(cudaFree(d_col));
    CUBLAS_CALL(cublasDestroy(handle));

    return 0;
}
