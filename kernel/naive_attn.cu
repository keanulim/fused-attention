#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul(const float *A, const float *B, float *C, int N, int D){
//each thread computes one element in the result matrix

    int i = blockDim.y * blockIdx.y + threadIdx.y; 
    int j = blockDim.x * blockIdx.x + threadIdx.x; 
    float dot = 0.0f;
    if(i < N && j < N){
        for(int k = 0; k < D; k++){
            dot+= A[i * D + k] * B[k * N + j];
        }
        C[i * N + j] = dot;
    }
}

__global__ void scale(const float *A, float *B, int N, int D){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N*N){
        B[i] = A[i] * 1 / sqrtf((float)D);
    }
}

__global__ void softmax(const float *S, float *P, int N){
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    float row_max = 0.0f;
    for(int i = 0; i < N; i++){
        if (S[j * N + i] > row_max) row_max = S[i];
    }

    float *exp_score;
    for(int i = 0; i < N; i++){
        exp_score[i] = exp(S[i] - row_max)
    }

    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += exp_score[i];
    }

    for(int i = 0; i < N; i++){
        P[j * N + i] = exp_score[i] / sum;
    }
}



__host__ void matmulLaunch(int N, int D, const float *h_A, const float *h_B, float *h_C){
    
    float *d_A, *d_B, *d_C;

    int size_ND = N * D * sizeof(float);
    int size_NN = N * N * sizeof(float);

    cudaMalloc(&d_A, size_ND);
    cudaMalloc(&d_B, size_ND);
    cudaMalloc(&d_C, size_NN);

    cudaMemcpy(d_A, h_A, size_ND, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_ND, cudaMemcpyHostToDevice);

    dim3 dimGrid (1,1,1);
    dim3 dimBlock(N,N,1);

    matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, D);

    cudaMemcpy(h_C, d_C, size_NN, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // A (2x2): [1 2; 3 4],  B (2x2): [5 6; 7 8]
    // Expected C = A @ B: [19 22; 43 50]
    const int N = 2;
    const int D = 2;

    float h_A[N * D] = {1.f, 2.f, 3.f, 4.f};
    float h_B[N * D] = {5.f, 6.f, 7.f, 8.f};
    float h_C[N * N] = {0.f};
    const float expected[N * N] = {19.f, 22.f, 43.f, 50.f};

    matmulLaunch(N, D, h_A, h_B, h_C);

    printf("C =\n");
    for (int i = 0; i < N; ++i) {
        printf("  [");
        for (int j = 0; j < N; ++j) {
            printf("%6.1f%s", h_C[i * N + j], j + 1 < N ? ", " : "");
        }
        printf("]\n");
    }

    int pass = 1;
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != expected[i]) {
            pass = 0;
            break;
        }
    }

    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
