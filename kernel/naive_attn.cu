#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__global__ void matmul(int N, int D, const float *d_Q, const float *d_K, float *d_B){
//each thread computes one element in the result matrix

    int i = blockDim.y * blockIdx.y + threadIdx.y; 
    int j = blockDim.x * blockIdx.x + threadIdx.x; 
    float dot = 0.0f;
    if(i < N && j < N){
        for(int k = 0; k < D; k++){
            dot+= d_Q[i * D + k] * d_K[j * D + k];
        }
        d_B[i * N + j] = dot;
    }
}

__global__ void matmul2(int N, int D, const float *d_B,
     const float *d_V, float *O){
    //each thread computes one element in the result matrix
    
        int i = blockDim.y * blockIdx.y + threadIdx.y; 
        int j = blockDim.x * blockIdx.x + threadIdx.x; 
        float dot = 0.0f;
        if(i < N && j < D){
            for(int k = 0; k < N; k++){
                dot+= d_B[i * N + k] * d_V[k * D + j];
            }
            O[i * D + j] = dot;
        }
    }
    

__global__ void scale(int N, int D, float *d_B, bool causal){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N && j < N){
        d_B[i * N + j] = d_B[i * N + j] * (1.0f / sqrtf((float)D));
        if(causal && i < j) d_B[i * N + j] = -INFINITY;
        
    }
}

__global__ void softmax(int N, float *d_B){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= N) return;

    float row_max = -INFINITY;
    for(int i = 0; i < N; i++){
        if (d_B[j * N + i] > row_max) row_max = d_B[j*N+i];
    }
    const int row_length = 4096;
    float exp_score[row_length];
    for(int i = 0; i < N; i++){
        exp_score[i] = expf(d_B[j*N+i] - row_max);
    }

    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += exp_score[i];
    }

    for(int i = 0; i < N; i++){
        d_B[j * N + i] = exp_score[i] / sum;
    }
}

__host__ void attentionLaunchDevice(
    const float *d_Q, const float *d_K, const float *d_V,
    float *d_O, int N, int D, bool causal){

    int sizeNN = N * N * sizeof(float);

    float *d_B;
    cudaMalloc(&d_B, sizeNN);
    
    const int TILE = 16;
    const int grid_n = (N + TILE - 1) / TILE;
    const int grid_d = (D + TILE - 1) / TILE;
    dim3 dimGridNN(grid_n, grid_n);
    dim3 dimGridND(grid_d, grid_n);
    dim3 dimBlock(TILE, TILE);
    

    matmul<<<dimGridNN, dimBlock>>>(N, D, d_Q, d_K, d_B);
    scale<<<dimGridNN, dimBlock>>>(N, D, d_B, causal);
    softmax<<<grid_n, TILE>>>(N, d_B);
    matmul2<<<dimGridND, dimBlock>>>(N, D, d_B, d_V, d_O);

    cudaFree(d_B);
}

__host__ void attentionLaunch(const float *h_Q, const float *h_K, const float *h_V, 
    float *h_O, int N, int D, bool causal){

    int sizeND = N * D * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, sizeND);
    cudaMalloc(&d_K, sizeND);
    cudaMalloc(&d_V, sizeND);
    cudaMalloc(&d_O, sizeND);

    cudaMemcpy(d_Q, h_Q, sizeND, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sizeND, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeND, cudaMemcpyHostToDevice);

    attentionLaunchDevice(d_Q, d_K, d_V, d_O, N, D, causal);

    cudaMemcpy(h_O, d_O, sizeND, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

#ifdef NAIVE_ATTN_STANDALONE
int main() {
    // A (2x2): [1 2; 3 4],  B (2x2): [5 6; 7 8]
    // Expected C = A @ B: [19 22; 43 50]
    const int N = 2;
    const int D = 2;
    float h_Q[N*D] = {1,1,1,1};
    float h_K[N*D] = {1,1,1,1};
    float h_V[N*D] = {1,0,0,1};
    float h_O[N*D];
    const float expected[N * N] = {0.5, 0.5, 0.5, 0.5};

    attentionLaunch(h_Q, h_K, h_V, h_O, N, D, false);

    printf("O =\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            printf("%f ", h_O[i * D + j]);
        }
    printf("\n");
    }

    int pass = 1;
    for (int i = 0; i < N * N; ++i) {
        if (abs(h_O[i] - expected[i]) > 1e-4) {
            pass = 0;
            break;
        }
    }

    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
#endif
