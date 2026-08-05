/*
 * Flash Attention Forward Pass — YOUR IMPLEMENTATION
 *
 * See instructions in the chat / plan. Keep v1 simple:
 *   F32, D=64 fixed, forward only, Br=Bc=32
 *
 * Implement:
 *   1. flash_attn_fwd_kernel  (__global__)
 *   2. flash_attn_fwd_launch  (host — called from flash_attn_binding.cpp)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

constexpr int kBlockM  = 32;
constexpr int kBlockN  = 32;
constexpr int kHeadDim = 64;

__global__ void flash_attn_fwd_kernel(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int B,
    int H,
    int N,
    int D,
    bool causal
) {
    int j  = blockIdx.y * kBlockM + threadIdx.x; 
    int bh = blockIdx.x;
    int batch = bh / H;
    int head = bh % H;
    //every thread loads in one query row of 64 floats.
    float qRow[kHeadDim];
    if(j < N){
        for(int i = 0; i < kHeadDim; i++){
            qRow[i] = Q[bh * N * kHeadDim + j * kHeadDim + i];
        }
    }
    __shared__ float kTile[kHeadDim * kBlockN];
    __shared__ float vTile[kHeadDim * kBlockN];

    //score (dot q row by 32 k rows)
    float score[kBlockN];
    //tile max
    float m = -INFINITY;
    //softmax denom (sum of exp(score-m))
    float l = 0;
    //output, weighted sum of V rows
    float o[kHeadDim] = {0};

    float mOld = -INFINITY;

    //loop thru kv tiles
    for(int kvTile = 0; kvTile < (N + 32 - 1) / 32; kvTile++){
        int kvStart = kvTile * 32;
        int tileLen = min(32, N - kvStart);

        //load in kv tile
        if(kvStart+threadIdx.x < N){
            for(int kFloat = 0; kFloat < kHeadDim; kFloat++){
                int kGlobal = bh * N * kHeadDim + (kvStart + threadIdx.x) * kHeadDim + kFloat;
                kTile[threadIdx.x * kHeadDim + kFloat] = K[kGlobal];
            }
            for(int vFloat = 0; vFloat < kHeadDim; vFloat++){
                int vGlobal = bh * N * kHeadDim + (kvStart + threadIdx.x) * kHeadDim + vFloat;
                vTile[threadIdx.x * kHeadDim + vFloat] = V[vGlobal];
            }
        }
        __syncthreads();
        //compute
    
        if(j < N){
            mOld = m;
            for(int i = 0; i < tileLen; i++){
                int keyIdx = kvStart + i;
                if (causal && keyIdx > j) {
                    score[i] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for(int k = 0; k < kHeadDim; k++){
                        dot += qRow[k] * kTile[i * kHeadDim + k];
                    }
                    score[i] = dot/sqrtf((float)kHeadDim);
                }
                if(score[i] > m) m = score[i];
            }

            float alpha = expf(mOld - m);
            l *= alpha;
            for (int k = 0; k < kHeadDim; k++) {
                o[k] *= alpha;
            }

            for (int i = 0; i < tileLen; i++) {
                float weight = expf(score[i] - m);
                l += weight;
                for (int k = 0; k < kHeadDim; k++) {
                    o[k] += weight * vTile[i * kHeadDim + k];
                }
            }
        }
        __syncthreads();
    }

    if (j < N) {
        for (int k = 0; k < kHeadDim; k++) {
            O[bh * N * kHeadDim + j * kHeadDim + k] = o[k] / l;
        }
    }
}

void flash_attn_fwd_launch(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O,
    bool causal
) {
   
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(O.scalar_type() == torch::kFloat32, "O must be float32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && O.is_contiguous(),
                "Q, K, V, O must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "expected Q shape [B, H, N, D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");
    TORCH_CHECK(O.sizes() == Q.sizes(), "O must match Q shape");
    TORCH_CHECK(Q.size(3) == kHeadDim, "dim must be 64");
    

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);


    dim3 dimBlock(kBlockM);
    dim3 dimGrid(B * H, (N + kBlockN - 1)/kBlockN);

    

    flash_attn_fwd_kernel<<<dimGrid, dimBlock>>>(Q.data_ptr<float>(),K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(), B, H, N, D, causal);

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "flash_attn_fwd_kernel launch failed: ", cudaGetErrorString(err));

    const cudaError_t sync_err = cudaDeviceSynchronize();
    TORCH_CHECK(sync_err == cudaSuccess,
        "flash_attn_fwd_kernel execution failed: ", cudaGetErrorString(sync_err));
}
