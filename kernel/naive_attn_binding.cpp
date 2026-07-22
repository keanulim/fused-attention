/*
 * PyBind11 bindings for the naive multi-kernel attention pipeline.
 */

#include <torch/extension.h>

void attentionLaunch(
    const float *h_Q,
    const float *h_K,
    const float *h_V,
    float *h_O,
    int N,
    int D
);

void attentionLaunchDevice(
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    int N,
    int D
);

static void naive_attn_launch(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(O.scalar_type() == torch::kFloat32, "O must be float32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && O.is_contiguous(),
                "Q, K, V, O must be contiguous");
    TORCH_CHECK(Q.dim() == 2, "expected Q shape [N, D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");
    TORCH_CHECK(O.sizes() == Q.sizes(), "O must match Q shape");

    const int N = static_cast<int>(Q.size(0));
    const int D = static_cast<int>(Q.size(1));

    attentionLaunchDevice(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N,
        D
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "naive_attn_fwd",
        &naive_attn_launch,
        "Naive CUDA attention forward pass (single head, float32)",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("O")
    );
}
