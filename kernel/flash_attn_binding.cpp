/*
 * PyBind11 bindings — exposes flash_attn_fwd_launch to Python via
 * torch.utils.cpp_extension.
 */

#include <torch/extension.h>

// Declared in flash_attn.cu
void flash_attn_fwd_launch(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor&       O,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_attn_fwd",
        &flash_attn_fwd_launch,
        "Flash Attention forward pass (CUDA, inference only)",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("O"),
        py::arg("causal") = false
    );
}
