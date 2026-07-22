"""
Correctness tests.

Two groups:
  - test_naive_*  : pure PyTorch, no kernel, runs anywhere right now.
  - test_kernel_* : requires the compiled naive CUDA kernel + GPU.

Run locally (CPU fine for naive tests):
    pytest tests/ -v

Run on Modal (kernel tests need GPU):
    modal run modal_app.py::run_tests
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Attempt to import the naive CUDA kernel.
# ---------------------------------------------------------------------------

try:
    from ops.attention import cuda_naive_attention
    _KERNEL_AVAILABLE = True
except Exception:
    _KERNEL_AVAILABLE = False

from ops.attention import naive_attention

requires_kernel = pytest.mark.skipif(
    not _KERNEL_AVAILABLE,
    reason="CUDA naive kernel not yet compiled"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device not available"
)

# ---------------------------------------------------------------------------
# Reference tests — always run, no kernel needed
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 1,   4,   4),   # tiny — easy to debug by eye
    (1, 4, 128,  64),
    (2, 8, 512, 128),
]

KERNEL_SHAPES = [
    (1, 1,   4,   4),
    (1, 2,  16,  16),
    (2, 2,  32,  16),   # requires N*N <= 1024 threads per block
]


@pytest.mark.parametrize("B,H,N,D", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_naive_output_shape(B, H, N, D, causal):
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    O = naive_attention(Q, K, V, causal=causal)
    assert O.shape == Q.shape


def test_naive_causal_mask():
    """Future positions must not influence past output."""
    B, H, N, D = 1, 1, 8, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)

    # Change V at position 3 and check that output at position 2 is unchanged
    V2 = V.clone()
    V2[:, :, 3:, :] += 100.0

    O1 = naive_attention(Q, K, V,  causal=True)
    O2 = naive_attention(Q, K, V2, causal=True)

    torch.testing.assert_close(O1[:, :, :3, :], O2[:, :, :3, :])


def test_naive_softmax_rows_sum_to_one():
    B, H, N, D = 1, 1, 16, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.eye(N).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

    # When V = I, output row i = attention weight row i
    O = naive_attention(Q, K, V, causal=False)
    row_sums = O.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Kernel tests — compare CUDA naive attention vs PyTorch reference
# ---------------------------------------------------------------------------

@requires_kernel
@requires_cuda
@pytest.mark.parametrize("B,H,N,D", KERNEL_SHAPES)
def test_kernel_vs_naive(B, H, N, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device="cuda")
    K = torch.randn(B, H, N, D, device="cuda")
    V = torch.randn(B, H, N, D, device="cuda")

    out_kernel = cuda_naive_attention(Q, K, V, causal=False)
    out_ref = naive_attention(Q, K, V, causal=False)

    torch.testing.assert_close(out_kernel, out_ref, atol=1e-3, rtol=1e-3)
