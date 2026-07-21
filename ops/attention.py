"""
PyTorch wrapper around the CUDA flash attention kernel (inference only).

The extension is JIT-compiled on first import via torch.utils.cpp_extension.
Compiled artifacts are cached in /tmp/flash_attn_ext so subsequent imports
inside the same Modal container are fast.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# ---------------------------------------------------------------------------
# Extension loading (JIT compile once per container)
# ---------------------------------------------------------------------------

_KERNEL_DIR = Path(__file__).parent.parent / "kernel"
_EXT: object | None = None


def _load_extension() -> object:
    global _EXT
    if _EXT is not None:
        return _EXT

    _EXT = load(
        name="flash_attn_cuda",
        sources=[
            str(_KERNEL_DIR / "flash_attn_binding.cpp"),
            str(_KERNEL_DIR / "flash_attn.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--use_fast_math",
            "-std=c++17",
        ],
        build_directory="/tmp/flash_attn_ext",
        verbose=True,
    )
    return _EXT


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Run fused flash attention (inference only).

    Args:
        Q: [batch, heads, seq_len, head_dim]  — must be float16, contiguous
        K: same shape as Q
        V: same shape as Q
        causal: apply causal (autoregressive) mask

    Returns:
        O: [batch, heads, seq_len, head_dim]  float16
    """
    assert Q.dtype == torch.float16, "flash_attention requires float16 inputs"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

    O = torch.empty_like(Q)
    ext = _load_extension()
    ext.flash_attn_fwd(Q, K, V, O, causal)
    return O


# ---------------------------------------------------------------------------
# Convenience nn.Module
# ---------------------------------------------------------------------------

class FlashAttention(nn.Module):
    """
    Drop-in multi-head self-attention using the fused CUDA kernel.

    Accepts the same Q, K, V already split into heads:
        Q, K, V: [batch, heads, seq_len, head_dim]
    """

    def __init__(self, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return flash_attention(Q, K, V, causal=self.causal)


# ---------------------------------------------------------------------------
# Reference implementation (F32 naive) — used in correctness tests
# ---------------------------------------------------------------------------

def naive_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Numerically-stable attention in float32 — reference only."""
    Q = Q.float()
    K = K.float()
    V = V.float()
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.einsum("bhnd,bhmd->bhnm", Q, K) * scale   # [B,H,N,N]
    if causal:
        N = Q.shape[2]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", P, V)
