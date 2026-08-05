"""
PyTorch wrapper around the CUDA flash attention kernel (inference only).

The extension is JIT-compiled on first import via torch.utils.cpp_extension.
Compiled artifacts are cached in /tmp/flash_attn_ext so subsequent imports
inside the same Modal container are fast.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# ---------------------------------------------------------------------------
# Extension loading (JIT compile once per container)
# ---------------------------------------------------------------------------

_KERNEL_DIR = Path(__file__).parent.parent / "kernel"
_EXT: object | None = None
_NAIVE_EXT: object | None = None
_NAIVE_EXT_LOCK = threading.Lock()


def _load_extension() -> object:
    global _EXT
    if _EXT is not None:
        return _EXT

    os.makedirs("/tmp/flash_attn_ext", exist_ok=True)
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


def _load_naive_extension() -> object:
    global _NAIVE_EXT
    if _NAIVE_EXT is not None:
        return _NAIVE_EXT

    with _NAIVE_EXT_LOCK:
        if _NAIVE_EXT is not None:
            return _NAIVE_EXT

        os.makedirs("/tmp/naive_attn_ext", exist_ok=True)
        _NAIVE_EXT = load(
            name="naive_attn_cuda",
            sources=[
                str(_KERNEL_DIR / "naive_attn_binding.cpp"),
                str(_KERNEL_DIR / "naive_attn.cu"),
            ],
            extra_cuda_cflags=[
                "-O2",
                "--use_fast_math",
                "-std=c++17",
            ],
            build_directory="/tmp/naive_attn_ext",
            verbose=True,
        )
        return _NAIVE_EXT


def cuda_naive_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Run the naive multi-kernel CUDA attention implementation.

    Supports [batch, heads, seq_len, head_dim] by looping over batch/heads.
    Uses float32 on CUDA and compares cleanly against naive_attention().

    Note: causal masking is applied in the scale kernel.
    """
    Q = Q.float().contiguous()
    K = K.float().contiguous()
    V = V.float().contiguous()

    if not Q.is_cuda:
        Q = Q.cuda()
        K = K.cuda()
        V = V.cuda()

    B, H, N, D = Q.shape
    O = torch.empty(B, H, N, D, dtype=torch.float32, device=Q.device)
    ext = _load_naive_extension()

    for b in range(B):
        for h in range(H):
            q = Q[b, h]
            k = K[b, h]
            v = V[b, h]
            o = O[b, h]
            ext.naive_attn_fwd(q, k, v, o, causal)

    return O


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Run fused flash attention (inference only).

    Args:
        Q: [batch, heads, seq_len, head_dim]  — float32 for v1 (head_dim=64)
        K: same shape as Q
        V: same shape as Q
        causal: apply causal (autoregressive) mask

    Returns:
        O: [batch, heads, seq_len, head_dim]  float32
    """
    Q = Q.float().contiguous()
    K = K.float().contiguous()
    V = V.float().contiguous()

    if not Q.is_cuda:
        Q = Q.cuda()
        K = K.cuda()
        V = V.cuda()

    if Q.size(-1) != 64:
        raise ValueError("flash_attention v1 supports head_dim=64 only")

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
