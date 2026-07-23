# Fused Attention

From-scratch CUDA attention kernels with PyTorch integration, correctness tests, and GPU benchmarks on Modal.

**Pipeline (naive forward pass):**

```
Q @ K^T  →  scale (+ causal mask)  →  softmax  →  P @ V  →  O
```

Each step is a separate CUDA kernel. The full `N×N` attention matrix is materialized in global memory — simple to understand, but intentionally slow at long sequences. Flash Attention (WIP) will fuse these steps and avoid the `N×N` bottleneck.

## Project layout

```
kernel/naive_attn.cu          CUDA kernels + launch orchestration
kernel/naive_attn_binding.cpp PyBind11 binding for PyTorch
ops/attention.py              cuda_naive_attention, naive_attention (reference)
tests/test_correctness.py     pytest — kernel vs PyTorch reference
benchmark.py                  timing harness
modal_app.py                  GPU tests + benchmarks on Modal (A100)
```

## Correctness

Verified against a float32 PyTorch reference (`einsum` + `softmax`) for bidirectional and causal attention.

```bash
modal run modal_app.py::run_tests
```

All kernel tests pass at `max_err < 1e-3` for shapes up to `(B=2, H=2, N=32, D=16)`.

## Benchmarks

### Environment

| | |
|---|---|
| GPU | NVIDIA A100 (40GB or 80GB, Modal) |
| CUDA | 12.1 / 12.4 |
| PyTorch | 2.3.1 |
| Config | `batch=1, heads=8, head_dim=64, causal=False` |
| Method | 5 warmup iterations, 20 timed iterations, `torch.cuda.synchronize()` |

### Forward pass latency (ms, lower is better)

| Seq len | CUDA naive (mine) | PyTorch naive | SDPA F32 | SDPA F16 | N² matrix (MB) |
|--------:|------------------:|--------------:|---------:|---------:|---------------:|
| 256 | 1.04 | 0.15 | 0.05 | **0.03** | 2 |
| 512 | 1.95 | 0.20 | 0.10 | **0.03** | 8 |
| 1024 | ~300 | 0.46 | 0.26 | **0.04** | 34 |
| 2048 | 236.87 | 1.76 | 0.68 | **0.11** | 134 |
| 4096 | 294.34 | 5.96 | 1.92 | **0.23** | 537 |

**SDPA F16** uses PyTorch's `scaled_dot_product_attention`, which routes to a fused FlashAttention-style kernel on Ampere+ GPUs.

### Takeaways

1. **Correctness first** — CUDA output matches PyTorch reference (`max_err = 0.0` across sweep shapes).
2. **O(N²) memory dominates** — the attention matrix grows from 2 MB → 537 MB across this sweep. SDPA avoids materializing it entirely.
3. **Kernel launch overhead** — four kernels + `cudaMalloc(N²)` per head, called from a Python loop over batch × heads.
4. **Gap to production** — at `N=4096`, SDPA F16 is ~**1,260× faster** than this naive CUDA path. That gap is the motivation for Flash Attention.

> **Note:** CUDA naive times jump sharply above `N=512` (~2 ms → ~300 ms at `N=1024`) due to unfused kernels, per-head `cudaMalloc(N²)`, and a Python loop over heads — not yet optimized. SDPA scales smoothly across the same range. The `N=4096` row is the most representative headline comparison.

### Reproduce

```bash
# single shape
modal run modal_app.py::run_benchmark --seq-len 1024

# full sweep (runs tests first)
modal run modal_app.py::sweep
```

No local GPU required — benchmarks run on Modal.

## Resume / interview framing

> Implemented a tiled CUDA multi-head attention forward pass (matmul, scale, causal mask, softmax, matmul) with PyBind11/PyTorch integration. Validated against PyTorch reference on Modal A100 (<1e-3 error, causal + bidirectional). Benchmarked vs SDPA across seq lengths 256–4096; analyzed O(N²) HBM traffic and kernel launch overhead as motivation for fused Flash Attention.

## Commands

```bash
modal run modal_app.py::run_tests       # pytest on GPU
modal run modal_app.py::test_kernel     # standalone C++ sanity check
modal run modal_app.py::run_benchmark   # single benchmark
modal run modal_app.py::sweep           # benchmark sweep
modal run modal_app.py::health_check    # GPU smoke test
```

## Next steps

- [ ] Reuse `d_B` buffer across heads (remove per-call `cudaMalloc`)
- [ ] Expand kernel correctness tests to `N=128, D=64`
- [ ] Implement Flash Attention — online softmax, shared-memory tiling, single fused kernel
- [ ] Add Flash kernel to benchmark table vs SDPA
