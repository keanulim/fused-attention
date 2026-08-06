# Fused Attention

From-scratch CUDA attention kernels with PyTorch integration, correctness tests, and GPU benchmarks on Modal.

**Two implementations:**

```
Naive:  Q @ K^T  →  scale (+ causal mask)  →  softmax  →  P @ V  →  O   (multi-kernel, materializes N×N)
Flash:  tiled Q/K/V loads  →  online softmax  →  fused output accumulate  →  O   (single kernel, no N×N)
```

The naive path is intentionally simple — each step is a separate kernel and the full attention matrix lives in global memory. The flash path fuses the forward pass with shared-memory tiling and online softmax (FlashAttention-style), avoiding the O(N²) memory bottleneck.

## Project layout

```
kernel/naive_attn.cu          Multi-kernel naive attention + launch orchestration
kernel/flash_attn.cu          Fused flash attention forward kernel (F32, D=64)
kernel/*_binding.cpp          PyBind11 bindings for PyTorch
ops/attention.py              cuda_naive_attention, flash_attention, naive_attention (reference)
tests/test_correctness.py     pytest — kernel vs PyTorch reference
benchmark.py                  Timing harness (naive CUDA, flash, PyTorch, SDPA)
modal_app.py                  GPU tests + benchmarks on Modal (A100)
```

## Correctness

Verified against a float32 PyTorch reference (`einsum` + `softmax`) for bidirectional and causal attention.

```bash
modal run modal_app.py::run_tests
```

- **22/22 tests passing** on Modal A100
- Naive CUDA kernel: `max_err < 1e-3` for shapes up to `(B=2, H=2, N=32, D=16)`
- Flash kernel: `max_err < 1e-3` for `(B=2, H=4, N=64, D=64)` including partial tiles (`N=17`) and causal masking

## Benchmarks

### Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (Modal) |
| CUDA | 12.1 / 12.4 |
| PyTorch | 2.3.1 |
| Config | `batch=1, heads=8, head_dim=64, causal=False` |
| Method | 5 warmup iterations, 20 timed iterations, `torch.cuda.synchronize()` |

### Forward pass latency (ms, lower is better)

| Seq len | Flash (mine) | CUDA naive (mine) | PyTorch naive | SDPA F32 | SDPA F16 | N² matrix (MB) | Peak GB |
|--------:|-------------:|------------------:|--------------:|---------:|---------:|---------------:|--------:|
| 256 | 0.21 | 1.06 | 0.15 | 0.06 | **0.03** | 2 | 0.01 |
| 512 | 0.50 | 1.97 | 0.20 | 0.10 | **0.03** | 8 | 0.02 |
| 1024 | 1.12 | 39.0 | 0.46 | 0.26 | **0.04** | 34 | 0.03 |
| 2048 | 2.87 | 49.3 | 1.46 | 0.53 | **0.08** | 134 | 0.05 |
| 4096 | 9.58 | 94.7 | 4.97 | 1.82 | **0.23** | 537 | 0.08 |

**SDPA F16** uses PyTorch's `scaled_dot_product_attention`, which routes to a fused FlashAttention-style kernel on Ampere+ GPUs.

### Takeaways

1. **Correctness first** — flash output matches PyTorch reference (`max_err ≈ 1e-6` across benchmark shapes).
2. **Fusion works** — flash avoids materializing the N×N matrix. Peak GPU memory stays under 0.1 GB vs 1.2 GB for PyTorch naive at `N=4096`.
3. **Naive CUDA degrades at long seq** — latency jumps from ~2 ms at `N=512` to ~39 ms at `N=1024` as the attention matrix grows and per-head kernel launch overhead adds up.
4. **Flash vs naive CUDA** — fused kernel is **~10–35× faster** at long sequences (e.g. 95 ms → 9.6 ms at `N=4096`).
5. **Gap to production SDPA** — v1 flash is still ~2× slower than PyTorch einsum and ~41× slower than SDPA F16 at `N=4096`. Next optimizations: cooperative block GEMM tiling, vectorized loads, FP16 tensor cores.

### Reproduce

```bash
# single shape
modal run modal_app.py::run_benchmark --seq-len 1024

# full sweep (runs tests first)
modal run modal_app.py::sweep
```

No local GPU required — benchmarks run on Modal.

## Commands

```bash
modal run modal_app.py::run_tests       # pytest on GPU
modal run modal_app.py::test_kernel     # standalone C++ sanity check
modal run modal_app.py::run_benchmark   # single benchmark
modal run modal_app.py::sweep           # benchmark sweep
modal run modal_app.py::health_check    # GPU smoke test
```

## Next steps

- [ ] Cooperative block GEMM tiling (warps collaborate on QKᵀ and PV instead of 1 thread per query row)
- [ ] Vectorized memory access (`float4` loads) and swizzled shared memory
- [ ] FP16 / BF16 with tensor cores (WMMA or CUTLASS)
- [ ] Causal tile skipping and load/compute pipelining (`cp.async`)
- [ ] Autotune tile sizes (Br, Bc) per head dim
