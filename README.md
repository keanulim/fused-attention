# Fused Attention

From-scratch CUDA attention kernels with PyTorch integration, correctness tests, and GPU benchmarks on Modal.

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

### Reproduce

```bash
# single shape
modal run modal_app.py::run_benchmark --seq-len 1024

# full sweep (runs tests first)
modal run modal_app.py::sweep
```

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
