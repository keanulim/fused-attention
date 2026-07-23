"""
Benchmark naive CUDA attention against PyTorch reference and SDPA.

Designed to run inside the Modal GPU container (see modal_app.py::run_benchmark).
All naive paths use float32; SDPA is timed in both F32 (fair compare) and F16 (typical prod).
"""

from __future__ import annotations

from typing import Any, Callable


def _time_cuda_fn(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iters: int = 50,
    **kwargs: Any,
) -> tuple[float, float]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak_gb


def _maybe_time(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iters: int = 50,
    **kwargs: Any,
) -> dict[str, float | str | None]:
    import torch

    try:
        ms, peak_gb = _time_cuda_fn(fn, *args, warmup=warmup, iters=iters, **kwargs)
        return {"ms": ms, "peak_gb": peak_gb, "error": None}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": "OOM"}
    except Exception as exc:  # pragma: no cover - surfaced in Modal logs
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": str(exc)}


def run_attention_benchmark(
    seq_len: int = 1024,
    batch: int = 1,
    heads: int = 8,
    head_dim: int = 64,
    causal: bool = False,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from ops.attention import _load_naive_extension, cuda_naive_attention, naive_attention

    _load_naive_extension()

    device = "cuda"
    shape = (batch, heads, seq_len, head_dim)

    torch.manual_seed(0)
    Q_f32 = torch.randn(shape, dtype=torch.float32, device=device)
    K_f32 = torch.randn(shape, dtype=torch.float32, device=device)
    V_f32 = torch.randn(shape, dtype=torch.float32, device=device)

    Q_f16 = Q_f32.half()
    K_f16 = K_f32.half()
    V_f16 = V_f32.half()

    def sdpa_f32() -> torch.Tensor:
        return F.scaled_dot_product_attention(Q_f32, K_f32, V_f32, is_causal=causal)

    def sdpa_f16() -> torch.Tensor:
        return F.scaled_dot_product_attention(Q_f16, K_f16, V_f16, is_causal=causal)

    # Correctness spot-check (CUDA naive vs PyTorch naive)
    out_cuda = cuda_naive_attention(Q_f32, K_f32, V_f32, causal=causal)
    out_ref = naive_attention(Q_f32, K_f32, V_f32, causal=causal)
    max_err = (out_cuda - out_ref).abs().max().item()

    timings = {
        "cuda_naive": _maybe_time(
            "cuda_naive",
            cuda_naive_attention,
            Q_f32,
            K_f32,
            V_f32,
            causal=causal,
            warmup=warmup,
            iters=iters,
        ),
        "pytorch_naive": _maybe_time(
            "pytorch_naive",
            naive_attention,
            Q_f32,
            K_f32,
            V_f32,
            causal=causal,
            warmup=warmup,
            iters=iters,
        ),
        "sdpa_f32": _maybe_time(
            "sdpa_f32", sdpa_f32, warmup=warmup, iters=iters
        ),
        "sdpa_f16": _maybe_time(
            "sdpa_f16", sdpa_f16, warmup=warmup, iters=iters
        ),
    }

    elem_bytes = 4  # float32 Q/K/V/O traffic estimate for naive paths
    qkv_bytes = 3 * batch * heads * seq_len * head_dim * elem_bytes
    o_bytes = batch * heads * seq_len * head_dim * elem_bytes
    attn_matrix_bytes = batch * heads * seq_len * seq_len * elem_bytes
    naive_traffic_bytes = qkv_bytes + o_bytes + attn_matrix_bytes

    props = torch.cuda.get_device_properties(0)

    result: dict[str, Any] = {
        "gpu": props.name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "seq_len": seq_len,
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "causal": causal,
        "max_err_cuda_vs_ref": round(max_err, 6),
        "naive_attn_matrix_mb": round(attn_matrix_bytes / 1e6, 1),
        "warmup": warmup,
        "iters": iters,
    }

    for name, stats in timings.items():
        ms = stats["ms"]
        result[f"{name}_ms"] = round(ms, 3) if ms is not None else None
        peak = stats["peak_gb"]
        result[f"{name}_peak_gb"] = round(peak, 3) if peak is not None else None
        result[f"{name}_error"] = stats["error"]
        if ms is not None and ms > 0 and name in ("cuda_naive", "pytorch_naive"):
            bw = naive_traffic_bytes / (ms * 1e-3) / 1e9
            result[f"{name}_bw_GBs"] = round(bw, 1)

    cuda_ms = timings["cuda_naive"]["ms"]
    sdpa_ms = timings["sdpa_f16"]["ms"]
    if cuda_ms and sdpa_ms and cuda_ms > 0:
        result["cuda_vs_sdpa_f16"] = round(cuda_ms / sdpa_ms, 2)
    if timings["pytorch_naive"]["ms"] and cuda_ms:
        result["cuda_vs_pytorch_naive"] = round(
            cuda_ms / timings["pytorch_naive"]["ms"], 2
        )

    return result


def format_benchmark_row(r: dict[str, Any]) -> str:
    return (
        f"{r['seq_len']:>8}  "
        f"{_fmt_ms(r.get('cuda_naive_ms')):>12}  "
        f"{_fmt_ms(r.get('pytorch_naive_ms')):>12}  "
        f"{_fmt_ms(r.get('sdpa_f32_ms')):>10}  "
        f"{_fmt_ms(r.get('sdpa_f16_ms')):>10}  "
        f"{r.get('naive_attn_matrix_mb', 0):>8.0f}  "
        f"{r.get('cuda_naive_peak_gb', 0) or 0:>8.2f}"
    )


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "OOM/err"
    return f"{value:.3f}"


def print_benchmark_header() -> None:
    print(
        f"\n{'seq_len':>8}  "
        f"{'cuda_naive':>12}  "
        f"{'pytorch_naive':>12}  "
        f"{'sdpa_f32':>10}  "
        f"{'sdpa_f16':>10}  "
        f"{'N^2 MB':>8}  "
        f"{'peak GB':>8}"
    )
    print("-" * 88)
    print("Times in ms (lower is better). N^2 MB = attention matrix size per batch (all heads).")
