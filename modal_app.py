"""
Modal application — fused attention kernel on GPU.

Usage
-----
# Test standalone matmul kernel
modal run modal_app.py::test_matmul

# Run correctness tests
modal run modal_app.py::run_tests

# Run a single benchmark
modal run modal_app.py::run_benchmark --seq-len 4096

# Sweep benchmark across sequence lengths
modal run modal_app.py::sweep

# Interactive shell inside the container (useful for debugging)
modal shell modal_app.py
"""

from __future__ import annotations

import time
import modal

# ---------------------------------------------------------------------------
# Image
# Use the official CUDA 12.4 devel image so nvcc is available for JIT
# compilation of the kernel inside the container.
# ---------------------------------------------------------------------------

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.3.1",          # pinned — update to taste
        "ninja",                  # speeds up JIT compilation
        "einops",
        "pytest",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(
        ".",
        remote_path="/root/fused-attention",
        ignore=[".git", "__pycache__", "*.pyc", "build", "/tmp"],
    )
)

app = modal.App("fused-attention", image=cuda_image)

# ---------------------------------------------------------------------------
# Shared container config
# ---------------------------------------------------------------------------

CONTAINER_KWARGS = dict(
    gpu="a100",          # A100-80GB — change to "h100" for FP8 experiments
    timeout=600,
)

# ---------------------------------------------------------------------------
# Helper: set up sys.path inside the container
# ---------------------------------------------------------------------------

def _setup():
    import sys
    sys.path.insert(0, "/root/fused-attention")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@app.function(**CONTAINER_KWARGS)
def test_matmul() -> str:
    """
    Compile and run kernel/naive_attn.cu on GPU.

        modal run modal_app.py::test_matmul
    """
    import subprocess

    kernel_dir = "/root/fused-attention/kernel"
    result = subprocess.run(
        [
            "nvcc",
            "naive_attn.cu",
            "-o",
            "matmul_test",
            "-O2",
            "--gpu-architecture=sm_80",
        ],
        cwd=kernel_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("nvcc compile failed")

    run = subprocess.run(
        ["./matmul_test"],
        cwd=kernel_dir,
        capture_output=True,
        text=True,
    )
    print(run.stdout)
    if run.returncode != 0:
        print(run.stderr)
        raise RuntimeError("matmul test failed")
    return run.stdout


@app.function(**CONTAINER_KWARGS)
def health_check() -> dict:
    """
    Verify GPU is reachable and the reference implementation works.
    No CUDA kernel required — run this before writing flash_attn.cu.

        modal run modal_app.py::health_check
    """
    _setup()
    import torch
    from ops.attention import naive_attention

    assert torch.cuda.is_available(), "no GPU visible"
    props = torch.cuda.get_device_properties(0)

    B, H, N, D = 1, 1, 4, 4
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    O = naive_attention(Q, K, V, causal=True)

    info = {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "naive_attn_output_shape": list(O.shape),
        "naive_attn_row0": O[0, 0, 0].tolist(),
    }
    print(info)
    return info


@app.function(**CONTAINER_KWARGS)
def run_tests():
    """Run the correctness test suite with pytest."""
    _setup()
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd="/root/fused-attention",
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Tests failed")
    return result.stdout


@app.function(**CONTAINER_KWARGS)
def run_benchmark(
    seq_len: int = 2048,
    batch: int = 4,
    heads: int = 16,
    head_dim: int = 128,
    causal: bool = True,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """
    Time the fused kernel against PyTorch's built-in SDPA.
    Returns a dict with latency and bandwidth stats.
    """
    _setup()
    import torch
    from ops.attention import flash_attention, naive_attention

    dtype  = torch.float16
    device = "cuda"
    shape  = (batch, heads, seq_len, head_dim)

    Q = torch.randn(shape, dtype=dtype, device=device)
    K = torch.randn(shape, dtype=dtype, device=device)
    V = torch.randn(shape, dtype=dtype, device=device)

    def time_fn(fn, *args, **kwargs):
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms

    # Fused kernel
    fused_ms = time_fn(flash_attention, Q, K, V, causal)

    # PyTorch SDPA (uses FlashAttention internally on Ampere+)
    sdpa_ms = time_fn(
        torch.nn.functional.scaled_dot_product_attention,
        Q, K, V,
        is_causal=causal,
    )

    # Theoretical memory traffic (bytes read + written, ignoring intermediates)
    elem_bytes  = 2  # float16
    qkv_bytes   = 3 * batch * heads * seq_len * head_dim * elem_bytes
    o_bytes     = batch * heads * seq_len * head_dim * elem_bytes
    total_bytes = qkv_bytes + o_bytes

    fused_bw  = total_bytes / (fused_ms * 1e-3) / 1e9   # GB/s
    sdpa_bw   = total_bytes / (sdpa_ms  * 1e-3) / 1e9

    result = {
        "seq_len":   seq_len,
        "batch":     batch,
        "heads":     heads,
        "head_dim":  head_dim,
        "causal":    causal,
        "fused_ms":  round(fused_ms,  3),
        "sdpa_ms":   round(sdpa_ms,   3),
        "fused_bw_GBs":  round(fused_bw,  1),
        "sdpa_bw_GBs":   round(sdpa_bw,   1),
        "speedup":   round(sdpa_ms / fused_ms, 3),
    }
    print(result)
    return result


@app.function(**CONTAINER_KWARGS)
def correctness_check(
    seq_len: int = 512,
    causal: bool = False,
) -> bool:
    """Quick sanity check: fused output matches naive F32 within tolerance."""
    _setup()
    import torch
    from ops.attention import flash_attention, naive_attention

    B, H, D = 2, 8, 64
    Q = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, seq_len, D, dtype=torch.float16, device="cuda")

    out_fused = flash_attention(Q, K, V, causal=causal)
    out_ref   = naive_attention(Q, K, V, causal=causal).half()

    max_err = (out_fused - out_ref).abs().max().item()
    mean_err = (out_fused - out_ref).abs().mean().item()
    ok = max_err < 0.02  # FP16 tolerance

    print(f"max_err={max_err:.6f}  mean_err={mean_err:.6f}  pass={ok}")
    return ok


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates a benchmark sweep
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep():
    seq_lens = [512, 1024, 2048, 4096, 8192]

    print("Running correctness check...")
    ok = correctness_check.remote(seq_len=512, causal=False)
    if not ok:
        raise RuntimeError("Correctness check failed — fix the kernel before benchmarking")

    print(f"\n{'seq_len':>8}  {'fused_ms':>10}  {'sdpa_ms':>9}  {'speedup':>8}  {'fused_BW':>10}")
    print("-" * 55)

    # Fire all benchmark calls in parallel
    futures = [run_benchmark.spawn(seq_len=s) for s in seq_lens]
    results = [f.get() for f in futures]

    for r in sorted(results, key=lambda x: x["seq_len"]):
        print(
            f"{r['seq_len']:>8}  "
            f"{r['fused_ms']:>10.3f}  "
            f"{r['sdpa_ms']:>9.3f}  "
            f"{r['speedup']:>8.3f}x  "
            f"{r['fused_bw_GBs']:>8.1f} GB/s"
        )
