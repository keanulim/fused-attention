"""
Modal application — fused attention kernel on GPU.

Usage
-----
# individual kernel testing
modal run modal_app.py::test_kernel

# Run correctness tests
modal run modal_app.py::run_tests

# Run a single benchmark (naive CUDA vs PyTorch vs SDPA)
modal run modal_app.py::run_benchmark --seq-len 1024

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
        "numpy",
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
def test_kernel() -> str:
    """
    Compile and run kernel/naive_attn.cu on GPU.

        modal run modal_app.py::test_kernel
    """
    import subprocess

    kernel_dir = "/root/fused-attention/kernel"
    result = subprocess.run(
        [
            "nvcc",
            "naive_attn.cu",
            "-o",
            "kernel_test",
            "-O2",
            "--gpu-architecture=sm_80",
            "-DNAIVE_ATTN_STANDALONE",
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
        ["./kernel_test"],
        cwd=kernel_dir,
        capture_output=True,
        text=True,
    )
    print(run.stdout)
    if run.returncode != 0:
        print(run.stderr)
        raise RuntimeError("kernel test failed")
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
    from ops.attention import _load_extension, _load_naive_extension
    _load_naive_extension()
    _load_extension()
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
    seq_len: int = 1024,
    batch: int = 1,
    heads: int = 8,
    head_dim: int = 64,
    causal: bool = False,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """
    Time cuda_naive_attention vs PyTorch naive vs SDPA (F32 and F16).

        modal run modal_app.py::run_benchmark --seq-len 1024
        modal run modal_app.py::run_benchmark --seq-len 512 --causal
    """
    _setup()
    from benchmark import format_benchmark_row, print_benchmark_header, run_attention_benchmark

    result = run_attention_benchmark(
        seq_len=seq_len,
        batch=batch,
        heads=heads,
        head_dim=head_dim,
        causal=causal,
        warmup=warmup,
        iters=iters,
    )
    print(result)
    print_benchmark_header()
    print(format_benchmark_row(result))
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
    Q = torch.randn(B, H, seq_len, D, dtype=torch.float32, device="cuda")
    K = torch.randn(B, H, seq_len, D, dtype=torch.float32, device="cuda")
    V = torch.randn(B, H, seq_len, D, dtype=torch.float32, device="cuda")

    out_fused = flash_attention(Q, K, V, causal=causal)
    out_ref   = naive_attention(Q, K, V, causal=causal)

    max_err = (out_fused - out_ref).abs().max().item()
    mean_err = (out_fused - out_ref).abs().mean().item()
    ok = max_err < 1e-3

    print(f"max_err={max_err:.6f}  mean_err={mean_err:.6f}  pass={ok}")
    return ok


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates a benchmark sweep
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep():
    from benchmark import format_benchmark_row, print_benchmark_header

    seq_lens = [256, 512, 1024, 2048, 4096]

    print("Running correctness tests...")
    tests_out = run_tests.remote()
    if "failed" in tests_out.lower():
        raise RuntimeError("Tests failed — fix correctness before benchmarking")

    print_benchmark_header()

    results = []
    for seq_len in seq_lens:
        r = run_benchmark.remote(seq_len=seq_len, warmup=5, iters=20)
        results.append(r)
        print(format_benchmark_row(r))

    print("\nDone. cuda_naive_ms / sdpa_f16 > 1 means SDPA is faster.")
    return results
