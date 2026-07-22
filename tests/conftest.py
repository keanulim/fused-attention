"""Shared pytest fixtures for the fused-attention test suite."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _compile_naive_kernel_once():
    """JIT-compile the CUDA extension before any kernel test runs."""
    try:
        from ops.attention import _load_naive_extension
        _load_naive_extension()
    except Exception:
        pass
