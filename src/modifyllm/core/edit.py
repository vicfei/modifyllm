"""Rank-1 projection edit (DESIGN.md §5.2).

W' = W − α·u(uᵀW), u unit norm, whole-matrix (no column subsetting in v0).
Semantics: for any h, the edited output o' = W'h satisfies
uᵀo' = (1 − α)·(uᵀWh) — i.e. the layer's write along u is scaled by (1 − α).
This identity is pinned by tests/unit/test_edit_math.py.
"""

from __future__ import annotations

import numpy as np


def assert_unit(u: np.ndarray, tol: float = 1e-9) -> None:
    n = float(np.linalg.norm(u))
    if abs(n - 1.0) > tol:
        raise ValueError(f"u must be unit norm (got {n:.6f}); normalize in probe")


def rank1_edit(W: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    """Return W' = W − α·u(uᵀW) as a new array. W: (d_model, d_ff); u: (d_model,)."""
    if W.ndim != 2:
        raise ValueError("W must be 2-D (d_model × d_ff)")
    if u.shape[0] != W.shape[0]:
        raise ValueError(f"u dim {u.shape[0]} != down_proj output dim {W.shape[0]}")
    assert_unit(u)
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    return W - alpha * np.outer(u, u @ W)


def layer_write_component(W: np.ndarray, h: np.ndarray, u: np.ndarray) -> float:
    """uᵀ(Wh): the component of the layer's write along u for pre-activation h."""
    return float(u @ (W @ h))
