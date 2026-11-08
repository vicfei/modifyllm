"""Direction extraction (DESIGN.md §5.1).

Convention (sign-critical): u = mean(h_unwanted − h_wanted), L2-normalized.
The direction to ABLATE comes first. Guarded by unit tests (tests/unit/test_probe.py)
and the ±alpha sign test (tests/unit/test_edit_math.py).
"""

from __future__ import annotations

import numpy as np


def direction_from_pair_deltas(deltas: list[np.ndarray]) -> np.ndarray:
    """deltas[i] = h_unwanted_i − h_wanted_i, each of shape (d_model,). Returns unit u."""
    if not deltas:
        raise ValueError("no pair deltas supplied")
    mat = np.asarray(deltas, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("each delta must be a 1-D vector")
    u = mat.mean(axis=0)
    norm = float(np.linalg.norm(u))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("degenerate direction (zero norm)")
    return u / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
