"""Screening statistics (DESIGN.md §5.4).

- norm-matched random unit controls
- paired sign-flip permutation test
- item-level bootstrap CI of the paired mean difference
All thresholds live in EditSpec; results carry value+unit+threshold+n.
"""

from __future__ import annotations

import numpy as np


def norm_matched_random_units(d: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """n random unit vectors in R^d — same norm (=1) as the candidate direction."""
    return [ (lambda v: v / np.linalg.norm(v))(rng.standard_normal(d)) for _ in range(n) ]


def permutation_sign_test(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Paired sign-flip test. H0: symmetric noise around 0. One-sided (mean > 0)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    observed = float(diffs.mean())
    if observed <= 0 or len(diffs) == 0:
        return 1.0
    flips = rng.integers(0, 2, size=(n_perm, len(diffs))) * 2 - 1
    perm_means = (diffs[None, :] * flips).mean(axis=1)
    p = float((perm_means >= observed).mean())
    return max(p, 1.0 / max(n_perm, 1))


def bootstrap_ci_mean(
    diffs: np.ndarray, B: int, seed: int, alpha: float = 0.05
) -> tuple[float, list[float]]:
    """Item-level bootstrap of the mean paired difference. Returns (mean, [lo, hi])."""
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(B, n))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), [float(lo), float(hi)]


def gate(value: float, threshold: float, ok: bool) -> bool:
    return ok and value <= threshold if threshold is not None else ok
