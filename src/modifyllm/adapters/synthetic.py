"""Deterministic synthetic world for tests, stats and the CLI demo.

A two-layer linear "model" (W0, W1): o = W1 (W0 h). A planted direction u*
plays the role of an unwanted behavior: the goal grader rewards reducing the
u*-component of outputs. With planted=False the world is null — pairs carry no
signal, and edits must (statistically) fail screening.
"""

from __future__ import annotations

import io

import numpy as np

from modifyllm.core.edit import rank1_edit

LAYERS = ["layers.14.mlp.down_proj", "layers.17.mlp.down_proj"]


def _dump(weights: list[np.ndarray]) -> dict[str, bytes]:
    out = {}
    for i, W in enumerate(weights):
        buf = io.BytesIO()
        np.save(buf, W, allow_pickle=False)
        out[f"layer{i}.npy"] = buf.getvalue()
    return out


def _load(files: dict[str, bytes]) -> list[np.ndarray]:
    names = sorted(files)
    return [np.load(io.BytesIO(files[n]), allow_pickle=False) for n in names]


class SyntheticAdapter:
    def __init__(self, seed: int = 7, planted: bool = True, d_model: int = 32, d_ff: int = 48,
                 ppl_inject: float | None = None, regression_inject: float | None = None):
        self.model_id = f"synthetic-{'planted' if planted else 'null'}-{seed}"
        self.layer_names = list(LAYERS)
        self.planted = planted
        self.ppl_inject = ppl_inject
        self.regression_inject = regression_inject
        rng = np.random.default_rng(seed)
        self.d_model, self.d_ff = d_model, d_ff
        self._rng = rng
        self.W = [rng.standard_normal((d_model, d_ff)) * 0.1 for _ in range(2)]
        u = rng.standard_normal(d_model)
        self.u_star = u / np.linalg.norm(u)  # the unwanted direction
        self._ppl_noise = np.random.default_rng(seed + 1)
        self.forward_passes = 0

    # -- ModelAdapter ------------------------------------------------------
    def source_files(self) -> dict[str, bytes]:
        return _dump(self.W)

    def total_params(self) -> int:
        return sum(W.size for W in self.W)

    def _pair_delta(self, rng: np.random.Generator) -> np.ndarray:
        if self.planted:
            return self.u_star + 0.25 * rng.standard_normal(self.d_model)
        return rng.standard_normal(self.d_model)  # null: no common direction

    def collect_pair_hiddens(self, n_pairs: int, rng: np.random.Generator) -> list[np.ndarray]:
        self.forward_passes += n_pairs  # one forward per side, counted per pair
        return [self._pair_delta(rng) for _ in range(n_pairs)]

    def goal_outputs(self, items: np.ndarray) -> np.ndarray:
        self.forward_passes += len(items)
        return self._goal_with(self.W, items)

    def edited_files(self, u: np.ndarray, alpha: float) -> dict[str, bytes]:
        return _dump([rank1_edit(W, u, alpha) for W in self.W])

    def goal_scores_with_weights(self, files: dict[str, bytes], items: np.ndarray) -> np.ndarray:
        self.forward_passes += len(items)
        return self._goal_with(_load(files), items)

    def measure_ppl_drift(self, files: dict[str, bytes]) -> float:
        n = self._ppl_noise.normal(0, 0.002)
        if self.ppl_inject is not None:
            return float(self.ppl_inject + n)
        return float(abs(n) + 0.004)  # small monotone-ish drift, well under gate

    def measure_regression_pp(self, files: dict[str, bytes]) -> float:
        if self.regression_inject is not None:
            return float(self.regression_inject)
        return float(abs(self._ppl_noise.normal(0, 0.05)))

    # -- internals ----------------------------------------------------------
    def _goal_with(self, weights: list[np.ndarray], items: np.ndarray) -> np.ndarray:
        """Per-item unwanted-component score: lower is better (the 'loop score').

        One-sided signed projection max(0, u*ᵀo): sensitive to the planted
        direction, but neutral to the global output-norm shrinkage that ANY
        rank-1 edit causes (that shrinkage would inflate null false positives).
        """
        items = np.asarray(items, dtype=np.float64)
        o = sum(items @ W.T for W in weights)  # (n, d_ff) → (n, d_model)
        proj = o @ self.u_star
        return np.maximum(proj, 0.0) / np.sqrt(self.d_model)

    def eval_items(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((n, self.d_ff))
        if self.planted:
            # sign-align: every item expresses the unwanted behavior positively,
            # so the one-sided grader sees a real target to remove.
            proj = (sum(base @ W.T for W in self.W)) @ self.u_star
            base[proj < 0] *= -1
            base *= 1.5
        return base
