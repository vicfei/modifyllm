"""Model adapters.

v0.1 ships SyntheticAdapter (numpy, deterministic, CPU-instant).
HFModelAdapter (torch/transformers, lazy imports) arrives with v0.2 nightly
experiments — its interface is fixed here so tests can program against it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ModelAdapter(Protocol):
    """The interface both synthetic and HF adapters implement."""

    model_id: str
    layer_names: list[str]

    def source_files(self) -> dict[str, bytes]:
        """Serialized weights of the pristine model (snapshot material)."""

    def total_params(self) -> int: ...

    def collect_pair_hiddens(self, n_pairs: int, rng: np.random.Generator) -> list[np.ndarray]:
        """n_pairs deltas h_unwanted − h_wanted per target layer block, output space."""

    def goal_outputs(self, items: np.ndarray) -> np.ndarray:
        """Forward pass over evaluation items; returns per-item goal scores (before)."""

    def edited_files(self, u: np.ndarray, alpha: float) -> dict[str, bytes]:
        """Weights after applying the rank-1 edit on all target layers."""

    def goal_scores_with_weights(self, files: dict[str, bytes], items: np.ndarray) -> np.ndarray:
        """Forward pass over items using edited weights (no mutation of the adapter)."""

    def measure_ppl_drift(self, files: dict[str, bytes]) -> float:
        """Relative PPL drift on held-out text, edited vs source (synthetic: modeled)."""

    def measure_regression_pp(self, files: dict[str, bytes]) -> float:
        """Absolute accuracy drop in percentage points on the regression gate set."""
