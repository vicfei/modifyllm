"""Torch↔numpy edit-math identity (v0.2): pins the layout convention.

down_proj.weight in torch is (d_model, d_ff) — the same convention as
core.edit.rank1_edit. This test would fail if anyone inserts a transpose.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from modifyllm.core.edit import rank1_edit  # noqa: E402


def test_torch_layout_identity():
    d_model, d_ff = 64, 96
    rng = np.random.default_rng(0)
    W_np = rng.standard_normal((d_model, d_ff))
    u = rng.standard_normal(d_model)
    u /= np.linalg.norm(u)

    for alpha in (0.5, 1.0, 2.0, 8.0):
        W_ed = rank1_edit(W_np, u, alpha)
        W_t = torch.from_numpy(W_np)
        W_ed_t = torch.from_numpy(W_ed)
        # the edited tensor is a valid drop-in: same shape, same dtype family
        assert W_ed_t.shape == W_t.shape

        # identity: u^T (W' h) == (1 - alpha) * u^T (W h) for any h in R^d_ff
        h = torch.from_numpy(rng.standard_normal(d_ff))
        lhs = float(u @ (W_ed_t @ h).numpy())
        rhs = (1.0 - alpha) * float(u @ (W_t @ h).numpy())
        assert np.isclose(lhs, rhs, atol=1e-8), f"identity broken at alpha={alpha}"

        # exact ablation magnitude: ||W'^T u|| == |1 - alpha| * ||W^T u|| (u unit;
        # u lives in the row space, so the product is W.T @ u)
        norm_ed = float(torch.norm(W_ed_t.T @ torch.from_numpy(u)))
        norm_src = float(torch.norm(W_t.T @ torch.from_numpy(u)))
        assert np.isclose(norm_ed, abs(1.0 - alpha) * norm_src, atol=1e-8), (
            f"ablation magnitude broken at alpha={alpha}"
        )


def test_patch_roundtrip_restores_source():
    """Adapter-style in-place copy_ patching restores bit-exact source values."""
    d_model, d_ff = 16, 24
    W = torch.randn(d_model, d_ff)
    orig = W.clone()
    edited = torch.from_numpy(rank1_edit(W.numpy(), np.eye(d_model)[0], 1.5))
    with torch.no_grad():
        W.copy_(edited)
        assert not torch.equal(W, orig)
        W.copy_(orig)
    assert torch.equal(W, orig)
