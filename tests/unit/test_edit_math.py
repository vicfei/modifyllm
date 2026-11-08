"""Acceptance ① — math determinism: shapes, projection identity, ±α sign, whole-matrix semantics."""

import numpy as np
import pytest

from modifyllm.core.edit import assert_unit, layer_write_component, rank1_edit

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def _world(seed=0, d_model=16, d_ff=24):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_model, d_ff))
    h = rng.standard_normal(d_ff)
    u = rng.standard_normal(d_model)
    u /= np.linalg.norm(u)
    return W, h, u


def test_identity_u_t_wh_prime_equals_one_minus_alpha():
    """The pinning identity: uᵀ(W'h) == (1 − α)·(uᵀWh) for every α in the grid."""
    for alpha in ALPHAS:
        W, h, u = _world(seed=hash(alpha) % 2**31)
        Wp = rank1_edit(W, u, alpha)
        before = layer_write_component(W, h, u)
        after = layer_write_component(Wp, h, u)
        assert after == pytest.approx((1 - alpha) * before, abs=1e-10), f"identity broken at α={alpha}"


def test_alpha_one_is_exact_projection_removal():
    W, h, u = _world(seed=1)
    Wp = rank1_edit(W, u, 1.0)
    assert layer_write_component(Wp, h, u) == pytest.approx(0.0, abs=1e-10)
    assert np.linalg.norm(Wp.T @ u) == pytest.approx(0.0, abs=1e-10)


def test_ablation_magnitude_identity_and_monotone_beyond_one():
    """|uᵀ(W′h)| == |1−α|·|uᵀ(Wh)| exactly; beyond α=1 the ablation deepens monotonically."""
    W, h, u = _world(seed=2)
    before = abs(layer_write_component(W, h, u))
    for alpha in [0.5, 1.0, 2.0, 4.0, 8.0]:
        after = abs(layer_write_component(rank1_edit(W, u, alpha), h, u))
        assert after == pytest.approx(abs(1 - alpha) * before, abs=1e-10)
    comps = [abs(layer_write_component(rank1_edit(W, u, a), h, u)) for a in [1.0, 2.0, 4.0, 8.0]]
    for smaller, larger in zip(comps, comps[1:]):
        assert larger > smaller
    with pytest.raises(ValueError):
        rank1_edit(W, u, -0.1)


def test_whole_matrix_rank1_semantics():
    """W' − W is exactly rank-1 and touches every column (no column subsetting in v0)."""
    W, _, u = _world(seed=3)
    delta = rank1_edit(W, u, 2.0) - W
    assert np.linalg.matrix_rank(delta, tol=1e-9) == 1
    assert all(np.any(delta[:, j] != 0) for j in range(W.shape[1]))


def test_shape_and_norm_guards():
    W, _, u = _world(seed=4)
    with pytest.raises(ValueError):
        rank1_edit(W[:3], u, 1.0)  # not 2-D
    with pytest.raises(ValueError):
        rank1_edit(W, u[:-1], 1.0)  # wrong dim
    with pytest.raises(ValueError):
        rank1_edit(W, u * 2.0, 1.0)  # non-unit
    assert_unit(u)  # the fixture itself is unit
