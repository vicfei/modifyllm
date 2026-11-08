"""Direction extraction: sign convention (u = unwanted − wanted) and recovery quality."""

import numpy as np

from modifyllm.adapters.synthetic import SyntheticAdapter
from modifyllm.core.probe import cosine, direction_from_pair_deltas


def test_sign_convention_unwanted_minus_wanted():
    """wanted = 0, unwanted = e1 ⇒ u must be exactly +e1 (the ablated direction first)."""
    e1 = np.zeros(8)
    e1[0] = 1.0
    u = direction_from_pair_deltas([e1 - np.zeros(8)] * 4)
    assert np.allclose(u, e1, atol=1e-12)


def test_planted_direction_recovered():
    ad = SyntheticAdapter(seed=7, planted=True)
    rng = np.random.default_rng(0)
    deltas = ad.collect_pair_hiddens(32, rng)
    u = direction_from_pair_deltas(deltas)
    assert cosine(u, ad.u_star) > 0.9


def test_null_direction_is_random_and_degenerate_guard():
    ad = SyntheticAdapter(seed=11, planted=False)
    rng = np.random.default_rng(0)
    u1 = direction_from_pair_deltas(ad.collect_pair_hiddens(16, rng))
    u2 = direction_from_pair_deltas(ad.collect_pair_hiddens(16, rng))
    assert abs(cosine(u1, u2)) < 0.5  # no shared structure
    try:
        direction_from_pair_deltas([np.zeros(4)] * 3)
        raise AssertionError("degenerate input must raise")
    except ValueError:
        pass
