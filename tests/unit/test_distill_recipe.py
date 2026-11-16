"""Recipe + MockTeacher are pure python — fully testable without torch."""

from __future__ import annotations

import math

from modifyllm.distill import recipe
from modifyllm.distill.teacher import MockTeacher

FORBIDDEN = ("gold", "answers", "expected", "labels")


def test_generate_deterministic_and_sized():
    a = recipe.generate(50, seed=42)
    b = recipe.generate(50, seed=42)
    assert a == b
    assert len(a) == 50
    tasks = {x["task"] for x in a}
    assert tasks == {"tool-approval", "retention", "sufficiency"}


def test_payloads_carry_no_supervision():
    for inst in recipe.generate(60, seed=1):
        flat = str(inst["payload"])
        for key in FORBIDDEN:
            assert f"'{key}'" not in flat, f"supervision key {key} leaked into payload"
        assert set(inst["gold"]) == set(inst["payload"]["questions"])


def test_tool_approval_shape():
    inst = next(x for x in recipe.generate(30, seed=3) if x["task"] == "tool-approval")
    impact = inst["payload"]["questions"]["impact"]
    assert impact["type"] == "choice" and len(impact["criteria"]) == 4
    for qid in ("destructive", "sensitive"):
        assert inst["payload"]["questions"][qid]["type"] == "noul"
    assert inst["gold"]["impact"] in inst["payload"]["questions"]["impact"]["criteria"]


def test_split_seeded_disjoint_and_covers():
    data = recipe.generate(100, seed=5)
    tr, fr = recipe.split(data, seed=7)
    tr2, fr2 = recipe.split(data, seed=7)
    assert {x["id"] for x in tr} == {x["id"] for x in tr2}
    assert not ({x["id"] for x in tr} & {x["id"] for x in fr})
    assert len(tr) + len(fr) == len(data)
    assert len(fr) == 20  # 0.2 * 100


def test_mock_teacher_distributions_valid():
    teacher = MockTeacher(seed=9)
    for inst in recipe.generate(20, seed=2):
        probs = teacher.score(inst)
        assert set(probs) == set(inst["payload"]["questions"])
        for qid, dist in probs.items():
            assert math.isclose(sum(dist), 1.0, abs_tol=1e-9)
            assert all(math.isfinite(p) and p >= 0 for p in dist)
            # heuristic gold should dominate (mock is gold-anchored)
            q = inst["payload"]["questions"][qid]
            keys = list(q["criteria"]) if q["type"] == "choice" else [False, True]
            assert max(range(len(dist)), key=lambda i: dist[i]) == keys.index(inst["gold"][qid])
