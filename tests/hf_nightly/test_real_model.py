"""Real-model nightly (v0.2): Qwen/Qwen3.5-0.8B at a pinned revision.

Machinery smoke — the run may legitimately end `discarded` ("no effect" is an
allowed conclusion per DESIGN §8); what must hold is that every path yields a
valid, honest receipt. Select with: pytest -m hf_nightly
"""

from __future__ import annotations

import math
import os

import pytest

pytestmark = pytest.mark.hf_nightly


def test_real_model_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from pathlib import Path

    from modifyllm.core.run import run_hf

    # writes under ./runs so the nightly workflow can archive the receipts
    receipt, run_dir = run_hf(
        Path("runs"),
        model_id=os.environ.get("MODIFYLLM_HF_MODEL", "Qwen/Qwen3.5-0.8B"),
        revision=os.environ.get(
            "MODIFYLLM_HF_REVISION", "2fc06364715b967f1860aea9cf38778875588b17"
        ),
    )
    assert receipt.status in {"applied", "discarded"}
    assert receipt.env["backend"] == "hf"
    assert receipt.env["revision"].startswith("2fc0636")
    assert receipt.budget.forward_passes > 0
    assert receipt.budget.staging_gb > 0
    if receipt.screen is not None:
        assert math.isfinite(receipt.screen.ppl_gate.value)
        assert math.isfinite(receipt.screen.controls.candidate_effect)
        assert math.isfinite(receipt.screen.regression.value)
    if receipt.screen is not None and receipt.screen.frozen_eval is not None:
        fe = receipt.screen.frozen_eval
        assert fe.ci95[0] <= fe.value <= fe.ci95[1]
    print(
        f"\nnightly: {receipt.model_in.id} → {receipt.status}/{receipt.verdict} "
        f"forwards={receipt.budget.forward_passes} staging={receipt.budget.staging_gb}GB"
    )
