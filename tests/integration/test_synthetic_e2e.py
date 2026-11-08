"""Acceptance ④ — tiny-model end-to-end on CPU within minutes.

The synthetic world is the tiny model for v0.1 CI; the HF-backed variant is
planned as the v0.2 nightly (skipped here unless torch + network are present).
"""

import time
from pathlib import Path

from modifyllm.core.receipt import load
from modifyllm.core.run import run_synthetic
from modifyllm.core.txn import resolve_deployed, verify_snapshot


def test_end_to_end_under_ten_minutes(tmp_path: Path):
    t0 = time.monotonic()
    receipt, run_dir = run_synthetic(tmp_path, seed=7, planted=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 600, f"e2e took {elapsed:.1f}s (budget: 600s)"

    assert receipt.status == "applied"
    assert receipt.budget is not None and receipt.budget.forward_passes > 0
    assert receipt.data_protocol["frozen_set"]["evaluated_once"] is True
    # reload from disk: receipt round-trips through the schema
    reloaded = load(run_dir)
    assert reloaded.receipt_id == receipt.receipt_id
    assert reloaded.screen is not None and reloaded.screen.frozen_eval is not None
    # deployed snapshot verifies against its manifest
    snap = resolve_deployed(run_dir)
    verify_snapshot(run_dir / snap)


def test_null_run_end_to_end(tmp_path: Path):
    receipt, _ = run_synthetic(tmp_path, seed=99, planted=False)
    assert receipt.status in {"applied", "discarded"}  # both legal; rate tested in stat/
    assert receipt.budget.forward_passes > 0
