"""Acceptance ③ — transaction reliability: apply / discard / bypass / crash / rollback."""

import json
import os
import subprocess
import sys
from pathlib import Path

from modifyllm.core.receipt import load
from modifyllm.core.run import run_synthetic
from modifyllm.core.txn import resolve_deployed, scan_interrupted, verify_snapshot


def _run_dir(root: Path, receipt) -> Path:
    return root / receipt.run_id


def test_applied_path_and_rollback(tmp_path: Path):
    receipt, run_dir = run_synthetic(tmp_path, seed=7, planted=True)
    assert receipt.status == "applied"
    out = resolve_deployed(run_dir)
    assert out == receipt.deployed_snapshot
    verify_snapshot(run_dir / out)

    from modifyllm.core.txn import rollback

    event = rollback(run_dir, load(run_dir))
    assert event.parent_receipt_id == receipt.receipt_id
    assert event.status == "rolled_back"
    assert resolve_deployed(run_dir) == "snap_src"
    assert load(run_dir).status == "rolled_back"
    # rollback target integrity: source snapshot hashes still verify
    verify_snapshot(run_dir / "snap_src")
    events = list((run_dir / "events").glob("*.json"))
    assert len(events) == 1
    json.loads(events[0].read_text())


def test_discarded_path(tmp_path: Path):
    receipt, run_dir = run_synthetic(tmp_path, seed=7, planted=True, ppl_inject=0.5)
    assert receipt.status == "discarded"
    assert not (run_dir / "DEPLOYED").exists()


def test_screen_bypass_is_never_pass(tmp_path: Path):
    receipt, _ = run_synthetic(tmp_path, seed=7, planted=True, screen_bypass=True)
    assert receipt.status == "applied"
    assert receipt.verdict == "screen_bypassed"


def test_crash_injection_and_recovery(tmp_path: Path):
    env = dict(os.environ, MODIFYLLM_CRASH_AT="after_stage")
    proc = subprocess.run(
        [sys.executable, "-m", "modifyllm", "edit", "--out", str(tmp_path), "--seed", "7"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 70, "crash injection must hard-exit with 70"

    recovered = scan_interrupted(tmp_path)
    assert len(recovered) == 1
    run_dir = tmp_path / recovered[0]
    assert load(run_dir).status == "interrupted"
