"""Acceptance ② — statistical validity on the synthetic world.

Planted signal must be detected (power); null world must yield a LOW measured
false-positive rate (we report the rate, we do not pretend it is zero).
"""

from pathlib import Path

from modifyllm.core.run import run_synthetic


def test_planted_signal_detected(tmp_path: Path):
    receipt, _ = run_synthetic(tmp_path, seed=7, planted=True)
    assert receipt.status == "applied"
    assert receipt.verdict == "passed"
    assert receipt.screen is not None
    assert receipt.screen.frozen_eval is not None
    assert receipt.screen.frozen_eval.ci95[0] > 0
    assert receipt.screen.controls.candidate_effect > receipt.screen.controls.max_control_effect


def test_detection_power(tmp_path: Path):
    hits = 0
    n = 20
    for seed in range(n):
        receipt, _ = run_synthetic(tmp_path, seed=100 + seed, planted=True)
        hits += receipt.verdict == "passed"
    assert hits >= 0.9 * n, f"detection power too low: {hits}/{n}"


def test_null_false_positive_rate_measured_and_low(tmp_path: Path):
    """Zero-hypothesis repetitions: count applied receipts; FP rate must stay low."""
    n = 200
    applied = 0
    for seed in range(n):
        receipt, _ = run_synthetic(tmp_path, seed=10_000 + seed, planted=False)
        applied += receipt.status == "applied"
    rate = applied / n
    print(f"\n[stat] null false-positive rate: {rate:.3f} ({applied}/{n})")
    assert rate < 0.08, f"false-positive rate too high: {rate:.3f}"


def test_gate_failures_produce_failed_receipts(tmp_path: Path):
    ppl_receipt, _ = run_synthetic(tmp_path, seed=7, planted=True, ppl_inject=0.5)
    assert ppl_receipt.status == "discarded"
    assert ppl_receipt.verdict == "failed"
    assert "ppl" in ppl_receipt.failure.stage

    reg_receipt, _ = run_synthetic(tmp_path, seed=7, planted=True, regression_inject=2.0)
    assert reg_receipt.status == "discarded"
    assert "regression" in reg_receipt.failure.stage
