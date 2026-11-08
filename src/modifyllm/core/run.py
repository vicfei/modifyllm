"""RunEngine: the v0.1 pipeline (DESIGN.md §5, §3).

probe direction → α-candidates on selection set → gates (ppl / controls+permutation /
regression) with escalation → single frozen-test with bootstrap CI → two-phase commit
→ receipt. Every path terminates in a legal receipt state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modifyllm.adapters.synthetic import SyntheticAdapter
from modifyllm.core import receipt as receipt_io
from modifyllm.core import txn
from modifyllm.core.ids import new_id
from modifyllm.core.probe import direction_from_pair_deltas
from modifyllm.core.screen import (
    bootstrap_ci_mean,
    norm_matched_random_units,
    permutation_sign_test,
)
from modifyllm.schema.models import (
    Budget,
    ControlsCheck,
    EditSpec,
    EvalSpec,
    Failure,
    FrozenEvalCheck,
    GoalSpec,
    MetricCheck,
    ModelRef,
    Receipt,
    ScreenReport,
)

LAYERS = ["layers.14.mlp.down_proj", "layers.17.mlp.down_proj"]


def run_synthetic(
    root: Path,
    seed: int = 7,
    planted: bool = True,
    screen_bypass: bool = False,
    ppl_inject: float | None = None,
    regression_inject: float | None = None,
) -> tuple[Receipt, Path]:
    adapter = SyntheticAdapter(
        seed=seed, planted=planted, ppl_inject=ppl_inject, regression_inject=regression_inject
    )
    goal = GoalSpec(
        kind=f"synthetic:{'planted' if planted else 'null'}",
        pairs_ref="synthetic://in-memory",
        grader={"type": "u_star-projection", "version": "0"},
    )
    ev = EvalSpec(direction_set_n=32, selection_set_n=120, frozen_set_n=200)
    spec = EditSpec(layers=list(LAYERS))
    receipt = RunEngine().run(root, adapter, goal, ev, spec, screen_bypass=screen_bypass)
    return receipt, root / receipt.run_id


class _Candidate(tuple):
    """(selection_effect, alpha, files, selection_diffs)."""


class RunEngine:
    def run(
        self,
        root: Path,
        adapter: SyntheticAdapter,
        goal: GoalSpec,
        ev: EvalSpec,
        spec: EditSpec,
        screen_bypass: bool = False,
    ) -> Receipt:
        run_id = new_id("run")
        receipt_id = new_id("rcp")
        run_dir = root / run_id
        txn.acquire_lock(run_dir)
        receipt = Receipt(
            receipt_id=receipt_id,
            run_id=run_id,
            status="running",
            ts_started=txn.utcnow(),
            model_in=ModelRef(id=adapter.model_id, revision="synthetic-v0"),
            rollback_to="snap_src",
            env={"code_version": "0.1.0", "backend": "numpy-synthetic",
                 "seeds": {"data": ev.seed_data, "bootstrap": ev.seed_bootstrap}},
            goal=goal.model_dump(),
            data_protocol={
                "direction_set": {"n": ev.direction_set_n},
                "selection_set": {"n": ev.selection_set_n},
                "frozen_set": {"n": ev.frozen_set_n, "evaluated_once": True},
            },
        )
        receipt_io.write(run_dir, receipt)

        # 1) source snapshot (rollback target, immutable)
        src_files = adapter.source_files()
        txn.write_snapshot(run_dir / "snap_src", src_files)
        receipt.model_in.files_sha256 = txn.files_digest(src_files)

        rng = np.random.default_rng(ev.seed_data)

        # 2) direction extraction — direction set only (DESIGN §5.1)
        deltas = adapter.collect_pair_hiddens(ev.direction_set_n, rng)
        u = direction_from_pair_deltas(deltas)

        # 3) α-candidates — selection set only, never frozen (DESIGN §5.3)
        sel_items = adapter.eval_items(ev.selection_set_n, ev.seed_data + 101)
        before_sel = adapter.goal_outputs(sel_items)
        candidates: list[tuple[float, float, dict[str, bytes], np.ndarray]] = []
        for alpha in spec.alpha_grid:
            files = adapter.edited_files(u, alpha)
            after = adapter.goal_scores_with_weights(files, sel_items)
            diffs = before_sel - after  # positive = improvement (unwanted score dropped)
            candidates.append((float(diffs.mean()), float(alpha), files, diffs))
        candidates.sort(key=lambda c: c[0], reverse=True)
        candidates = candidates[: spec.max_candidates]

        rng_ctrl = np.random.default_rng(ev.seed_data + 202)
        chosen: tuple[float, float, dict[str, bytes], np.ndarray] | None = None
        screen: ScreenReport | None = None
        failure_stage, passed_checks = "", []

        if screen_bypass:
            chosen = candidates[0]
        else:
            chosen, screen, failure_stage, passed_checks = self._screen(
                adapter, spec, ev, rng_ctrl, u, candidates, before_sel, sel_items
            )

        # 4) frozen test — evaluated once, only for a screened winner (DESIGN §5.4.4)
        if chosen is not None and not screen_bypass:
            effect, alpha, files, _ = chosen
            frozen_items = adapter.eval_items(ev.frozen_set_n, ev.seed_data + 303)
            before_f = adapter.goal_outputs(frozen_items)
            after_f = adapter.goal_scores_with_weights(files, frozen_items)
            diffs_f = before_f - after_f
            mean, ci = bootstrap_ci_mean(diffs_f, ev.bootstrap_B, ev.seed_bootstrap)
            ok = ci[0] > 0
            screen.frozen_eval = FrozenEvalCheck(
                value=mean, unit="paired_delta", ci95=ci,
                bootstrap={"unit": "item", "B": ev.bootstrap_B},
                verdict="pass" if ok else "fail",
            )
            if not ok:
                chosen, failure_stage = None, "frozen_eval"

        # 5) two-phase commit or discard; finalize receipt (DESIGN §3)
        receipt.operator = {
            "kind": "edit",
            "spec": {
                "recipe": "rank1",
                "layers": spec.layers,
                "alpha": chosen[1] if chosen else None,
                "u_norm": 1.0,
                "edit_kind": "whole-matrix rank-1",
                "matrices": len(spec.layers),
                "params_touched_pct": round(100.0 * len(spec.layers) / max(1, len(adapter.layer_names)), 3),
            },
        }
        if chosen is not None:
            _, alpha, files, _ = chosen
            txn.stage_candidate(run_dir, files)
            out_name = f"snap_{receipt_id}"
            txn.commit_staged(run_dir, out_name, files)
            txn.deploy(run_dir, out_name)
            receipt.status = "applied"
            receipt.verdict = "passed" if not screen_bypass else "screen_bypassed"
            receipt.deployed_snapshot = out_name
            receipt.model_out = ModelRef(
                id=adapter.model_id, revision=out_name, files_sha256=txn.files_digest(files)
            )
            if screen is not None:
                receipt.screen = screen
        else:
            receipt.status = "discarded"
            receipt.verdict = "failed"
            receipt.failure = Failure(
                stage=failure_stage or "screen",
                error="no candidate passed screening",
                passed_checks=passed_checks,
            )
            if screen is not None:
                receipt.screen = screen

        receipt.budget = Budget(forward_passes=adapter.forward_passes, staging_gb=0.001)
        receipt.ts_finished = txn.utcnow()
        receipt_io.write(run_dir, receipt)
        return receipt

    # -- gates with escalation across α (effect-desc order) -------------------
    def _screen(self, adapter, spec, ev, rng_ctrl, u, candidates, before_sel, sel_items):
        last_screen: ScreenReport | None = None
        failed_stage = "screen"
        for effect, alpha, files, diffs in candidates:
            ppl = adapter.measure_ppl_drift(files)
            ppl_ok = ppl <= spec.ppl_gate_max_relative_drift
            ctrl_effects = []
            for uc in norm_matched_random_units(len(u), spec.controls_n, rng_ctrl):
                cfiles = adapter.edited_files(uc, alpha)
                cafter = adapter.goal_scores_with_weights(cfiles, sel_items)
                ctrl_effects.append(float((before_sel - cafter).mean()))
            p = permutation_sign_test(diffs, ev.permutation_max, rng_ctrl)
            ctrl_ok = effect > max(ctrl_effects) and p < 0.05
            reg = adapter.measure_regression_pp(files)
            reg_ok = reg <= spec.regression_gate_max_pp
            checks = {"ppl": ppl_ok, "controls": ctrl_ok, "regression": reg_ok}
            last_screen = ScreenReport(
                ppl_gate=MetricCheck(value=ppl, unit="relative_drift",
                                     threshold=spec.ppl_gate_max_relative_drift, verdict="pass" if ppl_ok else "fail"),
                controls=ControlsCheck(type="norm_matched_random_unit", n=spec.controls_n,
                                       max_control_effect=max(ctrl_effects),
                                       candidate_effect=effect, permutation_p=p,
                                       verdict="pass" if ctrl_ok else "fail"),
                regression=MetricCheck(value=reg, unit="percentage_points",
                                       threshold=spec.regression_gate_max_pp,
                                       verdict="pass" if reg_ok else "fail"),
            )
            if all(checks.values()):
                return (effect, alpha, files, diffs), last_screen, "", []
            failed_stage = "gates:" + "+".join(k for k, ok in checks.items() if not ok)
        return None, last_screen, failed_stage, []
