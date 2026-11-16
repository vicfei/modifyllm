"""One-command distillation smoke (piccolino v0): recipe → teacher → LoRA → frozen eval → receipt.

Transactional snapshots land with v0.3 proper; this smoke still writes a legal
receipt (states, budget, data protocol, honest verdict gate) and saves the
trained LoRA adapter as the run artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modifyllm.core import receipt as receipt_io
from modifyllm.core import txn
from modifyllm.core.ids import new_id
from modifyllm.schema.models import Budget, Failure, ModelRef, Receipt

DEFAULT_STUDENT = "Qwen/Qwen3.5-0.8B"


def run_distill_smoke(
    out_root: Path,
    teacher: str = "mock",
    piccolo_dir: str | None = None,
    student_id: str = DEFAULT_STUDENT,
    n: int = 300,
    epochs: int = 2,
    device: str = "cuda:0",
    teacher_device: str = "cpu",
    seed: int = 42,
) -> tuple[Receipt, Path]:
    if teacher == "mock":
        from modifyllm.distill.teacher import MockTeacher

        teacher_client = MockTeacher(seed=seed + 1)
    elif teacher == "piccolo":
        if not piccolo_dir:
            raise ValueError("--piccolo-dir is required for --teacher piccolo")
        from modifyllm.distill.teacher import PiccoloTeacher

        teacher_client = PiccoloTeacher(piccolo_dir, device=teacher_device)
    else:
        raise ValueError(f"unknown teacher: {teacher}")

    from modifyllm.distill import recipe
    from modifyllm.distill.student import StudentScorer, evaluate, train_lora

    run_id = new_id("run")
    receipt_id = new_id("rcp")
    run_dir = Path(out_root) / run_id
    txn.acquire_lock(run_dir)
    receipt = Receipt(
        receipt_id=receipt_id,
        run_id=run_id,
        status="running",
        ts_started=txn.utcnow(),
        model_in=ModelRef(id=teacher_client.teacher_id, revision=teacher_client.revision),
        rollback_to="",
        env={"code_version": "0.3.0-smoke", "backend": f"distill-smoke:{teacher}"},
        goal={"kind": "piccolino-smoke:tool-approval+retention+sufficiency"},
        data_protocol={},
    )
    receipt.env.update(_versions())
    receipt_io.write(run_dir, receipt)

    # 1) recipe + split (gold never enters payloads)
    dataset = recipe.generate(n, seed=seed)
    train_set, frozen_set = recipe.split(dataset, seed=seed + 100)
    receipt.data_protocol = {
        "train_set": {"n": len(train_set)},
        "frozen_set": {"n": len(frozen_set), "evaluated_once": True},
        "recipe_seed": seed,
    }

    # 2) teacher scores every instance (train + frozen) exactly once each
    teacher_probs: dict[str, dict[str, list[float]]] = {}
    teacher_rows = 0
    for inst in dataset:
        teacher_probs[inst["id"]] = teacher_client.score(inst)
        teacher_rows += len(inst["payload"]["questions"])
    _dump_jsonl(run_dir / "teacher_probs.jsonl", [
        {"id": i["id"], "probs": teacher_probs[i["id"]]} for i in dataset
    ])

    # 3) student + pre-train reference KL on a train-side sample (selection only)
    scorer = StudentScorer(student_id, device=device)
    probe = train_set[: min(16, len(train_set))]
    pre = evaluate(scorer, probe, teacher_probs)

    # 4) LoRA KL training
    train_report = train_lora(scorer, train_set, teacher_probs, epochs=epochs)
    scorer.model.save_pretrained(run_dir / "adapter")

    # 5) frozen eval — once
    frozen_metrics = evaluate(scorer, frozen_set, teacher_probs)
    _dump_jsonl(run_dir / "metrics.jsonl", [
        {"phase": "pre_train_probe", **pre},
        {"phase": "frozen_once", **frozen_metrics},
        {"phase": "train", "steps": train_report["steps"],
         "final_loss": train_report["loss_curve"][-1]["mean_loss"] if train_report["loss_curve"] else None},
    ])

    # 6) honest gate: frozen KL must beat the student's own untrained level
    gate_ok = frozen_metrics["mean_kl_teacher"] < pre["mean_kl_teacher"]
    receipt.operator = {
        "kind": "distill",
        "spec": {
            "teacher": {"id": teacher_client.teacher_id, "revision": teacher_client.revision},
            "student": student_id,
            "loss": "KL(teacher‖student) over candidate distributions",
            "attention": "causal (deviation from teacher's bidirectional prompt attention)",
            "lora_r": train_report["lora_r"], "epochs": train_report["epochs"],
            "gate": "frozen mean_kl < pre-train probe mean_kl",
            "pre_kl": pre["mean_kl_teacher"], "frozen_kl": frozen_metrics["mean_kl_teacher"],
            "accuracy_vs_gold": frozen_metrics["accuracy_vs_gold"],
            "tv_vs_teacher": frozen_metrics["tv_vs_teacher"],
        },
    }
    receipt.model_out = ModelRef(id=student_id, revision=f"{run_id}/adapter")
    receipt.budget = Budget(
        forward_passes=teacher_rows + scorer.forward_passes,
        staging_gb=0.01,
    )
    if gate_ok:
        receipt.status, receipt.verdict = "applied", "passed"
    else:
        receipt.status, receipt.verdict = "discarded", "failed"
        receipt.failure = Failure(
            stage="frozen_eval",
            error=f"frozen KL {frozen_metrics['mean_kl_teacher']:.4f} did not beat "
                  f"pre-train {pre['mean_kl_teacher']:.4f}",
            passed_checks=["receipt_valid", "teacher_scored_all"],
        )
    receipt.ts_finished = txn.utcnow()
    receipt_io.write(run_dir, receipt)
    return receipt, run_dir


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _versions() -> dict[str, str]:
    import importlib.metadata as im

    def v(pkg: str) -> str:
        try:
            return im.version(pkg)
        except im.PackageNotFoundError:
            return "absent"

    return {"torch": v("torch"), "transformers": v("transformers"), "peft": v("peft")}
