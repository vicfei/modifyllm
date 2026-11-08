"""Receipt persistence: write / load / list / compare (DESIGN.md §6)."""

from __future__ import annotations

import json
from pathlib import Path

from modifyllm.schema.models import Receipt


def receipt_path(run_dir: Path) -> Path:
    return run_dir / "receipt.json"


def write(run_dir: Path, r: Receipt) -> Path:
    p = receipt_path(run_dir)
    tmp = run_dir / "receipt.json.tmp"
    tmp.write_text(json.dumps(r.model_dump(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(p)  # atomic on same filesystem
    return p


def load(run_dir: Path) -> Receipt:
    return Receipt.model_validate(json.loads(receipt_path(run_dir).read_text(encoding="utf-8")))


def list_runs(root: Path) -> list[Receipt]:
    out: list[Receipt] = []
    for p in sorted(root.glob("*/receipt.json")):
        try:
            out.append(Receipt.model_validate(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001 - unreadable runs are listed as absent
            continue
    return out


def compare(a: Receipt, b: Receipt) -> str:
    def line(r: Receipt) -> str:
        se = r.screen.frozen_eval if r.screen else None
        eff = "?"
        if se is not None:
            eff = f"{se.value:+.4f} CI[{se.ci95[0]:+.4f},{se.ci95[1]:+.4f}]"
        elif r.screen is not None:
            eff = f"sel={r.screen.controls.candidate_effect:+.4f}"
        return (f"{r.receipt_id[:16]}  status={r.status:<10} verdict={r.verdict:<8} "
                f"alpha={r.operator.get('spec', {}).get('alpha', '?'):<5} effect={eff}")

    return "\n".join([line(a), line(b)])
