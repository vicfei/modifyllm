"""Two-phase commit, snapshots, rollback, crash injection (DESIGN.md §3).

Layout under runs/<run_id>/:
  lock                 run lock (O_CREAT|O_EXCL)
  receipt.json         live receipt (atomic replace)
  stage/               staged candidate weights (pre-commit)
  snap_src/            immutable source snapshot + manifest.json
  snap_<receipt_id>/   committed output snapshot + manifest.json
  DEPLOYED             pointer file naming the currently deployed snapshot
  events/              event receipts (rollback, recovery)

Crash injection for tests: MODIFYLLM_CRASH_AT in {after_stage, before_commit, after_commit}.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from modifyllm.core import receipt as receipt_io
from modifyllm.schema.models import Failure, Receipt


def _crash_point() -> str | None:
    return os.environ.get("MODIFYLLM_CRASH_AT")


def maybe_crash(point: str) -> None:
    if _crash_point() == point:
        os._exit(70)  # hard exit simulates a crash; no cleanup runs


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def acquire_lock(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, utcnow().encode())
    os.close(fd)
    return lock


def files_digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(hashlib.sha256(files[name]).digest())
    return h.hexdigest()


def write_snapshot(snapshot_dir: Path, files: dict[str, bytes]) -> dict:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (snapshot_dir / name).write_bytes(blob)
    manifest = {
        "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
        "digest": files_digest(files),
        "created": utcnow(),
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_snapshot(snapshot_dir: Path) -> dict:
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        got = hashlib.sha256((snapshot_dir / name).read_bytes()).hexdigest()
        if got != sha:
            raise RuntimeError(f"snapshot corruption: {name}")
    return manifest


def stage_candidate(run_dir: Path, files: dict[str, bytes]) -> Path:
    stage = run_dir / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name, blob in files.items():
        (stage / name).write_bytes(blob)
    maybe_crash("after_stage")
    return stage


def commit_staged(run_dir: Path, out_name: str, files: dict[str, bytes]) -> Path:
    """Seal the staged candidate with its manifest, then atomically rename."""
    maybe_crash("before_commit")
    out_dir = run_dir / out_name
    if out_dir.exists():
        raise RuntimeError(f"output snapshot already exists: {out_name}")
    stage = run_dir / "stage"
    manifest = {
        "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())},
        "digest": files_digest(files),
        "created": utcnow(),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stage.rename(out_dir)  # atomic on same filesystem
    maybe_crash("after_commit")
    return out_dir


def deploy(run_dir: Path, snapshot_name: str) -> None:
    """Atomically point DEPLOYED at a snapshot (temp file + rename)."""
    tmp = run_dir / "DEPLOYED.tmp"
    tmp.write_text(snapshot_name, encoding="utf-8")
    tmp.replace(run_dir / "DEPLOYED")


def resolve_deployed(run_dir: Path) -> str:
    return (run_dir / "DEPLOYED").read_text(encoding="utf-8").strip()


def rollback(run_dir: Path, r: Receipt) -> Receipt:
    """Restore DEPLOYED to the source snapshot; mark original receipt; emit event receipt."""
    if not r.rollback_to:
        raise RuntimeError("receipt has no rollback target")
    src = run_dir / r.rollback_to
    verify_snapshot(src)
    deploy(run_dir, r.rollback_to)
    r.status = "rolled_back"
    receipt_io.write(run_dir, r)
    from modifyllm.core.ids import new_id  # local import to avoid cycle at module load

    event = Receipt(
        receipt_id=new_id("rcp"),
        run_id=r.run_id,
        parent_receipt_id=r.receipt_id,
        status="rolled_back",
        is_event=True,
        ts_started=utcnow(),
        ts_finished=utcnow(),
        env=r.env,
        model_in=r.model_in,
        rollback_to=r.rollback_to,
        verdict="na",
        warnings=[f"rollback of {r.receipt_id}; deployed restored to {r.rollback_to}"],
    )
    events = run_dir / "events"
    events.mkdir(exist_ok=True)
    (events / f"{event.receipt_id}.json").write_text(
        json.dumps(event.model_dump(), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return event


def scan_interrupted(root: Path) -> list[str]:
    """Mark stale 'running' receipts as interrupted. Returns recovered run ids."""
    recovered: list[str] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rp = receipt_io.receipt_path(run_dir)
        if not rp.exists():
            continue
        try:
            r = receipt_io.load(run_dir)
        except Exception:  # noqa: BLE001
            continue
        if r.status in {"started", "running"}:
            r.status = "interrupted"
            r.failure = r.failure or Failure(stage="unknown", error="process exited without terminal state")
            receipt_io.write(run_dir, r)
            recovered.append(r.run_id)
    return recovered
