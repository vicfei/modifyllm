"""modifyllm CLI (v0.1: synthetic world; HF backend arrives in v0.2)."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Modify with receipts.")


@app.command()
def edit(
    out: Path = typer.Option(Path("runs"), help="runs root directory"),
    seed: int = typer.Option(7),
    planted: bool = typer.Option(True, "--planted/--null", help="planted signal or null world"),
    no_screen: bool = typer.Option(False, "--no-screen", help="bypass screening (verdict=screen_bypassed)"),
):
    """Run the full pipeline on the synthetic world; prints the receipt summary."""
    from modifyllm.core.run import run_synthetic

    receipt, run_dir = run_synthetic(out, seed=seed, planted=planted, screen_bypass=no_screen)
    typer.echo(f"run_dir: {run_dir}")
    typer.echo(f"status:  {receipt.status}  verdict: {receipt.verdict}")
    if receipt.screen is not None and receipt.screen.frozen_eval is not None:
        fe = receipt.screen.frozen_eval
        typer.echo(f"frozen:  {fe.value:+.4f} CI95 [{fe.ci95[0]:+.4f}, {fe.ci95[1]:+.4f}] ({fe.verdict})")
    if receipt.failure is not None:
        typer.echo(f"failure: {receipt.failure.stage} — {receipt.failure.error}")


@app.command()
def edit_hf(
    out: Path = typer.Option(Path("runs"), help="runs root directory"),
    model: str | None = typer.Option(None, help="HF model id (default: Qwen/Qwen3.5-0.8B)"),
    revision: str | None = typer.Option(None, help="model revision sha or branch (default: pinned)"),
    preset: Path | None = typer.Option(None, help="preset JSON (default: bundled filler_prefix)"),
    layers: str | None = typer.Option(None, help="comma-separated target layer indices (default: auto mid-late)"),
    no_screen: bool = typer.Option(False, "--no-screen", help="bypass screening (verdict=screen_bypassed)"),
    direction_n: int = typer.Option(8),
    selection_n: int = typer.Option(16),
    frozen_n: int = typer.Option(24),
    controls_n: int = typer.Option(8),
):
    """Run the full pipeline on a real HF model (v0.2 backend)."""
    from modifyllm.core.run import run_hf

    layer_idx = [int(x) for x in layers.split(",")] if layers else None
    receipt, run_dir = run_hf(
        out, preset_path=preset, model_id=model, revision=revision, layers=layer_idx,
        screen_bypass=no_screen, n_direction=direction_n, n_selection=selection_n,
        n_frozen=frozen_n, controls_n=controls_n,
    )
    typer.echo(f"run_dir: {run_dir}")
    typer.echo(f"status:  {receipt.status}  verdict: {receipt.verdict}")
    if receipt.screen is not None:
        pg = receipt.screen.ppl_gate
        typer.echo(f"ppl:     {pg.value:+.4f} (gate {pg.threshold}) ({pg.verdict})")
        if receipt.screen.frozen_eval is not None:
            fe = receipt.screen.frozen_eval
            typer.echo(f"frozen:  {fe.value:+.4f} CI95 [{fe.ci95[0]:+.4f}, {fe.ci95[1]:+.4f}] ({fe.verdict})")
    if receipt.failure is not None:
        typer.echo(f"failure: {receipt.failure.stage} — {receipt.failure.error}")


@app.command()
def distill_smoke(
    out: Path = typer.Option(Path("runs"), help="runs root directory"),
    teacher: str = typer.Option("mock", help="mock | piccolo"),
    piccolo_dir: str | None = typer.Option(None, help="downloaded Piccolo checkpoint dir (teacher=piccolo)"),
    student: str = typer.Option("Qwen/Qwen3.5-0.8B", help="student model id"),
    n: int = typer.Option(300, help="instances"),
    epochs: int = typer.Option(2),
    device: str = typer.Option("cuda:0", help="student device"),
    teacher_device: str = typer.Option("cpu", help="piccolo teacher device"),
):
    """piccolino v0 smoke: recipe → teacher scores → LoRA KL → frozen eval → receipt."""
    from modifyllm.distill.run_smoke import run_distill_smoke

    receipt, run_dir = run_distill_smoke(
        out, teacher=teacher, piccolo_dir=piccolo_dir, student_id=student,
        n=n, epochs=epochs, device=device, teacher_device=teacher_device,
    )
    typer.echo(f"run_dir: {run_dir}")
    typer.echo(f"status:  {receipt.status}  verdict: {receipt.verdict}")
    spec = receipt.operator.get("spec", {})
    if spec:
        typer.echo(
            f"kl:      pre {spec.get('pre_kl', 0):.4f} → frozen {spec.get('frozen_kl', 0):.4f} "
            f"| acc_vs_gold {spec.get('accuracy_vs_gold', 0):.3f} | tv {spec.get('tv_vs_teacher', 0):.3f}"
        )
    if receipt.failure is not None:
        typer.echo(f"failure: {receipt.failure.stage} — {receipt.failure.error}")


@app.command()
def recover(root: Path = typer.Argument(..., help="runs root")):
    """Mark stale running receipts as interrupted."""
    from modifyllm.core.txn import scan_interrupted

    recovered = scan_interrupted(root)
    typer.echo(f"recovered {len(recovered)} run(s)" + (f": {', '.join(recovered)}" if recovered else ""))


@app.command()
def receipts_list(root: Path = typer.Argument(...)):
    from modifyllm.core.receipt import list_runs

    for r in list_runs(root):
        typer.echo(f"{r.run_id}  {r.status:<12} verdict={r.verdict:<8} {r.ts_started}")


@app.command()
def receipts_show(run_dir: Path = typer.Argument(...)):
    from modifyllm.core.receipt import load

    typer.echo(load(run_dir).model_dump_json(indent=2))


@app.command()
def rollback(run_dir: Path = typer.Argument(...)):
    """Restore DEPLOYED to the source snapshot; mark receipt rolled_back."""
    from modifyllm.core.receipt import load
    from modifyllm.core.txn import rollback as do_rollback

    event = do_rollback(run_dir, load(run_dir))
    typer.echo(f"rolled back; event receipt {event.receipt_id}")


@app.command()
def deployed(run_dir: Path = typer.Argument(...)):
    from modifyllm.core.txn import resolve_deployed

    typer.echo(resolve_deployed(run_dir))


def run() -> None:
    app()
