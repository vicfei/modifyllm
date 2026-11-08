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
