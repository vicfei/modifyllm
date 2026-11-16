# modifyllm

**Modify with receipts — verified, auditable, rollbackable model modification.**

One interface over the three model-modification operators — **edit** (weight-space
programming, implemented here), **finetune** / **distill** / **merge** (delegated
backends, v0.3) — where every modification is a *transaction*:

```
snapshot → staged edit → screening (random controls + permutation test + gates)
        → single frozen-test (bootstrap CI) → atomic commit → receipt → rollback-able
```

- No receipt, no modification — failures are receipted too.
- `--no-screen` never yields `passed`; it yields `screen_bypassed`.
- The math is pinned by identity tests (`uᵀ(W′h) = (1−α)·uᵀ(Wh)`).

Read [`docs/DESIGN.md`](docs/DESIGN.md) for the full specification (operator theory,
three-way data protocol, transaction states, receipt schema).

## Status

v0.1 engineering closure (see DESIGN §8):

- [x] rank-1 projection edit, whole-matrix, unit-direction (sign convention pinned)
- [x] screening: PPL gate · norm-matched random controls + sign-flip permutation · regression gate
- [x] direction / selection / frozen three-way data protocol; frozen set evaluated once
- [x] two-phase commit, crash recovery, rollback with event receipts
- [x] measured false-positive rate on a null world (stat suite)
- [ ] HF backend + real-model nightly (v0.2) · delegated operators (v0.3)

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                   # unit + stat + txn + integration (CPU, seconds)

modifyllm edit --out runs --planted      # full pipeline on the synthetic world
modifyllm receipts-list runs             # every run, every state
modifyllm rollback runs/<run_id>         # restore the source snapshot
```

The v0.1 backend is a deterministic synthetic world (numpy) — it exists so the
transaction and statistics machinery can be tested exactly, without a GPU. The
HF adapter (Qwen-class small models) lands in v0.2 behind the same interface.

## Safety boundary

No preset removes safety or refusal behaviors. Custom pair sets are permitted for
research; receipts record their reference hash. Bypassing screening marks the
verdict `screen_bypassed` — it can never be reported as `passed`.

## Related maps

modifyllm is the hands-on practice of three curated maps of the fields it draws on:

- [awesome-self-improving-coding-agents](https://github.com/vicfei/awesome-self-improving-coding-agents) — RSI × software engineering: the improvement loop this tool guards (verify before you keep)
- [awesome-agentic-ml](https://github.com/vicfei/awesome-agentic-ml) — agents building models & data: the operator theory (acquire / transfer / edit) the three commands implement
- [awesome-weight-editing](https://github.com/vicfei/awesome-weight-editing) — the editing operator in depth: the five research families behind the `edit` command

## License

Apache-2.0 (code) · CC-BY-4.0 (docs)
