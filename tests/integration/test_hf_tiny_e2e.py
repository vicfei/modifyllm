"""HF adapter end-to-end on a tiny random model — offline, no downloads.

Machinery assertions only (receipt states, gate numerics, snapshot integrity,
rollback roundtrip): per DESIGN §8 v0.2 the verdict itself is allowed to be
"no effect" on any given world.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")

from modifyllm.adapters.hf import HFModelAdapter  # noqa: E402
from modifyllm.core import txn  # noqa: E402
from modifyllm.core.receipt import load  # noqa: E402
from modifyllm.core.run import RunEngine  # noqa: E402
from modifyllm.schema.models import EditSpec, EvalSpec, GoalSpec  # noqa: E402


class FakeTokenizer:
    """Deterministic whitespace tokenizer — enough for adapter plumbing tests."""

    def __init__(self, vocab: int = 512):
        self.vocab_size = vocab

    def _ids(self, text: str, add_special: bool) -> list[int]:
        ids = [1] if add_special else []
        for w in text.lower().split():
            ids.append(10 + (sum(ord(c) for c in w) % (self.vocab_size - 16)))
        return ids or [2]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._ids(text, add_special_tokens)
        if return_tensors is None:
            return {"input_ids": ids}
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@pytest.fixture(scope="module")
def tiny_adapter():
    kwargs = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=512, max_position_embeddings=256,
    )
    model = None
    for arch in ("qwen3_5", "qwen3", "qwen2"):
        try:
            cfg = tr.AutoConfig.for_model(arch, **kwargs)
            model = tr.AutoModelForCausalLM.from_config(cfg)
            break
        except Exception:  # noqa: BLE001 — try next arch family
            continue
    if model is None:
        pytest.skip("installed transformers exposes no qwen-family config")
    model.eval()
    preset = {
        "pairs": [
            ["sure i would be happy to help with that answer", "answer"],
            ["great question let me walk you through it", "here is the result"],
            ["absolutely here is what you need to know now", "the fact"],
            ["of course let me break that down for you", "the summary"],
            ["certainly i would be delighted to explain this", "the explanation"],
            ["fantastic question to answer it simply put", "the answer"],
            ["sure thing happy to help with that one", "the output"],
            ["wonderful let me dive right in today", "the value"],
        ],
        "eval_texts": [
            "sure i would be happy to help",
            "great question let me explain",
            "absolutely here is the thing",
            "of course let me break it down",
            "certainly delighted to help",
            "fantastic simply put the answer",
            "sure thing coming right up",
            "wonderful let me dive in",
        ],
        "ppl_corpus": "the train leaves at seven the market opens at eight the library closes at ten",
        "mcq": [
            {"prompt": "Question: What is the capital of France?\nAnswer:", "answer": " Paris"},
            {"prompt": "Question: How many sides does a square have?\nAnswer:", "answer": " Four"},
        ],
    }
    return HFModelAdapter(
        preset=preset, model_id="tiny-test://random", revision="test-0",
        layers=[1, 2], model=model, tokenizer=FakeTokenizer(),
    )


def test_adapter_basics(tiny_adapter):
    assert len(tiny_adapter.layer_names) == 2
    assert all(n.endswith("mlp.down_proj.weight") for n in tiny_adapter.layer_names)
    assert tiny_adapter.readout_index == 3
    src = tiny_adapter.source_files()
    assert set(src) == set(tiny_adapter.layer_names)
    # weights unchanged by serialization
    for name, blob in src.items():
        W = np.load(io.BytesIO(blob), allow_pickle=False)
        assert np.array_equal(W, tiny_adapter._orig[name])


def test_split_half_probe_and_grader(tiny_adapter):
    rng = np.random.default_rng(11)
    deltas = tiny_adapter.collect_pair_hiddens(8, rng)
    assert len(deltas) == 8 and deltas[0].shape == (32,)
    from modifyllm.core.probe import direction_from_pair_deltas

    u = direction_from_pair_deltas(deltas[0::2])
    g = direction_from_pair_deltas(deltas[1::2])
    tiny_adapter.set_grader_direction(g)
    items = tiny_adapter.eval_items(4, seed=5)
    scores = tiny_adapter.goal_outputs(items)
    assert scores.shape == (4,) and np.all(np.isfinite(scores))


def test_edited_files_patch_and_restore(tiny_adapter):
    rng = np.random.default_rng(3)
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    files = tiny_adapter.edited_files(u, 1.0)
    assert set(files) == set(tiny_adapter.layer_names)
    items = tiny_adapter.eval_items(4, seed=7)
    tiny_adapter.set_grader_direction(u)
    tiny_adapter.goal_scores_with_weights(files, items)
    # restore must be bit-exact
    for name, p in tiny_adapter._params.items():
        assert np.array_equal(p.detach().numpy(), tiny_adapter._orig[name])
    drift = tiny_adapter.measure_ppl_drift(files)
    reg = tiny_adapter.measure_regression_pp(files)
    assert math.isfinite(drift) and math.isfinite(reg)
    # gates restored again
    for name, p in tiny_adapter._params.items():
        assert np.array_equal(p.detach().numpy(), tiny_adapter._orig[name])


def test_engine_e2e_tiny(tmp_path, tiny_adapter):
    goal = GoalSpec(
        kind="preset:filler-prefix", pairs_ref="tiny://in-memory",
        grader={"type": "split-half-hidden-direction", "version": "0"},
    )
    ev = EvalSpec(direction_set_n=8, selection_set_n=8, frozen_set_n=8,
                  bootstrap_B=500, permutation_max=1000)
    spec = EditSpec(layers=list(tiny_adapter.layer_names), alpha_grid=[0.5, 2.0], controls_n=4)
    receipt = RunEngine().run(tmp_path, tiny_adapter, goal, ev, spec)
    run_dir = tmp_path / receipt.run_id

    assert receipt.status in {"applied", "discarded"}
    assert receipt.env["backend"] == "hf"
    assert receipt.env["model_id"] == "tiny-test://random"
    assert receipt.budget.forward_passes > 0
    assert receipt.goal["grader"]["type"] == "split-half-hidden-direction"

    manifest = txn.verify_snapshot(run_dir / "snap_src")
    assert manifest["digest"] == receipt.model_in.files_sha256

    if receipt.status == "applied":
        out_snap = run_dir / receipt.deployed_snapshot
        txn.verify_snapshot(out_snap)
        assert txn.resolve_deployed(run_dir) == receipt.deployed_snapshot
        event = txn.rollback(run_dir, load(run_dir))
        assert event.parent_receipt_id == receipt.receipt_id
        assert txn.resolve_deployed(run_dir) == "snap_src"
        assert load(run_dir).status == "rolled_back"
    else:
        assert receipt.failure is not None and receipt.failure.stage
        assert not (run_dir / "DEPLOYED").exists()
    if receipt.screen is not None:
        assert math.isfinite(receipt.screen.ppl_gate.value)
        assert math.isfinite(receipt.screen.controls.candidate_effect)
