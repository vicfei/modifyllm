"""Teacher clients: score decision payloads into candidate distributions.

- MockTeacher: deterministic pseudo-teacher (seeded gold + noise) — drives the
  pipeline with zero downloads; research-grade signal NOT claimed.
- PiccoloTeacher: wraps the reference implementation shipped inside the Piccolo
  model repo (piccolo_inference.py / piccolo_layout.py, FlexDM MIT) — imports
  it from the downloaded model dir and reuses FlexCompiler + the bidirectional
  attention hook. Teacher id + revision are recorded on every score for the
  receipt.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


def _option_keys(question: dict[str, Any]) -> list[Any]:
    if question["type"] == "choice":
        return list(question.get("criteria", {}))
    return [False, True]  # noul: no / yes


class MockTeacher:
    """Seeded heuristic teacher: gold + temperature noise, renormalized."""

    def __init__(self, seed: int = 11, temperature: float = 0.35):
        self.teacher_id = "mock-heuristic-v0"
        self.revision = f"seed={seed}"
        self._rng = random.Random(seed)
        self._t = temperature

    def score(self, instance: dict[str, Any]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for qid, q in instance["payload"]["questions"].items():
            keys = _option_keys(q)
            gold = instance["gold"][qid]
            idx = keys.index(gold)
            raw = [self._rng.random() * self._t for _ in keys]
            raw[idx] += 1.0
            z = sum(raw)
            out[qid] = [r / z for r in raw]
        return out


class PiccoloTeacher:
    """Loads a downloaded Piccolo checkpoint dir and scores payloads natively.

    Mirrors inference/reference/piccolo_inference.py from the Piccolo repo:
    FlexCompiler (flex_twice), bidirectional prompt attention via forward
    pre-hooks on full-attention layers, candidate logits at the answer slot.
    CPU bfloat16 is fine; the smoke uses a few hundred instances.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu", dtype: str = "bfloat16"):
        import torch

        self.model_dir = Path(model_dir)
        self.device, self.dtype_name = device, dtype

        descriptor = json.loads((self.model_dir / "decision-inference.json").read_text())
        if descriptor.get("attention_mode") != "bidirectional_prompt" or descriptor.get("prompt_variant") != "flex_twice":
            raise ValueError("unsupported Piccolo inference contract")
        release = json.loads((self.model_dir / "release-manifest.json").read_text())
        self.teacher_id = f"piccolo/{self.model_dir.name}"
        self.revision = str(release.get("revision") or release.get("commit") or "unknown")

        # import the reference code shipped inside the model dir
        sys.path.insert(0, str(self.model_dir))
        try:
            spec = importlib.util.spec_from_file_location("piccolo_inference_mod", self.model_dir / "piccolo_inference.py")
            ref = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ref)
        finally:
            sys.path.remove(str(self.model_dir))

        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from types import SimpleNamespace

        from modifyllm.distill.flex_layout import Layout, render_question

        self._ref = ref
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.compiler = ref.FlexCompiler(
            self.tokenizer, SimpleNamespace(Layout=Layout, render_question=render_question)
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir, dtype=getattr(torch, dtype), device_map={"": device},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.decoder = self.model.model.language_model
        self.head = self.model.get_output_embeddings()
        self._attention_state = {"bidirectional": True, "calls": 0}
        self.full_attention_layers = 0
        self._register_bidirectional_hooks()

    def _register_bidirectional_hooks(self) -> None:
        import torch

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        def hook(module, positional, kwargs):
            if self._attention_state["bidirectional"]:
                hidden = kwargs["hidden_states"]
                length = hidden.shape[1]
                kwargs["attention_mask"] = torch.zeros(
                    (hidden.shape[0], 1, length, length), dtype=hidden.dtype, device=hidden.device
                )
                self._attention_state["calls"] += 1
            return positional, kwargs

        hooks = [
            m.register_forward_pre_hook(hook, with_kwargs=True)
            for m in self.decoder.modules() if isinstance(m, Qwen3_5Attention)
        ]
        expected = self.decoder.config.layer_types.count("full_attention")
        if len(hooks) != expected:
            raise ValueError(f"full-attention layer count mismatch: {len(hooks)} != {expected}")
        self.full_attention_layers = len(hooks)

    def score(self, instance: dict[str, Any]) -> dict[str, list[float]]:
        """One payload → {question_id: candidate probabilities} (teacher order preserved)."""
        import torch

        payload = instance["payload"]
        rows, _meta = self.compiler.compile(payload)
        by_qid = {row.question_id: row for row in rows}
        out: dict[str, list[float]] = {}
        with torch.inference_mode():
            for qid, row in by_qid.items():
                ids = torch.tensor([row.input_ids], device=self.device)
                hidden = self.decoder(input_ids=ids, attention_mask=None, use_cache=False).last_hidden_state
                cand = torch.tensor(row.candidate_ids, device=self.device)
                label_rows = self.head.weight[cand].float()
                logits = torch.mv(label_rows, hidden[0, -1].float())
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                if not all(math.isfinite(x) for x in probs):
                    raise ValueError("non-finite teacher probabilities")
                out[qid] = probs
        return out
