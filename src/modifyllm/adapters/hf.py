"""HF adapter: rank-1 edits on real causal-LM weights (DESIGN.md §5, v0.2).

Design notes pinned here so they are not re-litigated silently:

- **Layout.** ``down_proj.weight`` in torch is ``(d_model, d_ff)`` — the same
  matrix convention as :func:`modifyllm.core.edit.rank1_edit`; no transpose is
  needed. The torch↔numpy identity is pinned by tests/unit/test_hf_edit_math.py.
- **Readout.** The direction and the grader read the residual stream at the
  output of the *last* target layer (``hidden_states[last_idx + 1]``), i.e.
  after that layer's down_proj has written — per DESIGN §5.1.
- **Split-half grader.** No planted truth exists on real models; grading the
  edit against the very direction it was built from is circular. The engine
  therefore splits the direction set: even pairs → edit direction u, odd
  pairs → grader direction g (see RunEngine). Random-unit controls only mean
  something against an independently estimated grader.
- **Snapshots cover the touched tensors.** ``source_files``/``edited_files``
  serialize only the target layers' weights; the base checkpoint is referenced
  by ``model_id@revision`` in the receipt (``metadata``).
- **Regression gate.** v0.2 ships an in-repo mini MCQ set (deterministic,
  no downloads). MMLU-Pro subset is the backlog upgrade (DESIGN §5.4).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from modifyllm.core.edit import rank1_edit

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _dump(W: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(W, dtype=np.float32), allow_pickle=False)
    return buf.getvalue()


def _load(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _discover_down_projs(model: Any) -> list[tuple[int, str]]:
    """All down_proj weight params as (layer_index, param_name), natural order.

    Param names carry the ``.weight`` suffix — they index ``named_parameters()``
    directly (module paths would KeyError).
    """
    out: list[tuple[int, str]] = []
    for name in dict(model.named_parameters()):
        if "down_proj" in name and name.endswith(".weight"):
            m = re.search(r"layers\.(\d+)\.", name)
            idx = int(m.group(1)) if m else len(out)
            out.append((idx, name))
    out.sort()
    return out


class HFModelAdapter:
    """Loads a causal LM (or accepts an injected model), edits target down_projs."""

    split_half_grader = True  # engine: even pairs → u, odd pairs → grader g

    def __init__(
        self,
        preset: dict[str, Any],
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        layers: list[int] | None = None,
        dtype: str = "float32",  # v0.2 supports float32 (edit math in float64 numpy)
        device: str = "cpu",
        model: Any | None = None,  # injected instance (tests); skips hub load
        tokenizer: Any | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device
        self.preset = preset
        self.forward_passes = 0
        self.staging_gb_value = 0.001
        self._grader: np.ndarray | None = None

        if model is None:
            import torch  # lazy: hf extra
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
            except TypeError:  # older transformers: torch_dtype kwarg
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, torch_dtype=torch.float32
                )
            self.model.to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self._discovered = _discover_down_projs(self.model)
        if not self._discovered:
            raise ValueError("model exposes no *down_proj* weight parameters")
        by_idx = dict(self._discovered)
        if layers is None:
            layers = _default_layer_indices(len(self._discovered))
        chosen: list[tuple[int, str]] = []
        for i in layers:
            if int(i) in by_idx:
                chosen.append(by_idx[int(i)])
            elif 0 <= int(i) < len(self._discovered):
                chosen.append(self._discovered[int(i)])
            else:
                raise ValueError(f"layer {i} not found (model has {len(self._discovered)} down_proj layers)")
        self.layer_indices = [i for i, _ in chosen]
        self.layer_names = [n for _, n in chosen]
        self.readout_index = max(self.layer_indices) + 1  # residual after last target

        params = dict(self.model.named_parameters())
        self._params = {name: params[name] for name in self.layer_names}
        self._orig: dict[str, np.ndarray] = {}
        self._weights: dict[str, np.ndarray] = {}
        for name, p in self._params.items():
            W = p.detach().cpu().float().numpy()  # torch → numpy, always fp32 on disk
            self._orig[name] = W.copy()
            self._weights[name] = W
        self._ppl_source: float | None = None
        self._acc_source: float | None = None

    # -- ModelAdapter ------------------------------------------------------
    def source_files(self) -> dict[str, bytes]:
        return {name: _dump(self._orig[name]) for name in self.layer_names}

    def total_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def collect_pair_hiddens(self, n_pairs: int, rng: np.random.Generator) -> list[np.ndarray]:
        pairs = self.preset["pairs"]
        idx = rng.choice(len(pairs), size=n_pairs, replace=n_pairs > len(pairs))
        deltas = []
        for i in idx:
            h_u = self._readout_text(pairs[int(i)][0])
            h_w = self._readout_text(pairs[int(i)][1])
            deltas.append(h_u - h_w)
        return deltas

    def eval_items(self, n: int, seed: int) -> np.ndarray:
        texts = list(self.preset["eval_texts"])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(texts))[:n]
        while len(idx) < n:  # cycle deterministically if n exceeds corpus
            idx = np.concatenate([idx, rng.permutation(len(texts))[: n - len(idx)]])
        return np.asarray(texts, dtype=object)[idx]

    def set_grader_direction(self, g: np.ndarray) -> None:
        self._grader = np.asarray(g, dtype=np.float64)

    def goal_outputs(self, items: np.ndarray) -> np.ndarray:
        return self._goal_with(None, items)  # source weights: no patch needed

    def edited_files(self, u: np.ndarray, alpha: float) -> dict[str, bytes]:
        files = {name: _dump(rank1_edit(self._weights[name], u, alpha)) for name in self.layer_names}
        self.staging_gb_value = round(sum(len(b) for b in files.values()) / 1e9, 4)
        return files

    def goal_scores_with_weights(self, files: dict[str, bytes], items: np.ndarray) -> np.ndarray:
        return self._goal_with(files, items)

    def measure_ppl_drift(self, files: dict[str, bytes]) -> float:
        ppl_src = self._ppl_source_lazy()
        with self._patched(files):
            ppl_ed = self._ppl()
        return float(ppl_ed / ppl_src - 1.0)

    def measure_regression_pp(self, files: dict[str, bytes]) -> float:
        acc_src = self._acc_source_lazy()
        with self._patched(files):
            acc_ed = self._mcq_accuracy()
        return float((acc_src - acc_ed) * 100.0)  # percentage-point drop

    # -- HF specifics --------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        import importlib.metadata as im

        def _ver(pkg: str) -> str:
            try:
                return im.version(pkg)
            except im.PackageNotFoundError:
                return "absent"

        return {
            "backend": "hf",
            "torch": _ver("torch"),
            "transformers": _ver("transformers"),
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "n_model_layers": int(self.model.config.num_hidden_layers),
            "target_layer_indices": self.layer_indices,
            "snapshot_scope": "touched tensors; base referenced by model_id@revision",
        }

    # -- internals -----------------------------------------------------------
    def _ids(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    def _forward_hidden(self, text: str) -> np.ndarray:
        """Mean-pooled residual at readout_index over all tokens of `text` (DESIGN §5.1)."""
        ids = self._ids(text)
        self.forward_passes += 1
        import torch

        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[self.readout_index][0]  # (L, d)
        return h.detach().cpu().float().numpy().mean(axis=0).astype(np.float64)

    def _readout_text(self, text: str) -> np.ndarray:
        return self._forward_hidden(text)

    def _goal_with(self, files: dict[str, bytes] | None, items: np.ndarray) -> np.ndarray:
        if self._grader is None:
            raise RuntimeError("set_grader_direction() must be called before scoring (engine contract)")
        import contextlib

        g = self._grader
        d = float(np.sqrt(len(g)))
        scores = []
        ctx = self._patched(files) if files is not None else contextlib.nullcontext()
        with ctx:
            for text in items:
                h = self._forward_hidden(str(text))
                scores.append(max(0.0, float(g @ h)) / d)
        return np.asarray(scores, dtype=np.float64)

    def _patched(self, files: dict[str, bytes]):
        import contextlib
        import torch

        @contextlib.contextmanager
        def _ctx():
            with torch.no_grad():
                for name, blob in files.items():
                    W = torch.from_numpy(_load(blob)).to(self._params[name].device)
                    self._params[name].copy_(W)
            try:
                yield
            finally:
                with torch.no_grad():
                    for name in files:
                        orig = torch.from_numpy(self._orig[name]).to(self._params[name].device)
                        self._params[name].copy_(orig)

        return _ctx()

    def _ppl_ids(self):
        corpus = str(self.preset["ppl_corpus"])
        return self._ids(corpus)[0]  # 1-D over tokens

    def _ppl(self) -> float:
        import torch
        import torch.nn.functional as F

        ids = self._ppl_ids()
        self.forward_passes += 1
        with torch.no_grad():
            logits = self.model(ids[None, :]).logits[0].to(torch.float64)
        nll = F.cross_entropy(logits[:-1], ids[1:].to(logits.device)).item()
        return float(np.exp(nll))

    def _ppl_source_lazy(self) -> float:
        if self._ppl_source is None:
            self._ppl_source = self._ppl()
        return self._ppl_source

    def _mcq_accuracy(self) -> float:
        import torch

        correct = 0
        items = self.preset["mcq"]
        for item in items:
            ids = self._ids(str(item["prompt"]))
            self.forward_passes += 1
            with torch.no_grad():
                nxt = self.model(ids).logits[0, -1]
            want = self._first_token_id(str(item["answer"]))
            if int(torch.argmax(nxt).item()) == want:
                correct += 1
        return correct / len(items)

    def _first_token_id(self, text: str) -> int:
        enc = self.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]
        return int(ids[0])


def _default_layer_indices(n_layers: int) -> list[int]:
    """Two mid-late layer indices, clamped to [0, n_layers-1], de-duplicated."""
    if n_layers <= 2:
        return list(range(n_layers))
    i1 = min(n_layers - 1, max(0, round(n_layers * 2 / 3)))
    i2 = min(n_layers - 1, max(0, round(n_layers * 5 / 6)))
    return sorted({i1, i2})


def load_preset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
