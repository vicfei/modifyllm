"""Student: small causal LM scoring Piccolo-style candidate rows + LoRA KL trainer.

Documented deviation: the student keeps STANDARD CAUSAL attention (no
bidirectional prompt-attention hook) — the single fidelity gap of piccolino,
recorded in every receipt. Everything else (chat template, Context/Question/
Options rendering, single-token labels at "Answer: (") reuses the vendored
FlexDM layout, so teacher and student see byte-identical prompts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _decoder_and_head(model):
    """Access the text decoder + output head through plain or PEFT-wrapped models."""
    base = model
    if type(base).__name__ == "PeftModel":
        base = base.base_model.model  # LoraModel wrapper -> original model
    return base.model.language_model, base.get_output_embeddings()


class StudentScorer:
    """Causal candidate scorer over the shared flex layout."""

    def __init__(self, model_id: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        from modifyllm.distill.flex_layout import Layout

        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.layout = Layout(self.tokenizer)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map={"": device},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.forward_passes = 0

    def compile_rows(self, payload: dict[str, Any]):
        """Same row structure as the teacher compiler, but causal (once, no repeat)."""
        from modifyllm.distill.flex_layout import render_question

        texts, rendered = [], []
        for qid, q in payload["questions"].items():
            text, options, keys = render_question(dict(q))
            rendered.append((qid, text, options, keys))
            texts.append(text)
        ctx = self.layout.context(payload["state"], texts)
        rows = []
        for qid, text, options, keys in rendered:
            row_ids = ctx + self.layout.question(text, options)
            cand = tuple(self.layout.label_ids[: len(keys)])
            rows.append((qid, tuple(keys), cand, tuple(row_ids)))
        return rows

    def distribution(self, row_ids, candidate_ids) -> np.ndarray:
        """Softmax over candidate label logits at the answer slot (causal forward)."""
        import torch

        self.forward_passes += 1
        decoder, head = _decoder_and_head(self.model)
        ids = torch.tensor([list(row_ids)], device=self.device)
        with torch.inference_mode():
            hidden = decoder(input_ids=ids, attention_mask=None, use_cache=False).last_hidden_state
            cand = torch.tensor(list(candidate_ids), device=self.device)
            logits = torch.mv(head.weight[cand].float(), hidden[0, -1].float())
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy().astype(np.float64)


def train_lora(
    scorer: StudentScorer,
    dataset: list[dict[str, Any]],
    teacher_probs: dict[str, dict[str, list[float]]],
    epochs: int = 2,
    lr: float = 1e-4,
    lora_r: int = 16,
    seed: int = 0,
    log_every: int = 50,
) -> dict[str, Any]:
    """LoRA + KL(teacher ‖ student) over candidate distributions. Returns curve + adapter path info."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    lcfg = LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION",
    )
    scorer.model = get_peft_model(scorer.model, lcfg)
    params = [p for p in scorer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    def _loss_of(instance):
        row_cache = scorer.compile_rows(instance["payload"])
        losses = []
        for qid, keys, cand, row_ids in row_cache:
            target = torch.tensor(teacher_probs[instance["id"]][qid], device=scorer.device)
            ids = torch.tensor([list(row_ids)], device=scorer.device)
            decoder, head = _decoder_and_head(scorer.model)
            hidden = decoder(input_ids=ids, attention_mask=None, use_cache=False).last_hidden_state
            cand_t = torch.tensor(list(cand), device=scorer.device)
            logits = torch.mv(head.weight[cand_t].float(), hidden[0, -1].float())
            logq = torch.log_softmax(logits, dim=-1)
            logp = torch.log(target + 1e-12)
            losses.append(torch.sum(target * (logp - logq)))
        return torch.stack(losses).mean()

    curve = []
    scorer.model.train()
    step = 0
    for epoch in range(epochs):
        total, count = 0.0, 0
        for instance in dataset:
            loss = _loss_of(instance)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            total += float(loss.item())
            count += 1
            step += 1
            if step % log_every == 0:
                curve.append({"step": step, "mean_loss": total / max(count, 1)})
                total, count = 0.0, 0
    scorer.model.eval()
    return {"loss_curve": curve, "lora_r": lora_r, "epochs": epochs, "lr": lr, "steps": step}


def evaluate(
    scorer: StudentScorer,
    dataset: list[dict[str, Any]],
    teacher_probs: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    """Frozen metrics: mean KL vs teacher, accuracy vs gold (argmax), TV vs teacher."""
    kls, accs, tvs = [], [], []
    for instance in dataset:
        rows = scorer.compile_rows(instance["payload"])
        for qid, keys, cand, row_ids in rows:
            p = np.asarray(teacher_probs[instance["id"]][qid], dtype=np.float64)
            q = scorer.distribution(row_ids, cand)
            q_safe = np.clip(q, 1e-12, None)
            kls.append(float(np.sum(p * (np.log(np.clip(p, 1e-12, None)) - np.log(q_safe)))))
            tvs.append(float(0.5 * np.abs(p - q).sum()))
            pred = keys[int(np.argmax(q))]
            accs.append(1.0 if pred == instance["gold"][qid] else 0.0)
    return {
        "mean_kl_teacher": float(np.mean(kls)),
        "accuracy_vs_gold": float(np.mean(accs)),
        "tv_vs_teacher": float(np.mean(tvs)),
        "n_questions": len(kls),
    }
