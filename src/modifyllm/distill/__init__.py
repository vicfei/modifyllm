"""Decision-model distillation (v0.3 smoke scope).

piccolino: distill Piccolo decision distributions (teachers) into small
same-family students (Qwen3.5-0.8B / 2B), with modifyllm receipts.

- recipe.py   synthetic decision instances (tool approval / retention / sufficiency)
- teacher.py  MockTeacher + PiccoloTeacher (wraps the model repo's reference code)
- student.py  causal candidate scorer + LoRA KL trainer
- run_smoke.py one-command smoke pipeline with receipt
"""
