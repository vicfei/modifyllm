"""Pydantic schemas: GoalSpec / EvalSpec / OperatorSpec / Receipt (DESIGN.md §3, §6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ReceiptStatus = Literal[
    "started", "running", "passed", "failed", "applied", "discarded",
    "interrupted", "rolled_back", "screen_bypassed",
]
TopVerdict = Literal["passed", "failed", "na", "screen_bypassed"]
CheckVerdict = Literal["pass", "fail", "na"]


class GraderSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str  # e.g. "exact@math500-subset"
    version: str = "0"


class GoalSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str  # e.g. "preset:math-loops"
    pairs_ref: str  # path or sha256:... (content reference; never inlined into receipts)
    grader: GraderSpec


class EvalSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    direction_set_n: int
    selection_set_n: int
    frozen_set_n: int
    seed_data: int = 1234
    seed_bootstrap: int = 42
    bootstrap_B: int = 2000
    permutation_max: int = 5000


class EditSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    recipe: Literal["rank1"] = "rank1"
    layers: list[str]
    alpha_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    controls_n: int = 24  # rank test level 1/(n+1) ≈ 4%; n=8 would give 11%
    ppl_gate_max_relative_drift: float = 0.05
    regression_gate_max_pp: float = 1.0
    max_candidates: int = 12


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    revision: str = "head"
    files_sha256: str = ""


class MetricCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: float
    unit: str
    threshold: float | None = None
    n: int | None = None
    verdict: CheckVerdict


class ControlsCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    n: int
    max_control_effect: float
    candidate_effect: float
    permutation_p: float
    verdict: CheckVerdict


class FrozenEvalCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: float
    unit: str
    ci95: list[float]
    bootstrap: dict[str, Any] = Field(default_factory=dict)
    verdict: CheckVerdict


class ScreenReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    ppl_gate: MetricCheck
    controls: ControlsCheck
    regression: MetricCheck
    frozen_eval: FrozenEvalCheck | None = None


class Budget(BaseModel):
    model_config = ConfigDict(extra="allow")

    forward_passes: int
    staging_gb: float = 0.0
    gpu_mem_peak_gb: float = 0.0


class Failure(BaseModel):
    model_config = ConfigDict(extra="allow")

    stage: str
    error: str
    passed_checks: list[str] = Field(default_factory=list)


class Receipt(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "0.2"
    receipt_id: str
    run_id: str
    parent_receipt_id: str | None = None
    status: ReceiptStatus
    is_event: bool = False  # rollback/recovery events
    ts_started: str = ""
    ts_finished: str = ""
    env: dict[str, Any] = Field(default_factory=dict)
    model_in: ModelRef
    model_out: ModelRef | None = None
    rollback_to: str = ""
    deployed_snapshot: str = ""
    operator: dict[str, Any] = Field(default_factory=dict)
    goal: dict[str, Any] = Field(default_factory=dict)
    data_protocol: dict[str, Any] = Field(default_factory=dict)
    screen: ScreenReport | None = None
    budget: Budget | None = None
    verdict: TopVerdict = "na"
    warnings: list[str] = Field(default_factory=list)
    failure: Failure | None = None
    human_overrides: list[str] = Field(default_factory=list)
