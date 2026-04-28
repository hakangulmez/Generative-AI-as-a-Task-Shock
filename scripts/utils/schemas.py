"""
Pydantic schemas for Phase 4 scoring outputs.

These schemas serve two purposes:
  1. JSON schema generation via `.model_json_schema()` — fed to Claude's
     tool_use API as the input_schema for forced structured output.
  2. Response validation in llm_client.py — invalid responses trigger retry.

Supply (SupplyScore): the model returns only ticker, tasks, and
overall_reasoning. Aggregate counts and rho score are computed
post-hoc by compute_aggregates(), which is the single source of truth
for the formula: ρ = (R1 + 0.5×R2) / n_tasks × 99 + 1.

Demand (DemandScore): the model returns sub-scores and composite. The
validators enforce the 0.1-grid and composite formula:
  - Each sub-score must be on the 0.0, 0.1, 0.2, ..., 1.0 grid
  - Composite must equal (delta_switch + delta_error + delta_data) / 3

Mismatches raise pydantic.ValidationError → retryable failure in
llm_client.py. Final failure raises LLMScoringError.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ProductTaskWithRubric(BaseModel):
    """One software task with R0/R1/R2 classification and reasoning."""
    task: str = Field(
        ...,
        min_length=30,
        max_length=300,
        description="Software task description — what the product itself does (1-2 sentences)",
    )
    r_label: Literal["R0", "R1", "R2"] = Field(
        ...,
        description="R0 = outside LLM scope, R1 = direct LLM substitute, R2 = LLM + tools substitute",
    )
    reasoning: str = Field(
        ...,
        min_length=30,
        max_length=350,
        description="Justification for the R-label (1-3 sentences citing decision heuristics; keep concise)",
    )


class SupplyScore(BaseModel):
    """Supply-side LLM replicability score for one firm using R-rubric.

    Methodology: PHASE4_METHODOLOGY_v3.md Section 5.
    Aggregation: ρ_i = (R1_count + 0.5 × R2_count) / n_tasks × 99 + 1

    Only `ticker`, `tasks`, and `overall_reasoning` are model-reported.
    All numeric aggregates (r0_count, r1_count, r2_count, n_tasks,
    raw_exposure, normalized_score) are computed by the
    `compute_aggregates()` helper from `tasks` after validation.

    This separation prevents the model from reporting counts that
    disagree with its own task labels — a self-consistency failure
    observed in Step 6a smoke v2 with ZS (10 tasks, R0/R2
    ambiguous boundary cases).
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    tasks: list[ProductTaskWithRubric] = Field(..., min_length=6, max_length=12)
    overall_reasoning: str = Field(
        ...,
        min_length=150,
        max_length=2500,
        description="2-3 sentences describing what the product does and what determines its R-distribution",
    )

    # No model_validator — there is nothing to reconcile.
    # Counts and rho are computed by compute_aggregates() below.


def compute_aggregates(score: SupplyScore) -> dict:
    """Compute aggregate counts and rho score from a validated SupplyScore.

    Returns a dict with the same keys downstream code expects:
        r0_count, r1_count, r2_count, n_tasks,
        raw_exposure, normalized_score

    Formula (Eloundou 2024 β, methodology v3 Section 5):
        raw_exposure     = (r1_count + 0.5 * r2_count) / n_tasks
        normalized_score = round(raw_exposure * 99 + 1, 1)

    This is the single source of truth for rho computation. Any caller
    that needs aggregates must use this function — do NOT hand-compute
    them elsewhere.
    """
    r0_count = sum(1 for t in score.tasks if t.r_label == "R0")
    r1_count = sum(1 for t in score.tasks if t.r_label == "R1")
    r2_count = sum(1 for t in score.tasks if t.r_label == "R2")
    n_tasks = len(score.tasks)

    assert r0_count + r1_count + r2_count == n_tasks, (
        f"task labels malformed for {score.ticker}: "
        f"{r0_count}+{r1_count}+{r2_count} != {n_tasks}"
    )

    raw_exposure = (r1_count + 0.5 * r2_count) / n_tasks
    normalized_score = round(raw_exposure * 99 + 1, 1)

    return {
        "r0_count": r0_count,
        "r1_count": r1_count,
        "r2_count": r2_count,
        "n_tasks": n_tasks,
        "raw_exposure": round(raw_exposure, 4),
        "normalized_score": normalized_score,
    }


class DemandScore(BaseModel):
    """Demand-side friction score for one firm.

    Each sub-score must be on the 0.0, 0.1, 0.2, ..., 1.0 grid.
    Composite must equal (switch + error + data) / 3 within 0.005 tolerance.

    The 0.1 grid is enforced strictly because the prompt's 6-level rubric
    only defines anchor descriptors at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 — the
    odd deciles (0.1, 0.3, ...) are valid as intermediate judgments
    between adjacent rubric anchors.
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    delta_switch: float = Field(..., ge=0.0, le=1.0)
    delta_error: float = Field(..., ge=0.0, le=1.0)
    delta_data: float = Field(..., ge=0.0, le=1.0)
    delta_composite: float = Field(..., ge=0.0, le=1.0)
    switching_cost_reasoning: str = Field(
        ...,
        min_length=80,
        max_length=1200,
        description="1-2 sentences citing specific product features, contract structures, or integration depth from the 10-K",
    )
    error_cost_reasoning: str = Field(
        ...,
        min_length=80,
        max_length=1200,
        description="1-2 sentences citing the regulated domain, liability exposure, or consequence severity from the 10-K",
    )
    data_moat_reasoning: str = Field(
        ...,
        min_length=80,
        max_length=1200,
        description="1-2 sentences citing specific proprietary data assets, network size, or feedback mechanisms from the 10-K",
    )

    @field_validator("delta_switch", "delta_error", "delta_data")
    @classmethod
    def check_decile_grid(cls, v: float) -> float:
        """Sub-score must be on the 0.0, 0.1, 0.2, ..., 1.0 grid.

        Tolerance: 1e-6 (floating-point safety only — values like 0.65 or
        0.75 are explicitly rejected, not silently rounded).
        """
        scaled = round(v * 10)
        if abs(v * 10 - scaled) > 1e-6:
            raise ValueError(
                f"Sub-score {v} not on 0.1 grid. "
                f"Allowed: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0"
            )
        return v

    @model_validator(mode="after")
    def check_composite_formula(self):
        """STRICT MODE: delta_composite must equal (switch + error + data) / 3.

        Rounding: prompt specifies 3 decimal places.
        Tolerance: 0.005 (covers half a unit at the 3rd decimal).
        """
        formula = round(
            (self.delta_switch + self.delta_error + self.delta_data) / 3, 3
        )
        if abs(self.delta_composite - formula) > 0.005:
            raise ValueError(
                f"delta_composite reconciliation failed for {self.ticker}: "
                f"reported={self.delta_composite} but formula "
                f"({self.delta_switch} + {self.delta_error} + {self.delta_data}) / 3 "
                f"= {formula}"
            )
        return self


if __name__ == "__main__":
    # ============================================================
    # Supply: valid input — model returns only ticker/tasks/overall_reasoning
    # ============================================================
    valid_supply = {
        "ticker": "ZS",
        "tasks": [
            {
                "task": "real-time traffic inspection across customer network",
                "r_label": "R0",
                "reasoning": "Requires sub-second per-operation latency at line rate; LLM inference cannot meet this SLA.",
            },
            {
                "task": "zero-trust policy enforcement at packet level",
                "r_label": "R0",
                "reasoning": "Packet-level processing with microsecond response times is outside LLM inference economics.",
            },
            {
                "task": "threat intelligence aggregation from live feeds",
                "r_label": "R0",
                "reasoning": "Continuous real-time stream processing at line rate is a fundamental LLM operational limit.",
            },
            {
                "task": "certificate management for client devices",
                "r_label": "R0",
                "reasoning": "Cryptographic operations require deterministic computation that LLMs cannot reliably provide.",
            },
            {
                "task": "network access control decisions per session",
                "r_label": "R0",
                "reasoning": "Sub-millisecond per-session policy decisions at scale are outside LLM operational scope.",
            },
            {
                "task": "admin portal management and compliance reporting",
                "r_label": "R2",
                "reasoning": "SQL queries plus LLM synthesis can generate compliance reports from structured admin data.",
            },
        ],
        "overall_reasoning": (
            "Zscaler's product is a real-time security infrastructure where the core "
            "tasks operate on live network traffic streams and policy enforcement "
            "happens at the packet level with sub-millisecond latency requirements. "
            "Most tasks are R0 due to streaming-data and latency dependencies that "
            "LLMs cannot replicate at operational scale."
        ),
    }
    s = SupplyScore.model_validate(valid_supply)
    aggregates = compute_aggregates(s)
    assert aggregates["r0_count"] == 5
    assert aggregates["r2_count"] == 1
    assert aggregates["n_tasks"] == 6
    assert aggregates["normalized_score"] == 9.2  # (0 + 0.5)/6 * 99 + 1
    print(f"OK supply valid: {s.ticker} score={aggregates['normalized_score']} n={aggregates['n_tasks']}")

    # ============================================================
    # Supply: short per-task reasoning should fail (min_length 30)
    # ============================================================
    bad_tasks = [dict(t) for t in valid_supply["tasks"]]
    bad_tasks[0] = dict(bad_tasks[0])
    bad_tasks[0]["reasoning"] = "Too fast."  # 9 chars < 30
    bad_supply_per_task = dict(valid_supply)
    bad_supply_per_task["tasks"] = bad_tasks
    try:
        SupplyScore.model_validate(bad_supply_per_task)
        print("FAIL: should have rejected short per-task reasoning")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 30" in msg or "string_too_short" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects short per-task reasoning")

    # ============================================================
    # Supply: short overall_reasoning should fail (min_length 150)
    # ============================================================
    bad_supply_reasoning = dict(valid_supply)
    bad_supply_reasoning["overall_reasoning"] = "Real-time security. Mostly R0."
    try:
        SupplyScore.model_validate(bad_supply_reasoning)
        print("FAIL: should have rejected short overall_reasoning")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 150" in msg or "string_too_short" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects short overall_reasoning")

    # ============================================================
    # Supply: too few tasks (5) should fail (min_length 6)
    # ============================================================
    bad_supply_few = dict(valid_supply)
    bad_supply_few["tasks"] = valid_supply["tasks"][:5]
    try:
        SupplyScore.model_validate(bad_supply_few)
        print("FAIL: should have rejected 5 tasks")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 6" in msg or "too_short" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects too few tasks")

    # ============================================================
    # Demand: valid input (VEEV-like, FDA regulated)
    # ============================================================
    valid_demand = {
        "ticker": "VEEV",
        "delta_switch": 1.0, "delta_error": 1.0, "delta_data": 0.6,
        "delta_composite": 0.867,  # round((1.0+1.0+0.6)/3, 3)
        "switching_cost_reasoning": (
            "Veeva is embedded in FDA-validated pharma workflows; switching "
            "requires re-validating every clinical trial and regulatory "
            "submission system, effectively impossible during active trials."
        ),
        "error_cost_reasoning": (
            "Errors carry FDA penalties and patient safety implications; "
            "human oversight is mandatory by 21 CFR Part 11 regulations on "
            "electronic records in regulated industries."
        ),
        "data_moat_reasoning": (
            "Accumulated commercial and clinical data within the pharma "
            "context provides a moderate moat, but customer ownership of "
            "their own data limits exclusivity to the platform."
        ),
    }
    d = DemandScore.model_validate(valid_demand)
    print(f"OK demand valid: {d.ticker} composite={d.delta_composite}")

    # ============================================================
    # Demand: off-grid sub-score should fail (0.65 not on 0.1 grid)
    # ============================================================
    bad_demand_grid = dict(valid_demand)
    bad_demand_grid["delta_switch"] = 0.65
    bad_demand_grid["delta_composite"] = round((0.65 + 1.0 + 0.6) / 3, 3)
    try:
        DemandScore.model_validate(bad_demand_grid)
        print("FAIL: should have rejected off-grid sub-score")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "not on 0.1 grid" in msg, f"unexpected error: {msg[:200]}"
        print("OK demand rejects off-grid sub-score")

    # ============================================================
    # Demand: composite mismatch should fail
    # ============================================================
    bad_demand_comp = dict(valid_demand)
    bad_demand_comp["delta_composite"] = 0.500  # wrong: should be 0.867
    try:
        DemandScore.model_validate(bad_demand_comp)
        print("FAIL: should have rejected composite mismatch")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "composite reconciliation failed" in msg, f"unexpected error: {msg[:200]}"
        print("OK demand rejects composite mismatch")

    # ============================================================
    # Demand: short reasoning should fail (min_length 80)
    # ============================================================
    bad_demand_reasoning = dict(valid_demand)
    bad_demand_reasoning["switching_cost_reasoning"] = "FDA regulated, hard to switch."
    try:
        DemandScore.model_validate(bad_demand_reasoning)
        print("FAIL: should have rejected short reasoning")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 80" in msg or "string_too_short" in msg, f"unexpected error: {msg[:200]}"
        print("OK demand rejects short reasoning")

    print()
    print("All schema self-tests passed.")
