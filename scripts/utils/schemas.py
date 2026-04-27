"""
Pydantic schemas for Phase 4 scoring outputs.

These schemas serve two purposes:
  1. JSON schema generation via `.model_json_schema()` — fed to Claude's
     tool_use API as the input_schema for forced structured output.
  2. Response validation in llm_client.py — invalid responses trigger retry.

The validators enforce the methodology decisions made in Phase 3 (demand)
and Phase 4 v3.1 (supply, R-rubric):
  - Supply: normalized_score must equal (R1 + 0.5*R2)/n_tasks * 99 + 1
  - Demand: each sub-score must be on the 0.0, 0.1, 0.2, ..., 1.0 grid
  - Demand: composite must equal (delta_switch + delta_error + delta_data) / 3

Mismatches raise pydantic.ValidationError. The client treats this as a
retryable failure (the model gets a second chance to recompute), and on
final failure raises LLMScoringError. This is the strict formula
reconciliation policy: the model is not permitted to manually adjust
scores away from the formula.
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
        max_length=200,
        description="Brief justification for the R-label (1-2 sentences referencing decision heuristics)",
    )


class SupplyScore(BaseModel):
    """Supply-side LLM replicability score for one firm using R-rubric.

    Methodology: PHASE4_METHODOLOGY_v3.md Section 5.
    Aggregation: ρ_i = (R1_count + 0.5 × R2_count) / n_tasks × 99 + 1
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    tasks: list[ProductTaskWithRubric] = Field(..., min_length=6, max_length=12)
    r0_count: int = Field(..., ge=0, le=12)
    r1_count: int = Field(..., ge=0, le=12)
    r2_count: int = Field(..., ge=0, le=12)
    n_tasks: int = Field(..., ge=6, le=12)
    raw_exposure: float = Field(..., ge=0.0, le=1.0)
    normalized_score: float = Field(..., ge=1.0, le=100.0)
    overall_reasoning: str = Field(
        ...,
        min_length=150,
        max_length=2500,
        description="2-3 sentences describing what the product does and what determines its R-distribution",
    )

    @model_validator(mode="after")
    def check_task_counts(self):
        """n_tasks must equal len(tasks) and equal r0_count + r1_count + r2_count.
        Per-rubric counts must match the actual classifications in `tasks`."""
        actual_r0 = sum(1 for t in self.tasks if t.r_label == "R0")
        actual_r1 = sum(1 for t in self.tasks if t.r_label == "R1")
        actual_r2 = sum(1 for t in self.tasks if t.r_label == "R2")

        if self.n_tasks != len(self.tasks):
            raise ValueError(f"n_tasks={self.n_tasks} but len(tasks)={len(self.tasks)}")
        if self.r0_count != actual_r0:
            raise ValueError(f"r0_count={self.r0_count} but tasks contain {actual_r0} R0 entries")
        if self.r1_count != actual_r1:
            raise ValueError(f"r1_count={self.r1_count} but tasks contain {actual_r1} R1 entries")
        if self.r2_count != actual_r2:
            raise ValueError(f"r2_count={self.r2_count} but tasks contain {actual_r2} R2 entries")
        if self.n_tasks != self.r0_count + self.r1_count + self.r2_count:
            raise ValueError(
                f"n_tasks={self.n_tasks} but r0+r1+r2={self.r0_count + self.r1_count + self.r2_count}"
            )
        return self

    @model_validator(mode="after")
    def check_formula_reconciliation(self):
        """STRICT MODE: model-reported normalized_score must match the formula.

        formula:    raw_exposure     = (r1 + 0.5 * r2) / n_tasks
                    normalized_score = round(raw_exposure * 99 + 1, 1)

        Tolerances:
          raw_exposure:     0.02  (covers rounding to 3-4 decimals in JSON)
          normalized_score: 1.0   (covers compounding rounding in the final formula)

        Violation indicates the model manually adjusted the score, which is
        explicitly forbidden by the prompt and by Phase 4 v3.1 design intent.
        """
        formula_raw = (self.r1_count + 0.5 * self.r2_count) / self.n_tasks
        formula_normalized = round(formula_raw * 99 + 1, 1)

        if abs(self.raw_exposure - formula_raw) > 0.02:
            raise ValueError(
                f"raw_exposure reconciliation failed for {self.ticker}: "
                f"reported={self.raw_exposure} but "
                f"(r1={self.r1_count} + 0.5*r2={self.r2_count}) / n={self.n_tasks} "
                f"= {formula_raw:.4f}"
            )

        if abs(self.normalized_score - formula_normalized) > 1.0:
            raise ValueError(
                f"normalized_score reconciliation failed for {self.ticker}: "
                f"reported={self.normalized_score} but formula gives {formula_normalized}. "
                f"The model may have manually adjusted the score — re-score required."
            )
        return self


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
    # Supply: valid input (ZS-like, R0-dominant: 5 R0 + 1 R2)
    # raw_exposure = 0.5/6 = 0.0833; normalized = round(0.0833*99+1, 1) = 9.2
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
        "r0_count": 5, "r1_count": 0, "r2_count": 1, "n_tasks": 6,
        "raw_exposure": 0.0833,
        "normalized_score": 9.2,
        "overall_reasoning": (
            "Zscaler's product is a real-time security infrastructure where "
            "the core tasks operate on live network traffic streams and "
            "policy enforcement happens at the packet level with sub-millisecond "
            "latency requirements. Most tasks are R0 due to streaming-data "
            "and latency dependencies that LLMs cannot replicate at operational scale."
        ),
    }
    s = SupplyScore.model_validate(valid_supply)
    print(f"OK supply valid: {s.ticker} score={s.normalized_score} n={s.n_tasks}")

    # ============================================================
    # Supply: formula mismatch should fail (model "adjusted" score)
    # ============================================================
    bad_supply_score = dict(valid_supply)
    bad_supply_score["normalized_score"] = 50.0  # way off from 9.2
    try:
        SupplyScore.model_validate(bad_supply_score)
        print("FAIL: should have rejected formula mismatch")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "reconciliation failed" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects formula mismatch")

    # ============================================================
    # Supply: count mismatch should fail
    # r0_count=4 but tasks contain 5 R0 entries
    # ============================================================
    bad_supply_count = dict(valid_supply)
    bad_supply_count["r0_count"] = 4  # but tasks contain 5 R0 entries
    bad_supply_count["r1_count"] = 1  # adjust to keep n_tasks=6 sum consistent
    try:
        SupplyScore.model_validate(bad_supply_count)
        print("FAIL: should have rejected count mismatch")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "tasks contain" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects count mismatch")

    # ============================================================
    # Supply: short overall_reasoning should fail (min_length 150)
    # ============================================================
    bad_supply_reasoning = dict(valid_supply)
    bad_supply_reasoning["overall_reasoning"] = "Real-time security. Mostly R0."
    try:
        SupplyScore.model_validate(bad_supply_reasoning)
        print("FAIL: should have rejected short reasoning")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 150" in msg or "string_too_short" in msg, f"unexpected error: {msg[:200]}"
        print("OK supply rejects short reasoning")

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
