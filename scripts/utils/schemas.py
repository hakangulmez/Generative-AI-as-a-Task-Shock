"""
Pydantic schemas for Phase 4 scoring outputs.

These schemas serve two purposes:
  1. JSON schema generation via `.model_json_schema()` — fed to Claude's
     tool_use API as the input_schema for forced structured output.
  2. Response validation in llm_client.py — invalid responses trigger retry.

The validators enforce the methodology decisions made in Phase 3:
  - Supply: normalized_score must equal (E1 + 0.5*E2)/n_tasks * 99 + 1
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


class TaskClassification(BaseModel):
    """One customer-facing product task with its E0/E1/E2 classification."""
    task: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="Customer-facing task description (what the customer uses the product to accomplish)",
    )
    exposure: Literal["E0", "E1", "E2"] = Field(
        ...,
        description="E0 = no LLM exposure, E1 = direct LLM exposure, E2 = LLM + standard tools",
    )


class SupplyScore(BaseModel):
    """Supply-side LLM replicability score for one firm.

    Formula reconciliation: normalized_score must satisfy
        normalized_score == round((e1_count + 0.5*e2_count) / n_tasks * 99 + 1, 1)
    within tolerance of 1.0 (allows for the model rounding raw_exposure).
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    tasks: list[TaskClassification] = Field(..., min_length=6, max_length=12)
    e0_count: int = Field(..., ge=0, le=12)
    e1_count: int = Field(..., ge=0, le=12)
    e2_count: int = Field(..., ge=0, le=12)
    n_tasks: int = Field(..., ge=6, le=12)
    raw_exposure: float = Field(..., ge=0.0, le=1.0)
    normalized_score: float = Field(..., ge=1.0, le=100.0)
    reasoning: str = Field(
        ...,
        min_length=150,
        max_length=2500,
        description="2-3 sentences describing what the product does and what determines its exposure profile",
    )

    @model_validator(mode="after")
    def check_task_counts(self):
        """n_tasks must equal len(tasks) and equal e0_count + e1_count + e2_count.
        Per-exposure counts must match the actual classifications in `tasks`."""
        actual_e0 = sum(1 for t in self.tasks if t.exposure == "E0")
        actual_e1 = sum(1 for t in self.tasks if t.exposure == "E1")
        actual_e2 = sum(1 for t in self.tasks if t.exposure == "E2")

        if self.n_tasks != len(self.tasks):
            raise ValueError(
                f"n_tasks={self.n_tasks} but len(tasks)={len(self.tasks)}"
            )
        if self.e0_count != actual_e0:
            raise ValueError(
                f"e0_count={self.e0_count} but tasks contain {actual_e0} E0 entries"
            )
        if self.e1_count != actual_e1:
            raise ValueError(
                f"e1_count={self.e1_count} but tasks contain {actual_e1} E1 entries"
            )
        if self.e2_count != actual_e2:
            raise ValueError(
                f"e2_count={self.e2_count} but tasks contain {actual_e2} E2 entries"
            )
        if self.n_tasks != self.e0_count + self.e1_count + self.e2_count:
            raise ValueError(
                f"n_tasks={self.n_tasks} but e0+e1+e2={self.e0_count + self.e1_count + self.e2_count}"
            )
        return self

    @model_validator(mode="after")
    def check_formula_reconciliation(self):
        """STRICT MODE: model-reported normalized_score must match the formula.

        formula:    raw_exposure     = (e1 + 0.5 * e2) / n_tasks
                    normalized_score = round(raw_exposure * 99 + 1, 1)

        Tolerances:
          raw_exposure:     0.02  (covers rounding to 3-4 decimals in JSON)
          normalized_score: 1.0   (covers compounding rounding in the final formula)

        Violation indicates the model manually adjusted the score, which is
        explicitly forbidden by the prompt and by Phase 3 design intent.
        """
        formula_raw = (self.e1_count + 0.5 * self.e2_count) / self.n_tasks
        formula_normalized = round(formula_raw * 99 + 1, 1)

        if abs(self.raw_exposure - formula_raw) > 0.02:
            raise ValueError(
                f"raw_exposure reconciliation failed for {self.ticker}: "
                f"reported={self.raw_exposure} but "
                f"(e1={self.e1_count} + 0.5*e2={self.e2_count}) / n={self.n_tasks} "
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
    # Supply: valid input (ZS-like, real-time security E0-dominant)
    # ============================================================
    valid_supply = {
        "ticker": "ZS",
        "tasks": [
            {"task": "real-time traffic inspection across customer network", "exposure": "E0"},
            {"task": "zero-trust policy enforcement at packet level", "exposure": "E0"},
            {"task": "threat intelligence aggregation from live feeds", "exposure": "E0"},
            {"task": "certificate management for client devices", "exposure": "E0"},
            {"task": "network access control decisions per session", "exposure": "E0"},
            {"task": "admin portal management and reporting", "exposure": "E2"},
        ],
        "e0_count": 5, "e1_count": 0, "e2_count": 1, "n_tasks": 6,
        "raw_exposure": 0.0833,
        "normalized_score": 9.2,  # round(0.0833 * 99 + 1, 1) = 9.2
        "reasoning": (
            "Zscaler's product is a real-time security infrastructure where "
            "the core tasks operate on live network traffic streams and "
            "policy enforcement happens at the packet level with sub-millisecond "
            "latency requirements. Most tasks are E0 due to the streaming-data "
            "and physical-network dependencies that LLMs cannot replicate."
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
        print(f"OK supply rejects formula mismatch")

    # ============================================================
    # Supply: count mismatch should fail
    # ============================================================
    bad_supply_count = dict(valid_supply)
    bad_supply_count["e0_count"] = 4  # but tasks contain 5 E0 entries
    bad_supply_count["e1_count"] = 1  # adjust to keep n_tasks consistent
    try:
        SupplyScore.model_validate(bad_supply_count)
        print("FAIL: should have rejected count mismatch")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "tasks contain" in msg, f"unexpected error: {msg[:200]}"
        print(f"OK supply rejects count mismatch")

    # ============================================================
    # Supply: short reasoning should fail (min_length 150)
    # ============================================================
    bad_supply_reasoning = dict(valid_supply)
    bad_supply_reasoning["reasoning"] = "Real-time security. Mostly E0."
    try:
        SupplyScore.model_validate(bad_supply_reasoning)
        print("FAIL: should have rejected short reasoning")
        raise SystemExit(1)
    except Exception as e:
        msg = str(e)
        assert "at least 150" in msg or "string_too_short" in msg, f"unexpected error: {msg[:200]}"
        print(f"OK supply rejects short reasoning")

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
        print(f"OK demand rejects off-grid sub-score")

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
        print(f"OK demand rejects composite mismatch")

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
        print(f"OK demand rejects short reasoning")

    print()
    print("All schema self-tests passed.")
