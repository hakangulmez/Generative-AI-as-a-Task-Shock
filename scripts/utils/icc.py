"""Intraclass Correlation Coefficient (ICC) computation for reliability validation.

Used in Phase 4 Step 6 to assess inter-iteration reliability of LLM-based
supply-side scoring on the 15-firm anchor sample.

Reference: Koo & Li (2016), "A Guideline of Selecting and Reporting Intraclass
Correlation Coefficients for Reliability Research." Journal of Chiropractic
Medicine 15(2): 155-163.

ICC formulation: ICC(3,1) — two-way mixed-effects model, single rater, absolute
agreement. Treats firms as random and iterations as fixed (since iterations are
exchangeable repeated measures of the same firm).

Decision thresholds (per PHASE4_METHODOLOGY_v3.md Section 5.6):
    ICC >= 0.90      excellent → proceed single-run full sample
    0.75 <= ICC < 0.90  good  → proceed multi-run full sample
    ICC < 0.75       poor    → reject; methodology requires refinement

Methodology note: We report ICC(3,1) as the primary statistic. Confidence
intervals are reported via F-distribution (per Koo & Li 2016 formulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ICCResult:
    """Result of an ICC(3,1) computation on one score series."""
    icc: float                     # Point estimate
    ci_lower_95: float             # Lower bound of 95% CI
    ci_upper_95: float             # Upper bound of 95% CI
    n_subjects: int                # Number of firms (subjects)
    n_raters: int                  # Number of iterations (raters)
    f_statistic: float             # F from one-way ANOVA over subjects
    p_value: float                 # F-test p-value
    interpretation: str            # "excellent" | "good" | "moderate" | "poor"
    decision: str                  # "single-run" | "multi-run" | "reject"


def compute_icc_3_1(
    scores: np.ndarray,
) -> ICCResult:
    """Compute ICC(3,1) — two-way mixed-effects, single rater, absolute agreement.

    Parameters
    ----------
    scores : np.ndarray
        2D array of shape (n_subjects, n_raters). Each row is one firm,
        each column is one iteration's score. Must have >= 2 subjects and
        >= 2 raters. Cells must be non-NaN.

    Returns
    -------
    ICCResult
        Point estimate, 95% CI, F-statistic, p-value, interpretation,
        decision (per methodology v3 Section 5.6).

    Raises
    ------
    ValueError
        If shape is invalid, too few subjects/raters, or NaN cells present.

    Notes
    -----
    Formula (Shrout & Fleiss 1979, Case 3, single rater):
        ICC(3,1) = (BMS - EMS) / (BMS + (k-1) * EMS)
    where:
        BMS = Between-subjects mean square
        EMS = Error mean square (residual after removing subject and rater effects)
        k   = number of raters (iterations)

    For two-way mixed-effects single-rater absolute agreement, the formula is:
        ICC(3,1) = (BMS - EMS) / (BMS + (k-1) * EMS + (k/n) * (JMS - EMS))
    where:
        JMS = Between-raters (judges) mean square
        n   = number of subjects (firms)

    We implement the absolute-agreement form because iteration-level
    biases (e.g., systematic upward drift between iter1 and iter3) should
    count against reliability — not be partialled out.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {scores.shape}")
    n_subjects, n_raters = scores.shape
    if n_subjects < 2:
        raise ValueError(f"need >= 2 subjects, got {n_subjects}")
    if n_raters < 2:
        raise ValueError(f"need >= 2 raters, got {n_raters}")
    if np.isnan(scores).any():
        raise ValueError("scores contains NaN — drop incomplete rows first")

    # Mean squares (Shrout & Fleiss 1979 notation)
    grand_mean = scores.mean()
    subject_means = scores.mean(axis=1)  # shape (n_subjects,)
    rater_means = scores.mean(axis=0)    # shape (n_raters,)

    ss_between = n_raters * np.sum((subject_means - grand_mean) ** 2)
    df_between = n_subjects - 1
    ms_between = ss_between / df_between  # BMS

    ss_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
    df_raters = n_raters - 1
    ms_raters = ss_raters / df_raters  # JMS

    ss_total = np.sum((scores - grand_mean) ** 2)
    ss_error = ss_total - ss_between - ss_raters
    df_error = (n_subjects - 1) * (n_raters - 1)
    ms_error = ss_error / df_error  # EMS

    # ICC(3,1) — two-way mixed, absolute agreement, single rater
    numerator = ms_between - ms_error
    denominator = (
        ms_between
        + (n_raters - 1) * ms_error
        + (n_raters / n_subjects) * (ms_raters - ms_error)
    )
    icc = numerator / denominator if denominator > 0 else 0.0

    # F-statistic and p-value (test H0: ICC = 0)
    f_stat = ms_between / ms_error if ms_error > 0 else float("inf")
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_error) if ms_error > 0 else 0.0

    # 95% CI via F-distribution (Shrout & Fleiss 1979, eq. 4)
    f_lower = stats.f.ppf(0.025, df_between, df_error)
    f_upper = stats.f.ppf(0.975, df_between, df_error)

    fl = f_stat / f_upper
    fu = f_stat / f_lower

    ci_lower = (fl - 1) / (fl + (n_raters - 1)) if (fl + n_raters - 1) > 0 else 0.0
    ci_upper = (fu - 1) / (fu + (n_raters - 1)) if (fu + n_raters - 1) > 0 else 1.0

    # Interpretation per Koo & Li 2016
    if icc >= 0.90:
        interp = "excellent"
        decision = "single-run"
    elif icc >= 0.75:
        interp = "good"
        decision = "multi-run"
    elif icc >= 0.50:
        interp = "moderate"
        decision = "reject"
    else:
        interp = "poor"
        decision = "reject"

    return ICCResult(
        icc=round(icc, 4),
        ci_lower_95=round(ci_lower, 4),
        ci_upper_95=round(ci_upper, 4),
        n_subjects=n_subjects,
        n_raters=n_raters,
        f_statistic=round(f_stat, 4),
        p_value=round(p_value, 6),
        interpretation=interp,
        decision=decision,
    )


def icc_from_supply_csv(
    csv_path: Path,
    iterations: int = 3,
) -> ICCResult:
    """Convenience wrapper: load supply_rho.csv, extract iter scores, compute ICC.

    Parameters
    ----------
    csv_path : Path
        Path to supply_rho.csv (output of 08_score_supply_rho.py with --iterations >= 2).
    iterations : int
        Number of iterations to use (must be <= columns present). Default 3.

    Returns
    -------
    ICCResult

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if iterations < 2 or iterations > 3:
        raise ValueError(f"iterations must be 2 or 3, got {iterations}")

    df = pd.read_csv(csv_path)
    cols = [f"rho_iter{i}" for i in range(1, iterations + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Drop rows where any iteration is NaN (incomplete firms)
    subset = df[cols].dropna()
    if len(subset) < 2:
        raise ValueError(
            f"need >= 2 firms with all {iterations} iterations complete; "
            f"got {len(subset)} after dropna"
        )

    return compute_icc_3_1(subset.to_numpy())


# Self-test (only runs when invoked directly)
if __name__ == "__main__":
    # Synthetic case 1: high agreement (should report excellent)
    np.random.seed(42)
    n_firms, n_iter = 15, 3
    base = np.random.uniform(20, 80, n_firms)
    high_agreement = np.column_stack([
        base + np.random.normal(0, 1, n_firms),
        base + np.random.normal(0, 1, n_firms),
        base + np.random.normal(0, 1, n_firms),
    ])
    r = compute_icc_3_1(high_agreement)
    print(f"High-agreement synthetic: ICC={r.icc} [{r.ci_lower_95}, {r.ci_upper_95}], "
          f"interpretation={r.interpretation}, decision={r.decision}")
    assert r.interpretation in ("excellent", "good"), (
        f"high-agreement test should yield excellent or good, got {r.interpretation}"
    )

    # Synthetic case 2: low agreement (should report poor)
    np.random.seed(43)
    low_agreement = np.random.uniform(20, 80, (15, 3))  # all noise, no firm signal
    r = compute_icc_3_1(low_agreement)
    print(f"Low-agreement synthetic: ICC={r.icc} [{r.ci_lower_95}, {r.ci_upper_95}], "
          f"interpretation={r.interpretation}, decision={r.decision}")
    assert r.decision == "reject", f"low-agreement test should reject, got {r.decision}"

    # Synthetic case 3: perfect agreement (should be near 1.0)
    perfect = np.tile(base.reshape(-1, 1), (1, 3))
    r = compute_icc_3_1(perfect)
    print(f"Perfect agreement synthetic: ICC={r.icc}")
    assert r.icc > 0.99, f"perfect agreement should yield ICC ~ 1.0, got {r.icc}"

    print("All ICC self-tests passed.")
