"""
Build master analysis panel — merges financial data with both scoring frameworks.

Inputs:
  data/processed/financial_panel.csv    — quarterly financials
  data/processed/lit_scores_master.csv  — Framework B: C1-C10 rubric scores [1,100]
  data/processed/lit_scores_task.csv    — Framework A: Eloundou task scores  [1,100]
  data/raw/firm_universe.csv            — firm metadata

Output:
  data/processed/master_panel.csv

Key columns added:
  post         — 1 if period_end >= 2022-10-01
  ln_revenue   — log(revenue)
  score_b      — Framework B rubric score (primary)
  score_a      — Framework A task score
  sme_200_m1   — avg annual revenue < $200M (pre-shock)
  sme_200_m2   — avg quarterly revenue < $50M (pre-shock)
  sme_500      — avg quarterly revenue < $125M (pre-shock)
  b2b          — 1 for all (SIC 7370-7379 = B2B software)
  pre_shock_revenue — mean quarterly revenue pre-shock (USD)
"""

import os, math
import pandas as pd

SHOCK_DATE = "2022-10-01"

fin  = pd.read_csv("data/processed/financial_panel.csv")

# New API scores only (B2B firms, consistent calibration)
sb_new = pd.read_csv("data/processed/lit_scores.csv")[["ticker","normalized_score"]]
sb = sb_new[sb_new["normalized_score"].notna()].rename(columns={"normalized_score":"lit_score"})

sa   = pd.read_csv("data/processed/lit_scores_task.csv")[["ticker","task_score"]]
fu   = pd.read_csv("data/raw/firm_universe.csv")[["ticker","company_name","sic"]]

fin["period_end"] = pd.to_datetime(fin["period_end"])

# ── Pre-shock stats ───────────────────────────────────────────────────────────
pre = fin[fin["period_end"] < SHOCK_DATE].copy()
pre_q  = pre.groupby("ticker")["revenue"].mean().rename("pre_shock_revenue")
pre_nq = pre.groupby("ticker")["revenue"].count().rename("pre_shock_quarters")
pre_ann = (pre.groupby(["ticker","fiscal_year"])["revenue"]
             .sum()
             .groupby("ticker").mean()
             .rename("pre_shock_annual"))

# ── Merge ─────────────────────────────────────────────────────────────────────
panel = (fin
    .merge(sb.rename(columns={"lit_score":"score_b"}), on="ticker", how="inner")
    .merge(sa.rename(columns={"task_score":"score_a"}), on="ticker", how="left")
    .merge(pre_q,   on="ticker", how="left")
    .merge(pre_nq,  on="ticker", how="left")
    .merge(pre_ann, on="ticker", how="left")
    .merge(fu,      on="ticker", how="left")
)

# ── Derived variables ─────────────────────────────────────────────────────────
panel["post"]       = (panel["period_end"] >= SHOCK_DATE).astype(int)
panel               = panel[panel["revenue"] > 0].copy()
panel["ln_revenue"] = panel["revenue"].apply(math.log)

# SME flags
panel["sme_200_m1"] = (panel["pre_shock_annual"]  < 200_000_000).astype(int)
panel["sme_200_m2"] = (panel["pre_shock_revenue"] <  50_000_000).astype(int)
panel["sme_500"]    = (panel["pre_shock_revenue"] < 125_000_000).astype(int)

# B2B = all (SIC 7370-7379 software)
panel["b2b"] = 1

# Financial ratios (winsorized)
def winsor(s, lo, hi): return s.clip(lower=lo, upper=hi)
panel["gross_margin"]     = winsor(panel["gross_profit"]     / panel["revenue"], -1, 1)
panel["operating_margin"] = winsor(panel["operating_income"] / panel["revenue"], -2, 2)
panel["rd_intensity"]     = winsor(panel["rd_expense"]       / panel["revenue"],  0, 5)
panel["sga_intensity"]    = winsor(panel["sga_expense"]      / panel["revenue"],  0, 5)

# year_quarter label
panel["year_quarter"] = panel["fiscal_year"].astype(str) + "Q" + panel["fiscal_quarter"].astype(str)
panel["pre_rev_m"]    = panel["pre_shock_revenue"] / 1e6

panel = panel.sort_values(["ticker","period_end"]).reset_index(drop=True)
panel.to_csv("data/processed/master_panel.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"Master panel: {len(panel):,} obs, {panel['ticker'].nunique()} firms")
for label, mask in [
    ("Framework B scored",    panel["score_b"].notna()),
    ("Framework A scored",    panel["score_a"].notna()),
    ("SME200 M1 (ann<200M)",  panel["sme_200_m1"]==1),
    ("SME200 M2 (q<50M)",     panel["sme_200_m2"]==1),
]:
    sub = panel[mask]
    print(f"  {label:<26}: {sub['ticker'].nunique()} firms, {len(sub):,} obs")

print("\nSME200 M1 score_b distribution:")
s = panel[panel["sme_200_m1"]==1].drop_duplicates("ticker")["score_b"]
print(f"  n={len(s)}, mean={s.mean():.1f}, std={s.std():.1f}")

print("\nSME200 M2 score_b distribution:")
s2 = panel[panel["sme_200_m2"]==1].drop_duplicates("ticker")["score_b"]
print(f"  n={len(s2)}, mean={s2.mean():.1f}, std={s2.std():.1f}")
