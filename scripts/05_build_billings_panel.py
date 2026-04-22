"""
Phase 2.8 — build billings panel from revenue + ΔRPO.

Billings captures total invoiced amounts including newly-contracted future
revenue. For SaaS firms it is often a leading indicator ahead of recognized
revenue.

Formula: billings_t = revenue_t + (rpo_t - rpo_{t-1})

Pure arithmetic — no EDGAR traffic. Requires both panels to be pre-built.

Input:   data/processed/financial_panel.csv   (filter metric == 'revenue')
         data/processed/rpo_quarterly.csv
Output:  data/processed/billings_panel.csv
         data/processed/billings_panel_qa.json

Columns: ticker, cik, period_end, fiscal_quarter, fiscal_year,
         revenue, rpo_value, rpo_delta, billings

Merge strategy: left-join RPO onto revenue rows. All revenue observations are
kept; rpo_value is NaN for firms/quarters with no RPO disclosure.

rpo_delta is gap-aware: consecutive diff within each firm sorted by period_end,
set to NaN when the gap between consecutive RPO observations exceeds 120 days
(annual-only reporters, missing quarters). This prevents multi-quarter RPO
swings from being assigned to a single quarter. The first observation per firm
is always NaN (no prior period to diff against).

billings is NaN wherever rpo_delta is NaN. Revenue is never NaN.

No coverage filter is applied at build time. Inclusion thresholds
(e.g. n_quarters_billings >= 6) are applied at analysis time in R.

Negative billings (rpo_delta < 0) are valid: they reflect customer churn or
contract restructuring. Kept as-is.

Usage:
    python3 scripts/05_build_billings_panel.py
    python3 scripts/05_build_billings_panel.py --tickers ZS CRM MSFT
    python3 scripts/05_build_billings_panel.py --output data/processed/billings_test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FINANCIAL = Path("data/processed/financial_panel.csv")
INPUT_RPO       = Path("data/processed/rpo_quarterly.csv")
OUTPUT_PATH     = Path("data/processed/billings_panel.csv")
QA_PATH         = Path("data/processed/billings_panel_qa.json")

GAP_THRESHOLD_DAYS = 120  # diffs > this are treated as non-consecutive quarters
MIN_BILLINGS_QA    = 6    # threshold used only for QA status labelling


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.8: build billings panel (revenue + ΔRPO)"
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Process specific tickers only (updates their rows in existing output)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH), metavar="PATH",
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    out_path = Path(args.output)

    # --- Load inputs ---
    fin_df = pd.read_csv(INPUT_FINANCIAL)
    rpo_df = pd.read_csv(INPUT_RPO)

    rev_df = (
        fin_df[fin_df["metric"] == "revenue"]
        [["ticker", "cik", "period_end", "fiscal_quarter", "fiscal_year", "value"]]
        .rename(columns={"value": "revenue"})
        .copy()
    )

    rpo_slim = rpo_df[["ticker", "period_end", "rpo_value"]].copy()

    if args.tickers:
        requested = {t.upper() for t in args.tickers}
        rev_df   = rev_df[rev_df["ticker"].isin(requested)]
        rpo_slim = rpo_slim[rpo_slim["ticker"].isin(requested)]
        missing_in_rev = requested - set(rev_df["ticker"].unique())
        missing_in_rpo = requested - set(rpo_slim["ticker"].unique())
        if missing_in_rev:
            print(f"WARNING: no revenue data for: {', '.join(sorted(missing_in_rev))}")
        if missing_in_rpo:
            print(f"INFO: no RPO data for (will have NaN billings): {', '.join(sorted(missing_in_rpo))}")

    print(f"Revenue observations      : {len(rev_df):,}  ({rev_df['ticker'].nunique()} firms)")
    print(f"RPO observations          : {len(rpo_slim):,}  ({rpo_slim['ticker'].nunique()} firms)")
    print()

    # --- Left-join RPO onto revenue rows ---
    merged = rev_df.merge(rpo_slim, on=["ticker", "period_end"], how="left")
    merged["period_end"] = pd.to_datetime(merged["period_end"])
    merged = merged.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    # --- Gap-aware rpo_delta ---
    # period_diff_days: consecutive gap within each firm's RPO sequence.
    # Computed on the full merged frame; NaN for first row per firm.
    merged["_period_diff_days"] = (
        merged.groupby("ticker")["period_end"]
        .diff()
        .dt.days
    )

    merged["rpo_delta"] = merged.groupby("ticker")["rpo_value"].diff()

    # Null out diffs that cross a gap (including first row, where _period_diff_days is NaN)
    gap_mask = merged["_period_diff_days"].isna() | (merged["_period_diff_days"] > GAP_THRESHOLD_DAYS)
    merged.loc[gap_mask, "rpo_delta"] = float("nan")

    merged.drop(columns=["_period_diff_days"], inplace=True)

    # --- Billings ---
    merged["billings"] = merged["revenue"] + merged["rpo_delta"]

    # --- QA log ---
    qa_records: list[dict] = []

    rev_counts = rev_df.groupby("ticker").size().to_dict()
    rpo_counts = rpo_slim.groupby("ticker").size().to_dict()

    all_tickers = sorted(merged["ticker"].unique())

    for ticker in all_tickers:
        cik_vals = rev_df[rev_df["ticker"] == ticker]["cik"].values
        cik = int(cik_vals[0]) if len(cik_vals) > 0 else None

        firm_rows = merged[merged["ticker"] == ticker]
        n_rev     = rev_counts.get(ticker, 0)
        n_rpo     = int(firm_rows["rpo_value"].notna().sum())
        n_bill    = int(firm_rows["billings"].notna().sum())

        # Count gaps: consecutive RPO pairs where diff > threshold
        rpo_periods = firm_rows.dropna(subset=["rpo_value"])["period_end"].sort_values()
        if len(rpo_periods) >= 2:
            rpo_diffs = rpo_periods.diff().dt.days.dropna()
            n_gaps = int((rpo_diffs > GAP_THRESHOLD_DAYS).sum())
        else:
            n_gaps = 0

        if n_bill >= MIN_BILLINGS_QA:
            status = "billings_computable"
        elif n_bill >= 1:
            status = "billings_sparse"
        else:
            status = "no_billings"

        qa_records.append({
            "ticker":                ticker,
            "cik":                   cik,
            "n_quarters_revenue":    n_rev,
            "n_quarters_rpo":        n_rpo,
            "n_quarters_billings":   n_bill,
            "n_gaps_detected":       n_gaps,
            "status":                status,
        })

    # --- Assemble output ---
    out_df = merged[
        ["ticker", "cik", "period_end", "fiscal_quarter", "fiscal_year",
         "revenue", "rpo_value", "rpo_delta", "billings"]
    ].copy()

    out_df["period_end"] = out_df["period_end"].dt.strftime("%Y-%m-%d")

    # Merge with existing output if --tickers
    if args.tickers and out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[~existing["ticker"].isin({t.upper() for t in args.tickers})]
        out_df   = pd.concat([existing, out_df], ignore_index=True)

    out_df = out_df.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa_records, indent=2), encoding="utf-8")

    # --- Validation report ---
    computable = [q for q in qa_records if q["status"] == "billings_computable"]
    sparse     = [q for q in qa_records if q["status"] == "billings_sparse"]
    no_bill    = [q for q in qa_records if q["status"] == "no_billings"]

    n_bill_rows    = int(out_df["billings"].notna().sum())
    n_total_rows   = len(out_df)

    print("=" * 60)
    print(f"Total firms               : {len(all_tickers)}")
    print(f"  billings_computable     : {len(computable)}  (≥{MIN_BILLINGS_QA} billings quarters)")
    print(f"  billings_sparse         : {len(sparse)}  (1–{MIN_BILLINGS_QA - 1} billings quarters)")
    print(f"  no_billings             : {len(no_bill)}  (0 computable billings quarters)")
    print()
    print(f"Total rows in output      : {n_total_rows:,}  (= revenue observations)")
    print(f"Non-NaN billings rows     : {n_bill_rows:,}")
    print(f"NaN billings rows         : {n_total_rows - n_bill_rows:,}")
    print()

    # rpo_delta distribution (non-NaN only)
    rpo_delta_valid = out_df["rpo_delta"].dropna()
    if not rpo_delta_valid.empty:
        n_neg  = (rpo_delta_valid < 0).sum()
        n_zero = (rpo_delta_valid == 0).sum()
        n_pos  = (rpo_delta_valid > 0).sum()
        total  = len(rpo_delta_valid)
        print("rpo_delta distribution (non-NaN):")
        print(f"  negative : {n_neg:>5} ({100*n_neg/total:.1f}%)")
        print(f"  zero     : {n_zero:>5} ({100*n_zero/total:.1f}%)")
        print(f"  positive : {n_pos:>5} ({100*n_pos/total:.1f}%)")
        print()

    # ZS spot check
    if "ZS" in out_df["ticker"].values:
        zs = out_df[out_df["ticker"] == "ZS"].sort_values("period_end")
        shock_qtrs = zs[zs["period_end"].between("2022-10-01", "2023-06-30")]
        if not shock_qtrs.empty:
            print("ZS billings around shock (2022Q4 – 2023Q2):")
            print(shock_qtrs[
                ["period_end", "fiscal_quarter", "fiscal_year",
                 "revenue", "rpo_delta", "billings"]
            ].to_string(index=False))
        else:
            print("ZS: no rows in 2022Q4–2023Q2 window")
        print()

    print(f"Output written to         : {out_path}")
    print(f"QA log written to         : {QA_PATH}")


if __name__ == "__main__":
    main()
