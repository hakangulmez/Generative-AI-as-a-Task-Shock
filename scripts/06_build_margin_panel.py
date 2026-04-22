"""
Phase 2.9 — build margin and intensity panel from financial_panel.

Pure arithmetic — no EDGAR traffic. Requires financial_panel to be pre-built.

Input:   data/processed/financial_panel.csv
Output:  data/processed/margin_panel.csv
         data/processed/margin_panel_qa.json

Columns: ticker, cik, period_end, fiscal_quarter, fiscal_year, metric, value, value_winsorized

Metrics computed (5 ratios):
  gross_margin   = (revenue - cogs) / revenue
  rd_intensity   = rd / revenue
  sga_intensity  = sga / revenue
  sbc_intensity  = sbc / revenue
  opex_ratio     = (cogs + rd + sga) / revenue

Winsorization: P1/P99 clipping across full non-NaN distribution per metric.
NaN values remain NaN — not imputed, not dropped.
No firm-level exclusion at build time — all 321 firms present in output.

Usage:
    python3 scripts/06_build_margin_panel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FINANCIAL = Path("data/processed/financial_panel.csv")
OUTPUT_PATH     = Path("data/processed/margin_panel.csv")
QA_PATH         = Path("data/processed/margin_panel_qa.json")

SOURCE_METRICS = ["revenue", "cogs", "rd", "sga", "sbc"]
RATIO_METRICS  = ["gross_margin", "rd_intensity", "sga_intensity", "sbc_intensity", "opex_ratio"]
ID_COLS        = ["ticker", "cik", "period_end", "fiscal_quarter", "fiscal_year"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fin_df = pd.read_csv(INPUT_FINANCIAL)

    # --- Pivot to wide: one row per (ticker, period_end) with metric columns ---
    source_df = fin_df[fin_df["metric"].isin(SOURCE_METRICS)].copy()

    wide = source_df.pivot_table(
        index=ID_COLS,
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Ensure all source columns exist (some metrics may be absent for all firms)
    for col in SOURCE_METRICS:
        if col not in wide.columns:
            wide[col] = float("nan")

    print(f"Wide panel shape          : {wide.shape}  ({wide['ticker'].nunique()} firms)")
    print()

    # --- Compute ratios ---
    # Revenue = 0 treated as missing (avoids inf, preserves NaN semantics)
    rev = wide["revenue"].where(wide["revenue"] != 0)

    wide["gross_margin"]  = (wide["revenue"] - wide["cogs"]) / rev
    wide["rd_intensity"]  = wide["rd"]  / rev
    wide["sga_intensity"] = wide["sga"] / rev
    wide["sbc_intensity"] = wide["sbc"] / rev
    # opex_ratio: NaN propagates naturally if any component is NaN
    wide["opex_ratio"]    = (wide["cogs"] + wide["rd"] + wide["sga"]) / rev

    # --- Melt back to long ---
    long_df = wide[ID_COLS + RATIO_METRICS].melt(
        id_vars=ID_COLS,
        value_vars=RATIO_METRICS,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.sort_values(["ticker", "period_end", "metric"]).reset_index(drop=True)

    # --- Winsorize per metric (P1/P99, NaN-safe) ---
    thresholds: dict[str, dict] = {}
    long_df["value_winsorized"] = float("nan")

    for metric in RATIO_METRICS:
        mask = long_df["metric"] == metric
        vals = long_df.loc[mask, "value"]
        p1  = float(vals.quantile(0.01))
        p99 = float(vals.quantile(0.99))
        thresholds[metric] = {"p1": round(p1, 6), "p99": round(p99, 6)}
        long_df.loc[mask, "value_winsorized"] = vals.clip(lower=p1, upper=p99)

    # --- QA log: per-firm counts and warnings ---
    cik_map = (
        wide[["ticker", "cik"]]
        .drop_duplicates("ticker")
        .set_index("ticker")["cik"]
        .to_dict()
    )

    qa_records: list[dict] = []

    for ticker, grp in wide.groupby("ticker"):
        cik_raw = cik_map.get(ticker)
        cik = int(cik_raw) if pd.notna(cik_raw) else None

        n_obs = {f"n_obs_{m}": int(grp[m].notna().sum()) for m in RATIO_METRICS}

        warnings: list[str] = []

        gm_bad = grp.loc[grp["gross_margin"].notna() & (grp["gross_margin"] > 1), "period_end"].tolist()
        if gm_bad:
            warnings.append(f"gross_margin > 1 in {len(gm_bad)} quarter(s): {gm_bad}")

        rd_bad = grp.loc[grp["rd_intensity"].notna() & (grp["rd_intensity"] > 1), "period_end"].tolist()
        if rd_bad:
            warnings.append(f"rd_intensity > 1 in {len(rd_bad)} quarter(s): {rd_bad}")

        qa_records.append({
            "ticker":   ticker,
            "cik":      cik,
            **n_obs,
            "warnings": warnings,
        })

    # --- Write outputs ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUTPUT_PATH, index=False)

    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa_records, indent=2), encoding="utf-8")

    # --- Validation report ---
    print("=" * 60)
    print(f"Total rows in output      : {len(long_df):,}")
    print(f"Expected (~41,700)        : {321 * 26 * 5:,}  (321 firms × 26 qtrs × 5 metrics)")
    print()

    print("Non-NaN coverage by metric:")
    for metric in RATIO_METRICS:
        n = int(long_df.loc[long_df["metric"] == metric, "value"].notna().sum())
        print(f"  {metric:<20}: {n:>5}")
    print()

    print("Winsorization thresholds (P1 / P99):")
    for metric, t in thresholds.items():
        print(f"  {metric:<20}: P1={t['p1']:+.4f}  P99={t['p99']:+.4f}")
    print()

    # MSFT spot check
    msft = wide[wide["ticker"] == "MSFT"].sort_values("period_end")
    if not msft.empty:
        sample = msft[msft["period_end"] >= "2019-01-01"].tail(12)
        print("MSFT gross_margin and rd_intensity (last 12 quarters shown):")
        print(sample[
            ["period_end", "fiscal_quarter", "fiscal_year", "gross_margin", "rd_intensity"]
        ].to_string(index=False))
        print()
    else:
        print("MSFT: not found in panel")
        print()

    # Flag extreme gross_margin (outside [-1, 1])
    extreme = long_df[
        (long_df["metric"] == "gross_margin") &
        long_df["value"].notna() &
        ((long_df["value"] > 1) | (long_df["value"] < -1))
    ]
    if not extreme.empty:
        flagged = sorted(extreme["ticker"].unique().tolist())
        print(f"Gross margin outside [-1, 1]: {len(extreme)} obs across {len(flagged)} firms:")
        print(f"  {flagged}")
    else:
        print("Gross margin range check: all values within [-1, 1]")
    print()

    # QA summary
    warned = [q for q in qa_records if q["warnings"]]
    print(f"Firms with QA warnings    : {len(warned)}")
    if warned:
        print(f"  (e.g. {warned[0]['ticker']}: {warned[0]['warnings'][0][:80]}...)")
    print()

    print(f"Output written to         : {OUTPUT_PATH}")
    print(f"QA log written to         : {QA_PATH}")


if __name__ == "__main__":
    main()
