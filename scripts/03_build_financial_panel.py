"""
Phase 3 — build quarterly financial panel in long format.

Reads companyfacts from the EDGAR cache (data/raw/edgar_cache/companyfacts/).
Does NOT fetch from EDGAR; run build_firm_universe.py / collect_10k_text.py first
so that all companyfacts JSON files are already cached.

Input:   data/raw/firm_universe.csv
         data/raw/edgar_cache/companyfacts/{padded_cik}.json  (must exist)
Output:  data/processed/financial_panel.csv

Columns: ticker, cik, period_end, fiscal_quarter, fiscal_year, metric, value

Metrics (10):
  revenue, cogs, rd, sga, capex, sbc, op_income, net_income,
  deferred_rev_current, deferred_rev_noncurrent

SGA fallback: if SellingGeneralAndAdministrativeExpense is absent, sums
  SellingAndMarketingExpense + GeneralAndAdministrativeExpense.  Logged to
  data/processed/financial_panel_qa.json.

Usage:
    python3 scripts/03_build_financial_panel.py                  # all firms
    python3 scripts/03_build_financial_panel.py --tickers MSFT CRM
    python3 scripts/03_build_financial_panel.py --output data/processed/panel_test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import edgar, xbrl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = Path("data/raw/firm_universe.csv")
OUTPUT_PATH = Path("data/processed/financial_panel.csv")
QA_PATH     = Path("data/processed/financial_panel_qa.json")

# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------
# Each entry: tags (priority-ordered), is_instant, optional fallback_sum
# "revenue" is handled separately via extract_quarterly_revenue.
METRICS: dict[str, dict] = {
    "cogs": {
        "tags": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
        "is_instant": False,
    },
    "rd": {
        "tags": [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
        "is_instant": False,
    },
    "sga": {
        "tags": ["SellingGeneralAndAdministrativeExpense"],
        "fallback_sum": ["SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"],
        "is_instant": False,
    },
    "capex": {
        "tags": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "is_instant": False,
    },
    "sbc": {
        "tags": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
        "is_instant": False,
    },
    "op_income": {
        "tags": ["OperatingIncomeLoss"],
        "is_instant": False,
    },
    "net_income": {
        "tags": ["NetIncomeLoss", "ProfitLoss"],
        "is_instant": False,
    },
    "deferred_rev_current": {
        "tags": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
        "is_instant": True,
    },
    "deferred_rev_noncurrent": {
        "tags": ["ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent"],
        "is_instant": True,
    },
}


# ---------------------------------------------------------------------------
# Per-firm extraction
# ---------------------------------------------------------------------------

def _process_firm(
    ticker: str,
    cik: int,
) -> tuple[list[dict], dict]:
    """
    Load companyfacts from cache and extract all 10 metrics in long format.

    Returns (rows, qa_record).
    rows      : list of {ticker, cik, period_end, fiscal_quarter, fiscal_year, metric, value}
    qa_record : per-firm summary for QA log
    """
    qa: dict = {
        "ticker":        ticker,
        "cik":           cik,
        "status":        None,
        "n_rows":        0,
        "metrics_found": [],
        "sga_fallback":  False,
        "warnings":      [],
    }

    # Load companyfacts (cache must exist — no HTTP)
    padded     = edgar.pad_cik(cik)
    cache_path = Path("data/raw/edgar_cache/companyfacts") / f"{padded}.json"
    if not cache_path.exists():
        qa["status"] = "no_cache"
        qa["warnings"].append(f"companyfacts cache missing: {cache_path}")
        return [], qa

    try:
        cf = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        qa["status"] = "cache_read_error"
        qa["warnings"].append(str(exc))
        return [], qa

    rows: list[dict] = []

    # --- Revenue (dedicated extractor) ---
    rev_series = xbrl.extract_quarterly_revenue(cf)
    if rev_series:
        qa["metrics_found"].append("revenue")
        for r in rev_series:
            rows.append({
                "ticker":         ticker,
                "cik":            cik,
                "period_end":     r["period_end"],
                "fiscal_quarter": r["fiscal_quarter"],
                "fiscal_year":    r["fiscal_year"],
                "metric":         "revenue",
                "value":          r["revenue"],
            })
    else:
        qa["warnings"].append("no revenue data")

    # --- All other metrics ---
    for metric_name, conf in METRICS.items():
        data, tag_used = xbrl.extract_quarterly_metric(
            cf,
            conf["tags"],
            is_instant=conf.get("is_instant", False),
            fallback_sum_tags=conf.get("fallback_sum"),
        )

        if tag_used is None:
            qa["warnings"].append(f"{metric_name}: no data")
            continue

        qa["metrics_found"].append(metric_name)

        if tag_used == "fallback_sum":
            qa["sga_fallback"] = True
            qa["warnings"].append(f"{metric_name}: used fallback component sum")

        for qdata in data.values():
            rows.append({
                "ticker":         ticker,
                "cik":            cik,
                "period_end":     qdata["period_end"],
                "fiscal_quarter": qdata["fiscal_quarter"],
                "fiscal_year":    qdata["fiscal_year"],
                "metric":         metric_name,
                "value":          qdata["val"],
            })

    qa["n_rows"]  = len(rows)
    qa["status"]  = "ok" if rev_series else "no_revenue"
    return rows, qa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: build quarterly financial panel (long format)"
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
    df_universe = pd.read_csv(INPUT_PATH)

    rows_input = list(df_universe.itertuples(index=False))

    if args.tickers:
        requested = {t.upper() for t in args.tickers}
        rows_input = [r for r in rows_input if r.ticker in requested]
        missing = requested - {r.ticker for r in rows_input}
        if missing:
            print(f"WARNING: not in firm_universe.csv: {', '.join(sorted(missing))}")

    print(f"Firms to process : {len(rows_input)}")
    print(f"Output           : {out_path}")
    print()

    all_rows: list[dict] = []
    qa_records: list[dict] = []
    sga_fallback_firms: list[str] = []

    t0 = time.time()
    for i, row in enumerate(rows_input, start=1):
        ticker = row.ticker
        cik    = int(row.cik)

        firm_rows, qa = _process_firm(ticker, cik)
        all_rows.extend(firm_rows)
        qa_records.append(qa)

        if qa["sga_fallback"]:
            sga_fallback_firms.append(ticker)

        status_tag = qa["status"]
        n_metrics  = len(qa["metrics_found"])
        print(f"  {ticker:<8}  {status_tag:<16}  "
              f"metrics={n_metrics}/10  rows={qa['n_rows']:>5}")

        if i % 50 == 0 or i == len(rows_input):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(rows_input)}]  elapsed={elapsed:.0f}s")

    # --- Merge with existing output if --tickers ---
    if args.tickers and out_path.exists():
        existing = pd.read_csv(out_path, dtype={"cik": str})
        updated_tickers = {r.ticker for r in rows_input}
        existing = existing[~existing["ticker"].isin(updated_tickers)]
        new_df   = pd.DataFrame(all_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = pd.DataFrame(all_rows)

    # Sort: ticker → metric → period_end
    if not combined.empty:
        combined = combined.sort_values(
            ["ticker", "metric", "period_end"]
        ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    # --- QA log ---
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa_records, indent=2), encoding="utf-8")

    # --- Validation report ---
    elapsed = time.time() - t0
    n_ok      = sum(1 for q in qa_records if q["status"] == "ok")
    n_no_rev  = sum(1 for q in qa_records if q["status"] == "no_revenue")
    n_no_cache = sum(1 for q in qa_records if q["status"] == "no_cache")
    n_error   = sum(1 for q in qa_records if q["status"] not in ("ok", "no_revenue", "no_cache"))

    print()
    print("=" * 60)
    print(f"Firms processed      : {len(qa_records)}")
    print(f"  ok (has revenue)   : {n_ok}")
    print(f"  no_revenue         : {n_no_rev}")
    print(f"  no_cache           : {n_no_cache}")
    print(f"  other errors       : {n_error}")
    print()
    print(f"Total rows written   : {len(combined):,}")
    print(f"Distinct firms       : {combined['ticker'].nunique() if not combined.empty else 0}")
    print()

    # Per-metric coverage
    if not combined.empty:
        metric_counts = combined.groupby("metric")["ticker"].nunique().sort_values(ascending=False)
        print("Metric coverage (firms with ≥1 observation):")
        for metric, count in metric_counts.items():
            print(f"  {metric:<28} {count:>4} firms")

    print()
    if sga_fallback_firms:
        print(f"SGA fallback (component sum) used for {len(sga_fallback_firms)} firms:")
        print(f"  {', '.join(sorted(sga_fallback_firms))}")
    else:
        print("SGA fallback: not needed for any firm")

    print()
    print(f"Time elapsed         : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output written to    : {out_path}")
    print(f"QA log written to    : {QA_PATH}")


if __name__ == "__main__":
    main()
