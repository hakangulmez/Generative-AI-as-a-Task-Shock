"""
Phase 2.7 — build RPO (Remaining Performance Obligations) quarterly panel.

RPO is a balance-sheet stock (point-in-time snapshot at period end), not a
flow.  No Q4 formula is applied.

4-tier fallback hierarchy (first tier with data wins):
  Tier 1: RevenueRemainingPerformanceObligation           (ASC 606 gold standard)
  Tier 2: ContractWithCustomerLiability                   (combined contract liability)
  Tier 3: ContractWithCustomerLiabilityCurrent
           + ContractWithCustomerLiabilityNoncurrent      (component sum)
  Tier 4: DeferredRevenueCurrent
           + DeferredRevenueNoncurrent                    (legacy pre-ASC 606 sum)

Input:   data/raw/firm_universe.csv
         data/raw/edgar_cache/companyfacts/{padded_cik}.json  (must exist)
Output:  data/processed/rpo_quarterly.csv
         data/processed/rpo_quarterly_qa.json

Columns: ticker, cik, period_end, fiscal_quarter, fiscal_year, rpo_value, tag_used

Usage:
    python3 scripts/04_build_rpo_quarterly.py                  # all firms
    python3 scripts/04_build_rpo_quarterly.py --tickers ZS CRM
    python3 scripts/04_build_rpo_quarterly.py --output data/processed/rpo_test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import edgar, xbrl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = Path("data/raw/firm_universe.csv")
OUTPUT_PATH = Path("data/processed/rpo_quarterly.csv")
QA_PATH     = Path("data/processed/rpo_quarterly_qa.json")

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------
_TIER1 = "RevenueRemainingPerformanceObligation"
_TIER2 = "ContractWithCustomerLiability"
_TIER3 = ("ContractWithCustomerLiabilityCurrent",
           "ContractWithCustomerLiabilityNoncurrent")
_TIER4 = ("DeferredRevenueCurrent",
           "DeferredRevenueNoncurrent")


# ---------------------------------------------------------------------------
# Per-firm extraction
# ---------------------------------------------------------------------------

def _process_firm(ticker: str, cik: int) -> tuple[list[dict], dict]:
    """
    Try 4-tier RPO fallback for one firm.

    Returns (rows, qa_record).
    rows      : list of {ticker, cik, period_end, fiscal_quarter, fiscal_year,
                          rpo_value, tag_used}  — empty if no RPO found
    qa_record : per-firm QA summary
    """
    qa: dict = {
        "ticker":       ticker,
        "cik":          cik,
        "status":       None,
        "tier":         None,
        "n_quarters":   0,
        "period_range": None,
        "warnings":     [],
    }

    padded     = edgar.pad_cik(cik)
    cache_path = Path("data/raw/edgar_cache/companyfacts") / f"{padded}.json"
    if not cache_path.exists():
        qa["status"] = "no_cache"
        qa["tier"]   = "no_rpo"
        qa["warnings"].append(f"companyfacts cache missing: {cache_path}")
        return [], qa

    try:
        cf = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        qa["status"] = "cache_read_error"
        qa["tier"]   = "no_rpo"
        qa["warnings"].append(str(exc))
        return [], qa

    us_gaap = cf.get("facts", {}).get("us-gaap", {})

    # Try each tier in order; stop at first with data
    data:       dict = {}
    tier_label: str  = "no_rpo"

    # Tier 1 — single tag
    d = xbrl._extract_instant_raw(us_gaap, _TIER1)
    if d:
        data, tier_label = d, "tier_1_rpo"

    # Tier 2 — single tag
    if not data:
        d = xbrl._extract_instant_raw(us_gaap, _TIER2)
        if d:
            data, tier_label = d, "tier_2_cwcl_combined"

    # Tier 3 — component sum
    if not data:
        d = xbrl.extract_instant_sum(us_gaap, *_TIER3)
        if d:
            data, tier_label = d, "tier_3_cwcl_sum"

    # Tier 4 — legacy component sum
    if not data:
        d = xbrl.extract_instant_sum(us_gaap, *_TIER4)
        if d:
            data, tier_label = d, "tier_4_def_rev_sum"

    qa["tier"] = tier_label

    if not data:
        qa["status"] = "no_rpo"
        return [], qa

    rows = [
        {
            "ticker":         ticker,
            "cik":            cik,
            "period_end":     v["period_end"],
            "fiscal_quarter": v["fiscal_quarter"],
            "fiscal_year":    v["fiscal_year"],
            "rpo_value":      v["val"],
            "tag_used":       tier_label,
        }
        for v in sorted(data.values(), key=lambda x: x["period_end"])
    ]

    ends             = [r["period_end"] for r in rows]
    qa["status"]     = "ok"
    qa["n_quarters"] = len(rows)
    qa["period_range"] = f"{ends[0]} to {ends[-1]}"
    return rows, qa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.7: build RPO quarterly panel (long format)"
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

    out_path    = Path(args.output)
    df_universe = pd.read_csv(INPUT_PATH)
    rows_input  = list(df_universe.itertuples(index=False))

    if args.tickers:
        requested  = {t.upper() for t in args.tickers}
        rows_input = [r for r in rows_input if r.ticker in requested]
        missing    = requested - {r.ticker for r in rows_input}
        if missing:
            print(f"WARNING: not in firm_universe.csv: {', '.join(sorted(missing))}")

    print(f"Firms to process : {len(rows_input)}")
    print(f"Output           : {out_path}")
    print()

    all_rows:   list[dict] = []
    qa_records: list[dict] = []

    t0 = time.time()
    for i, row in enumerate(rows_input, start=1):
        ticker = row.ticker
        cik    = int(row.cik)

        # Placebo firms (pharma, energy, manufacturing, payments) don't have the
        # SaaS/subscription model where RPO is meaningful; skip to avoid false
        # positives contaminating H4 null-effect analysis.
        if row.tier == "placebo":
            qa_records.append({
                "ticker":       ticker,
                "cik":          cik,
                "status":       "placebo_skipped",
                "tier":         "n/a",
                "n_quarters":   0,
                "period_range": None,
                "warnings":     [],
            })
            continue

        firm_rows, qa = _process_firm(ticker, cik)
        all_rows.extend(firm_rows)
        qa_records.append(qa)

        tier = qa["tier"]
        nq   = qa["n_quarters"]
        if qa["status"] == "ok":
            print(f"  {ticker:<8}  {tier:<22}  quarters={nq:>3}  {qa['period_range']}")
        else:
            warn = qa["warnings"][0] if qa["warnings"] else ""
            print(f"  {ticker:<8}  {tier:<22}  {qa['status']}  {warn}")

        if i % 50 == 0 or i == len(rows_input):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(rows_input)}]  elapsed={elapsed:.0f}s")

    # --- Merge with existing output if --tickers ---
    if args.tickers and out_path.exists():
        existing        = pd.read_csv(out_path, dtype={"cik": str})
        updated_tickers = {r.ticker for r in rows_input}
        existing        = existing[~existing["ticker"].isin(updated_tickers)]
        combined        = pd.concat([existing, pd.DataFrame(all_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(all_rows)

    if not combined.empty:
        combined = combined.sort_values(
            ["ticker", "period_end"]
        ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_PATH.write_text(json.dumps(qa_records, indent=2), encoding="utf-8")

    # --- Validation report ---
    elapsed = time.time() - t0

    ok_records      = [q for q in qa_records if q["status"] == "ok"]
    no_rpo          = [q for q in qa_records if q["tier"] == "no_rpo"]
    placebo_skipped = [q for q in qa_records if q["status"] == "placebo_skipped"]
    attempted       = [q for q in qa_records if q["status"] != "placebo_skipped"]

    print()
    print("=" * 60)
    print(f"Firms in universe    : {len(qa_records)}")
    print(f"  placebo skipped    : {len(placebo_skipped)}  (not processed — no SaaS RPO concept)")
    print(f"  attempted          : {len(attempted)}")
    print(f"    with RPO data    : {len(ok_records)}")
    print(f"    no RPO found     : {len(no_rpo)}")
    print()

    # Tier breakdown (attempted firms only)
    from collections import Counter
    tier_counts = Counter(q["tier"] for q in attempted)
    print("Tier usage (non-placebo firms):")
    for tier in ("tier_1_rpo", "tier_2_cwcl_combined", "tier_3_cwcl_sum",
                 "tier_4_def_rev_sum", "no_rpo"):
        print(f"  {tier:<26} {tier_counts.get(tier, 0):>4} firms")
    print()

    # Coverage by universe tier
    df_universe_full = pd.read_csv(INPUT_PATH)
    tier_map = dict(zip(df_universe_full["ticker"], df_universe_full["tier"]))
    for utier in ("primary_software", "primary_knowledge", "placebo"):
        firms_in_tier = [q for q in qa_records if tier_map.get(q["ticker"]) == utier]
        covered       = [q for q in firms_in_tier if q["status"] == "ok"]
        skipped       = [q for q in firms_in_tier if q["status"] == "placebo_skipped"]
        if skipped:
            print(f"  {utier:<22} {len(covered):>3}/{len(firms_in_tier):<3}  ({len(skipped)} skipped)")
        else:
            print(f"  {utier:<22} {len(covered):>3}/{len(firms_in_tier):<3} firms with RPO data")
    print()

    # Tier-1 integrity check — all tier_1 firms should have _TIER1 in us_gaap
    tier1_firms = [q["ticker"] for q in qa_records if q["tier"] == "tier_1_rpo"]
    print(f"Tier-1 integrity: {len(tier1_firms)} firms assigned tier_1_rpo")
    print("  (structural guarantee: tier_1 wins when _extract_instant_raw returns data)")
    print()

    # ZS spot check
    if not combined.empty and "ZS" in combined["ticker"].values:
        zs = combined[combined["ticker"] == "ZS"].sort_values("period_end")
        print(f"ZS RPO spot check: {len(zs)} quarters  "
              f"[{zs['period_end'].iloc[0]} → {zs['period_end'].iloc[-1]}]")
        print(f"  tag_used: {zs['tag_used'].iloc[0]}")
        print(zs[["period_end", "fiscal_quarter", "fiscal_year", "rpo_value"]].to_string(index=False))
    print()

    print(f"Total rows written   : {len(combined):,}")
    print(f"Time elapsed         : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output written to    : {out_path}")
    print(f"QA log written to    : {QA_PATH}")


if __name__ == "__main__":
    main()
