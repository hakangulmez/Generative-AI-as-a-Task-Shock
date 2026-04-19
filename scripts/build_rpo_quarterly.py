"""
Build quarterly RPO panel from SEC EDGAR companyfacts API.

RPO is a balance-sheet stock (point-in-time snapshot), not a flow metric.
Extraction: take the value at each quarter-end date from 10-Q/10-K filings.

Tag hierarchy (best available per firm):
  1. RevenueRemainingPerformanceObligation
  2. ContractWithCustomerLiability
  3. ContractWithCustomerLiabilityCurrent + ContractWithCustomerLiabilityNoncurrent
  4. DeferredRevenue + DeferredRevenueNoncurrent

Output: data/processed/rpo_quarterly.csv
  ticker, period_end, fiscal_year, fiscal_quarter, rpo, rpo_tag
"""

import time, requests, logging, sys
import pandas as pd
from datetime import datetime

OUTPUT   = "data/processed/rpo_quarterly.csv"
HEADERS  = {"User-Agent": "thesis-research hakanzekigulmez@gmail.com"}
RATE     = 0.15
PANEL_START = "2019-01-01"
PANEL_END   = "2025-12-31"

RPO_TAGS = [
    ("rpo",      ["RevenueRemainingPerformanceObligation"]),
    ("cwcl",     ["ContractWithCustomerLiability"]),
    ("cwcl_sum", ["ContractWithCustomerLiabilityCurrent",
                  "ContractWithCustomerLiabilityNoncurrent"]),
    ("deferred", ["DeferredRevenue", "DeferredRevenueNoncurrent"]),
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def quarter_from_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        return dt.year, q
    except:
        return None, None


def extract_rpo_series(us_gaap: dict) -> tuple:
    """
    Returns (tag_name, {quarter_label: (period_end, val)}) for best tag.

    Balance sheet stock: take the value at each 10-Q/10-K end date.
    No cumulative filtering needed — one snapshot per period end.
    For duplicate period_end dates, keep the latest filed.
    """
    for tag_name, tag_keys in RPO_TAGS:
        by_end: dict[str, dict] = {}

        for tk in tag_keys:
            entries = us_gaap.get(tk, {}).get("units", {}).get("USD", [])
            for e in entries:
                end = e.get("end", "")
                if not end or not (PANEL_START <= end <= PANEL_END):
                    continue
                if e.get("form", "") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                filed = e.get("filed", "")
                val = e.get("val")
                if val is None:
                    continue
                # For sum tags: accumulate across keys at same end date
                if end not in by_end:
                    by_end[end] = {"val": 0, "filed": filed, "count": 0}
                if filed >= by_end[end]["filed"]:
                    by_end[end]["val"] += val
                    by_end[end]["filed"] = filed
                    by_end[end]["count"] += 1

        if not by_end:
            continue

        # For sum tags: only keep dates where both components were found
        if len(tag_keys) > 1:
            by_end = {d: v for d, v in by_end.items() if v["count"] == len(tag_keys)}

        if not by_end:
            continue

        # Build quarter-keyed dict
        result = {}
        for end_date, info in sorted(by_end.items()):
            yr, q = quarter_from_date(end_date)
            if yr is None:
                continue
            label = f"{yr}Q{q}"
            # For duplicate quarters keep latest end_date (most recent in quarter)
            if label not in result or end_date > result[label]["period_end"]:
                result[label] = {"period_end": end_date, "fiscal_year": yr,
                                  "fiscal_quarter": q, "val": info["val"]}

        if result:
            return tag_name, result

    return None, {}


def main():
    panel = pd.read_csv("data/processed/financial_panel.csv")[["ticker","period_end","fiscal_year","fiscal_quarter"]]
    fu    = pd.read_csv("data/raw/firm_universe.csv")[["ticker","cik"]]
    tickers = panel["ticker"].unique()

    log.info("Fetching RPO for %d firms", len(tickers))

    rows = []
    no_data = []

    for i, ticker in enumerate(sorted(tickers)):
        cik_row = fu[fu["ticker"] == ticker]
        if len(cik_row) == 0:
            continue
        cik = str(cik_row["cik"].values[0]).zfill(10)

        try:
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                             headers=HEADERS, timeout=15)
            us_gaap = r.json().get("facts", {}).get("us-gaap", {})
        except Exception as e:
            log.warning("  %s: fetch error %s", ticker, e)
            no_data.append(ticker)
            time.sleep(RATE)
            continue

        tag_name, series = extract_rpo_series(us_gaap)

        if not series:
            no_data.append(ticker)
        else:
            for label, d in series.items():
                rows.append({
                    "ticker":         ticker,
                    "period_end":     d["period_end"],
                    "fiscal_year":    d["fiscal_year"],
                    "fiscal_quarter": d["fiscal_quarter"],
                    "rpo":            d["val"],
                    "rpo_tag":        tag_name,
                })

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] %s tag=%s obs=%d",
                     i+1, len(tickers), ticker, tag_name or "none", len(series))

        time.sleep(RATE)

    df = pd.DataFrame(rows)

    # Merge with financial panel quarters to align
    panel["period_end"] = pd.to_datetime(panel["period_end"]).dt.strftime("%Y-%m-%d")
    merged = panel.merge(df[["ticker","period_end","rpo","rpo_tag"]],
                         on=["ticker","period_end"], how="left")

    merged.to_csv(OUTPUT, index=False)

    n_firms_with = merged[merged["rpo"].notna()]["ticker"].nunique()
    n_obs = merged["rpo"].notna().sum()
    log.info("Saved: %s", OUTPUT)
    log.info("Firms with RPO data: %d / %d", n_firms_with, len(tickers))
    log.info("Obs with RPO: %d / %d", n_obs, len(merged))
    log.info("No data: %s", no_data)


if __name__ == "__main__":
    main()
