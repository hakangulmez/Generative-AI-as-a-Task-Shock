"""
Build quarterly financial panel from SEC EDGAR companyfacts API.

For each firm in firm_universe.csv, fetches quarterly revenue and
cost metrics from XBRL data. Outputs a firm-quarter panel covering
2019Q1–2025Q4.

Usage:
  python3 scripts/build_financial_panel.py
  python3 scripts/build_financial_panel.py --tickers MSFT CRM
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_CSV = "data/raw/firm_universe.csv"
OUTPUT_CSV = "data/processed/financial_panel.csv"

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 thesis-research hakanzekigulmez@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
RATE_LIMIT = 0.15

PANEL_START = "2019-01-01"
PANEL_END = "2025-12-31"
PRE_SHOCK_END = "2022-09-30"
MIN_PRE_SHOCK_QUARTERS = 6

# Revenue tags — tried in priority order, first with data wins
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomer",
]

# Additional metrics — each maps tag_name → output column
METRIC_TAGS = {
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pad_cik(cik: int) -> str:
    return str(cik).zfill(10)


def quarter_from_date(date_str: str) -> Optional[tuple]:
    """'2021-06-30' → (2021, 2, '2021Q2') or None."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        return dt.year, q, f"{dt.year}Q{q}"
    except (ValueError, TypeError):
        return None


def extract_quarterly(
    us_gaap: dict, tag: str
) -> dict[str, dict]:
    """
    Extract quarterly (non-cumulative) entries for a single XBRL tag.

    Returns {quarter_label: {"end": ..., "val": ..., "filed": ...}}

    Filters out cumulative (YTD) entries by checking period length:
    - ~60-100 days = quarterly (keep)
    - >100 days = cumulative YTD (skip)
    Also uses 'frame' field as a signal: 'CY2022Q2' = quarterly.
    For duplicate quarters, keeps the entry with the latest filed date.
    """
    if tag not in us_gaap:
        return {}

    entries = us_gaap[tag].get("units", {}).get("USD", [])
    result: dict[str, dict] = {}

    for e in entries:
        if e.get("fp", "") == "FY":
            continue
        end = e.get("end", "")
        if not end or not (PANEL_START <= end <= PANEL_END):
            continue

        # Filter: only true quarterly periods (~90 days), not cumulative
        start = e.get("start", "")
        if start and end:
            try:
                days = (datetime.strptime(end, "%Y-%m-%d") -
                        datetime.strptime(start, "%Y-%m-%d")).days
                if days > 100:
                    continue  # cumulative (YTD), skip
            except ValueError:
                pass

        parsed = quarter_from_date(end)
        if not parsed:
            continue
        year, q, label = parsed
        filed = e.get("filed", "")
        val = e.get("val")

        if label not in result or filed > result[label]["filed"]:
            result[label] = {
                "period_end": end,
                "fiscal_year": year,
                "fiscal_quarter": q,
                "val": val,
                "filed": filed,
            }

    return result


def extract_revenue(us_gaap: dict) -> tuple[dict[str, dict], str]:
    """Try revenue tags in priority order. Returns (quarters_dict, tag_used)."""
    for tag in REVENUE_TAGS:
        result = extract_quarterly(us_gaap, tag)
        if result:
            return result, tag
    return {}, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build quarterly financial panel from EDGAR companyfacts"
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Process only these tickers",
    )
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    df = df[df["meets_filters"] == True].reset_index(drop=True)

    if args.tickers:
        requested = [t.upper() for t in args.tickers]
        df = df[df["ticker"].isin(requested)].reset_index(drop=True)
        missing = set(requested) - set(df["ticker"])
        if missing:
            logger.warning("Not in universe: %s", ", ".join(sorted(missing)))

    total = len(df)
    logger.info("Processing %d firms from %s", total, INPUT_CSV)

    session = requests.Session()
    rows = []
    firms_ok = 0
    firms_no_revenue = []
    low_coverage = []

    for idx, (_, firm) in enumerate(df.iterrows()):
        cik = int(firm["cik"])
        ticker = firm["ticker"]
        company = firm["company_name"]

        time.sleep(RATE_LIMIT)
        try:
            resp = session.get(
                COMPANYFACTS_URL.format(cik=pad_cik(cik)),
                headers=HEADERS, timeout=30,
            )
            if resp.status_code == 429:
                logger.warning("Rate-limited, sleeping 12s…")
                time.sleep(12)
                resp = session.get(
                    COMPANYFACTS_URL.format(cik=pad_cik(cik)),
                    headers=HEADERS, timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("  %s: fetch failed: %s", ticker, e)
            firms_no_revenue.append(ticker)
            continue

        us_gaap = data.get("facts", {}).get("us-gaap", {})

        # Revenue
        rev_quarters, rev_tag = extract_revenue(us_gaap)
        if not rev_quarters:
            firms_no_revenue.append(ticker)
            if (idx + 1) % 50 == 0:
                logger.info("  [%d/%d] %s — no revenue", idx + 1, total, ticker)
            continue

        # Additional metrics
        metrics = {}
        for xbrl_tag, col_name in METRIC_TAGS.items():
            metrics[col_name] = extract_quarterly(us_gaap, xbrl_tag)

        # Build rows for this firm
        firm_rows = 0
        for label, rev in sorted(rev_quarters.items()):
            row = {
                "cik": cik,
                "ticker": ticker,
                "company_name": company,
                "period_end": rev["period_end"],
                "fiscal_year": rev["fiscal_year"],
                "fiscal_quarter": rev["fiscal_quarter"],
                "revenue": rev["val"],
            }
            for col_name, m_quarters in metrics.items():
                if label in m_quarters:
                    row[col_name] = m_quarters[label]["val"]
                else:
                    row[col_name] = None
            rows.append(row)
            firm_rows += 1

        firms_ok += 1

        # Pre-shock coverage check
        pre_shock_q = sum(
            1 for q in rev_quarters
            if rev_quarters[q]["period_end"] <= PRE_SHOCK_END
        )
        if pre_shock_q < MIN_PRE_SHOCK_QUARTERS:
            low_coverage.append((ticker, pre_shock_q))

        if (idx + 1) % 50 == 0:
            logger.info(
                "  [%d/%d] %s — %d quarters (pre-shock: %d) tag=%s",
                idx + 1, total, ticker, firm_rows, pre_shock_q,
                rev_tag[:40],
            )

    # Build DataFrame and save
    panel = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    panel.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d rows → %s", len(panel), OUTPUT_CSV)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("PANEL BUILD SUMMARY")
    logger.info("=" * 50)
    logger.info("Firms with revenue data: %d / %d", firms_ok, total)
    logger.info("Total firm-quarter observations: %d", len(panel))

    if firms_no_revenue:
        logger.info("Firms with NO revenue: %d — %s",
                     len(firms_no_revenue), ", ".join(firms_no_revenue[:20]))

    if low_coverage:
        logger.info("")
        logger.info("Firms with < %d pre-shock quarters:", MIN_PRE_SHOCK_QUARTERS)
        for t, n in sorted(low_coverage, key=lambda x: x[1]):
            logger.info("  %s: %d quarters", t, n)

    # Pre-shock coverage distribution
    if not panel.empty:
        pre = panel[panel["period_end"] <= PRE_SHOCK_END]
        pre_counts = pre.groupby("ticker").size()
        logger.info("")
        logger.info("Pre-shock quarter coverage (2019Q1-2022Q3):")
        logger.info("  Mean: %.1f quarters", pre_counts.mean())
        logger.info("  Median: %.0f", pre_counts.median())
        logger.info("  Min: %d (%s)", pre_counts.min(),
                     pre_counts.idxmin())
        logger.info("  Max: %d", pre_counts.max())
        bins = [0, 5, 7, 9, 11, 15, 99]
        labels = ["1-5", "6-7", "8-9", "10-11", "12-15", "15+"]
        dist = pd.cut(pre_counts, bins=bins, labels=labels).value_counts().sort_index()
        logger.info("  Distribution:")
        for bucket, count in dist.items():
            if count > 0:
                logger.info("    %s quarters: %d firms", bucket, count)


if __name__ == "__main__":
    main()
