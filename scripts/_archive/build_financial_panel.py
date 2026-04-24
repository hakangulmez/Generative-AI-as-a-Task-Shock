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

# Additional metrics — each maps col_name → [tag, ...] tried in priority order
# For sga_expense: many SaaS firms report S&M + G&A separately instead of combined
# For gross_profit: fallback is Revenue - COGS (handled in main loop)
METRIC_TAGS = {
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcess",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
    ],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
        # fallback: sum of components (handled separately in extract_metric)
    ],
}

# COGS tags for gross_profit fallback: Revenue - COGS
COGS_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
]

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


def extract_annual(us_gaap: dict, tag: str) -> dict[int, dict]:
    """
    Extract annual (FY, ~365-day) entries for a single XBRL tag.

    Returns {fiscal_year_end_year: {"end": ..., "val": ..., "filed": ...}}
    Keyed by the calendar year of the period end date.
    """
    if tag not in us_gaap:
        return {}

    entries = us_gaap[tag].get("units", {}).get("USD", [])
    result: dict[int, dict] = {}

    for e in entries:
        # Only accept entries explicitly marked as full-year (FY)
        if e.get("fp", "") != "FY":
            continue
        end = e.get("end", "")
        start = e.get("start", "")
        if not end or not start:
            continue
        if not (PANEL_START <= end <= PANEL_END):
            continue

        try:
            days = (datetime.strptime(end, "%Y-%m-%d") -
                    datetime.strptime(start, "%Y-%m-%d")).days
        except ValueError:
            continue

        # Annual entries: ~340-400 days
        if days < 340 or days > 400:
            continue

        dt_end = datetime.strptime(end, "%Y-%m-%d")
        filed = e.get("filed", "")
        val = e.get("val")

        # Key by (year, month) of period end to handle non-calendar FY
        # Prefer larger value over later-filed: amended 10-K/A filings can
        # reduce a segment or strip divestitures, making Q4 = Annual - Q1-Q2-Q3
        # go negative. The original full-year figure is the correct denominator.
        key = (dt_end.year, dt_end.month)
        existing_val = result[key]["val"] if key in result else None
        if key not in result or (val is not None and existing_val is not None and val > existing_val):
            result[key] = {
                "period_end": end,
                "val": val,
                "filed": filed,
            }

    return result


def compute_q4(quarterly: dict[str, dict], annual: dict,
               all_quarterly_labels: set[str] = None) -> dict[str, dict]:
    """
    Compute Q4 = Annual FY - (Q1 + Q2 + Q3) for each fiscal year.

    Groups quarterly entries by the fiscal year they belong to (based on
    which annual period they fall within), then computes the residual.

    Returns {quarter_label: {"period_end": ..., "val": ..., ...}} for
    Q4 entries only.
    """
    if not annual:
        return {}

    q4_entries = {}

    for (fy_year, fy_month), ann in annual.items():
        ann_end = datetime.strptime(ann["period_end"], "%Y-%m-%d")
        ann_val = ann["val"]
        if ann_val is None:
            continue

        # Find the 3 quarterly entries that belong to this fiscal year.
        # A quarter belongs to a FY if its period_end is within 365 days
        # before (and including) the annual period_end.
        from datetime import timedelta
        fy_start_approx = ann_end - timedelta(days=365)
        q_sum = 0
        q_count = 0
        for qlabel, qdata in quarterly.items():
            q_end = datetime.strptime(qdata["period_end"], "%Y-%m-%d")
            if fy_start_approx < q_end <= ann_end:
                if qdata["val"] is not None:
                    q_sum += qdata["val"]
                    q_count += 1

        if q_count != 3:
            continue  # Need exactly 3 quarters to compute Q4

        q4_val = ann_val - q_sum
        if q4_val <= 0:
            continue  # Implausible — annual < sum of 3 quarters (tag mismatch / restatement)

        # Q4 period_end = annual period end date
        q4_end = ann["period_end"]
        parsed = quarter_from_date(q4_end)
        if not parsed:
            continue
        year, q, label = parsed

        # Skip if we already have this quarter from direct quarterly data
        if label in quarterly:
            continue
        if all_quarterly_labels and label in all_quarterly_labels:
            continue

        q4_entries[label] = {
            "period_end": q4_end,
            "fiscal_year": year,
            "fiscal_quarter": q,
            "val": q4_val,
            "filed": ann["filed"],
        }

    return q4_entries


def extract_metric(us_gaap: dict, col_name: str,
                   rev_labels: set) -> dict[str, dict]:
    """Extract a metric using fallback tags, with Q4 computation.

    For sga_expense: tries combined tag first, then sums
    SellingAndMarketingExpense + GeneralAndAdministrativeExpense per quarter.
    """
    tags = METRIC_TAGS[col_name]

    # Try each tag in priority order
    for tag in tags:
        q_data = extract_quarterly(us_gaap, tag)
        ann_data = extract_annual(us_gaap, tag)
        if ann_data:
            q4s = compute_q4(q_data, ann_data)
            q_data.update(q4s)
        if q_data:
            return q_data

    # sga_expense special case: sum S&M + G&A components
    if col_name == "sga_expense":
        sm_q = extract_quarterly(us_gaap, "SellingAndMarketingExpense")
        ga_q = extract_quarterly(us_gaap, "GeneralAndAdministrativeExpense")
        sm_ann = extract_annual(us_gaap, "SellingAndMarketingExpense")
        ga_ann = extract_annual(us_gaap, "GeneralAndAdministrativeExpense")
        if sm_ann:
            q4s = compute_q4(sm_q, sm_ann)
            sm_q.update(q4s)
        if ga_ann:
            q4s = compute_q4(ga_q, ga_ann)
            ga_q.update(q4s)
        if sm_q or ga_q:
            combined: dict[str, dict] = {}
            all_labels = set(sm_q.keys()) | set(ga_q.keys())
            for label in all_labels:
                sm_val = sm_q[label]["val"] if label in sm_q else None
                ga_val = ga_q[label]["val"] if label in ga_q else None
                if sm_val is None and ga_val is None:
                    continue
                total = (sm_val or 0) + (ga_val or 0)
                base = sm_q[label] if label in sm_q else ga_q[label]
                combined[label] = {**base, "val": total}
            return combined

    return {}


def extract_cogs(us_gaap: dict, rev_labels: set) -> dict[str, dict]:
    """Try COGS tags in priority order, with Q4 computation.

    Returns quarterly COGS data for use in gross_profit = revenue - cogs fallback.
    """
    for tag in COGS_TAGS:
        q_data = extract_quarterly(us_gaap, tag)
        ann_data = extract_annual(us_gaap, tag)
        if ann_data:
            q4s = compute_q4(q_data, ann_data)
            q_data.update(q4s)
        if q_data:
            return q_data
    return {}


def extract_revenue(us_gaap: dict) -> tuple[dict[str, dict], str]:
    """Try revenue tags in priority order, including Q4 from annual.

    Returns (quarters_dict, tag_used) where quarters_dict includes
    computed Q4 entries.
    """
    for tag in REVENUE_TAGS:
        result = extract_quarterly(us_gaap, tag)
        if result:
            # Compute Q4 from annual data
            ann = extract_annual(us_gaap, tag)
            if ann:
                q4s = compute_q4(result, ann)
                result.update(q4s)
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

    # Firms excluded due to material revenue restatements post-privatization
    EXCLUDED_TICKERS = {"IIIV"}

    df = pd.read_csv(INPUT_CSV)
    df = df[df["meets_filters"] == True].reset_index(drop=True)
    df = df[~df["ticker"].isin(EXCLUDED_TICKERS)].reset_index(drop=True)

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

        # Additional metrics (with Q4 from annual, with fallback tags)
        rev_labels = set(rev_quarters.keys())
        metrics = {}
        for col_name in METRIC_TAGS:
            metrics[col_name] = extract_metric(us_gaap, col_name, rev_labels)

        # gross_profit fallback: Revenue - COGS for quarters still missing
        gp_quarters = metrics["gross_profit"]
        missing_gp = rev_labels - set(gp_quarters.keys())
        if missing_gp:
            cogs_quarters = extract_cogs(us_gaap, rev_labels)
            for label in missing_gp:
                if label in cogs_quarters and label in rev_quarters:
                    rev_val = rev_quarters[label]["val"]
                    cogs_val = cogs_quarters[label]["val"]
                    if rev_val is not None and cogs_val is not None:
                        gp_quarters[label] = {
                            **rev_quarters[label],
                            "val": rev_val - cogs_val,
                        }

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
