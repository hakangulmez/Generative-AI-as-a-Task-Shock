"""
Build the firm universe from SEC EDGAR for the thesis:
"Generative AI as a Task Shock: Product-Level LLM Substitution
 in B2B Software Markets"

Filters (applied in order):
  1. SIC 7370–7379 (Computer Programming, Data Processing, etc.)
  2. Listed on NYSE or Nasdaq as of 2022Q4
  3. ≥6 quarterly revenue data points in companyfacts XBRL API
     during 2020Q1–2022Q3. Excludes 40-F filers without quarterly
     XBRL (e.g. DSGX) and firms with data gaps (e.g. SLP).
  4. Consumer-facing flag (>50% consumer revenue) — flagged, NOT excluded

Pipeline:
  Phase A — collect CIKs for SIC 7370-7379 via submissions endpoint.
             For each SIC match, fetch companyfacts to check revenue existence.
  Phase B — apply filters 2-4, write outputs.

Output:
  data/raw/firm_universe.csv
  data/raw/universe_build_summary.txt
  logs/universe_build_log.json

Usage:
  python3 scripts/build_firm_universe.py              # full run
  python3 scripts/build_firm_universe.py --dry-run    # 7 test firms only
"""

import os
import sys
import json
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
SIC_RANGE = range(7370, 7380)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
COMPANY_TICKERS_EXCHANGE_URL = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 thesis-research hakanzekigulmez@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT = 0.11          # seconds between EDGAR requests
PANEL_START = "2020-01-01"
PRE_SHOCK_END = "2022-09-30"  # end of 2022Q3
MIN_PRE_SHOCK_QUARTERS = 6   # ≥6 out of 11 possible (2020Q1-2022Q3)

VALID_EXCHANGES = {"NYSE", "NASDAQ"}

# Revenue XBRL tags — tried in priority order, first match wins
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomer",
]

# Consumer-facing heuristic: company-name substrings
CONSUMER_NAME_HINTS = [
    "match group", "bumble", "spotify", "dropbox", "grubhub",
    "shutterstock", "yelp", "zillow", "trivago", "cargurus",
    "angi homeservices", "angi inc", "iac/", "iac inc", "vimeo",
    "fiverr", "etsy", "wix.com", "godaddy",
]

OUTPUT_CSV = "data/raw/firm_universe.csv"
SUMMARY_TXT = "data/raw/universe_build_summary.txt"
BUILD_LOG = "logs/universe_build_log.json"
SIC_CHECKPOINT = "logs/sic_firms_checkpoint.json"

# Known SIC-737x firms for --dry-run (verified CIKs)
DRY_RUN_CIKS = [
    796343,   # ADBE — ~9 quarters → PASS
    1050140,  # DSGX — ~2 quarters (40-F filer) → FAIL filter 3
    1023459,  # SLP  — ~2 quarters (data gap) → FAIL filter 3
    1617553,  # ZIP  — ~8 quarters (retroactive) → PASS
    1577526,  # AI   — ~8 quarters (retroactive) → PASS
    1676238,  # BRZE — ~6 quarters (borderline) → PASS
    1477720,  # ASAN — ~8 quarters → PASS
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
def edgar_get(url: str, session: requests.Session) -> Optional[dict]:
    """GET JSON from EDGAR with rate limiting and retry on 429."""
    time.sleep(RATE_LIMIT)
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            logger.warning("Rate-limited by EDGAR, sleeping 12 s …")
            time.sleep(12)
            resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed %s: %s", url, e)
        return None


def pad_cik(cik: int) -> str:
    return str(cik).zfill(10)


# ---------------------------------------------------------------------------
# companyfacts revenue quarter counter
# ---------------------------------------------------------------------------
def count_pre_shock_quarters(facts_data: Optional[dict]) -> tuple[int, str]:
    """
    Count unique quarters with revenue data in companyfacts
    during 2020Q1-2022Q3. FY (full-year) entries excluded.

    Returns (count, tag_used).
    """
    if not facts_data:
        return 0, ""

    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

    for tag in REVENUE_TAGS:
        if tag not in us_gaap:
            continue
        entries = us_gaap[tag].get("units", {}).get("USD", [])
        quarters: set[str] = set()
        for entry in entries:
            if entry.get("fp", "") == "FY":
                continue
            end_date = entry.get("end", "")
            if end_date and PANEL_START <= end_date <= PRE_SHOCK_END:
                dt = datetime.strptime(end_date, "%Y-%m-%d")
                quarters.add(f"{dt.year}Q{(dt.month - 1) // 3 + 1}")
        if quarters:
            return len(quarters), tag

    return 0, ""


# ---------------------------------------------------------------------------
# Phase A — Identify SIC 7370-7379 firms + enrich
# ---------------------------------------------------------------------------
def phase_a_get_sic_firms(
    session: requests.Session, *, dry_run: bool = False
) -> list[dict]:
    """
    Return a list of dicts, one per firm with SIC 7370-7379.

    For each firm, fetches both submissions (metadata) and
    companyfacts (quarterly revenue coverage).

    In --dry-run mode, use 7 hard-coded CIKs.
    """
    logger.info("=" * 60)
    logger.info("PHASE A: Identify SIC 7370-7379 firms")
    logger.info("=" * 60)

    if dry_run:
        return _fetch_firms_by_cik(DRY_RUN_CIKS, session, label="dry-run")

    # Check checkpoint
    if os.path.exists(SIC_CHECKPOINT):
        logger.info("  Found SIC checkpoint at %s — loading …", SIC_CHECKPOINT)
        with open(SIC_CHECKPOINT) as fh:
            firms = json.load(fh)
        logger.info("  Loaded %d SIC 737x firms from checkpoint.", len(firms))
        return firms

    # 1. Bulk ticker list → unique CIKs
    logger.info("  Downloading company_tickers_exchange.json …")
    time.sleep(RATE_LIMIT)
    resp = session.get(COMPANY_TICKERS_EXCHANGE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    fields = raw["fields"]
    cik_idx = fields.index("cik")

    unique_ciks = sorted({row[cik_idx] for row in raw["data"]})
    logger.info("  Total unique CIKs to check: %d", len(unique_ciks))

    # 2. Check SIC for each CIK; enrich SIC matches
    sic_firms = []
    for i, cik in enumerate(unique_ciks, 1):
        if i % 500 == 0:
            logger.info(
                "  Checked %d / %d CIKs — %d SIC 737x so far",
                i, len(unique_ciks), len(sic_firms),
            )
        data = edgar_get(SUBMISSIONS_URL.format(cik=pad_cik(cik)), session)
        if data is None:
            continue
        sic = data.get("sic", "")
        if sic.isdigit() and int(sic) in SIC_RANGE:
            sic_firms.append(_extract_firm_info(cik, data, session))

    logger.info("  Done. Found %d firms with SIC 7370-7379.", len(sic_firms))

    # Save checkpoint
    os.makedirs(os.path.dirname(SIC_CHECKPOINT), exist_ok=True)
    with open(SIC_CHECKPOINT, "w") as fh:
        json.dump(sic_firms, fh, indent=2, default=str)
    logger.info("  Saved checkpoint → %s", SIC_CHECKPOINT)

    return sic_firms


def _fetch_firms_by_cik(
    ciks: list[int], session: requests.Session, *, label: str = ""
) -> list[dict]:
    """Fetch submissions + companyfacts for a known list of CIKs."""
    firms = []
    for cik in ciks:
        data = edgar_get(SUBMISSIONS_URL.format(cik=pad_cik(cik)), session)
        if data is None:
            continue
        firms.append(_extract_firm_info(cik, data, session))
    logger.info("  [%s] Fetched %d / %d firms.", label, len(firms), len(ciks))
    return firms


def _extract_firm_info(
    cik: int, data: dict, session: requests.Session
) -> dict:
    """
    Pull metadata from submissions JSON + check revenue existence
    via companyfacts API.
    """
    tickers = data.get("tickers", [])
    exchanges = data.get("exchanges", [])

    # companyfacts API — count pre-shock quarterly revenue data points
    facts = edgar_get(COMPANYFACTS_URL.format(cik=pad_cik(cik)), session)
    q_count, rev_tag = count_pre_shock_quarters(facts)

    return {
        "cik": cik,
        "ticker": tickers[0] if tickers else "",
        "company_name": data.get("name", ""),
        "sic": data.get("sic", ""),
        "exchange": exchanges[0] if exchanges else "",
        "pre_shock_quarters": q_count,
        "revenue_tag_used": rev_tag,
    }


# ---------------------------------------------------------------------------
# Phase B — Apply filters
# ---------------------------------------------------------------------------
def phase_b_apply_filters(firms: list[dict]) -> tuple[list[dict], dict]:
    """
    Apply filters 2-4. Returns (all_firms_annotated, build_log).
    Every firm gets meets_filters + filter_failed_at.
    """
    log: dict = {
        "timestamp": datetime.now().isoformat(),
        "steps": [],
    }

    # Step 1 (already done): SIC 7370-7379
    log["steps"].append({
        "step": 1,
        "description": "SIC 7370-7379 from SEC EDGAR",
        "firms_remaining": len(firms),
        "dropped": 0,
    })
    logger.info("After filter 1 (SIC 7370-7379): %d firms", len(firms))

    # Step 2: Exchange = NYSE or Nasdaq
    for f in firms:
        f["_pass_2"] = (f.get("exchange") or "").upper() in VALID_EXCHANGES

    n2 = sum(f["_pass_2"] for f in firms)
    d2 = len(firms) - n2
    log["steps"].append({
        "step": 2,
        "description": "Listed on NYSE or Nasdaq",
        "firms_remaining": n2,
        "dropped": d2,
        "sample_dropped": [
            f"{f['ticker']} (exchange={f.get('exchange','')})"
            for f in firms if not f["_pass_2"]
        ][:10],
    })
    logger.info("After filter 2 (NYSE/Nasdaq): %d firms remain (%d dropped)", n2, d2)

    # Step 3: ≥6 quarterly revenue data points in companyfacts (2020Q1-2022Q3)
    for f in firms:
        f["_pass_3"] = (
            f.get("pre_shock_quarters", 0) >= MIN_PRE_SHOCK_QUARTERS
            if f["_pass_2"] else False
        )

    n3 = sum(f["_pass_3"] for f in firms)
    d3 = n2 - n3
    log["steps"].append({
        "step": 3,
        "description": (
            f"≥{MIN_PRE_SHOCK_QUARTERS} quarterly revenue data points "
            f"in companyfacts (2020Q1-2022Q3)"
        ),
        "firms_remaining": n3,
        "dropped": d3,
        "sample_dropped": [
            f"{f['ticker']} (quarters={f.get('pre_shock_quarters', 0)})"
            for f in firms if f["_pass_2"] and not f["_pass_3"]
        ][:15],
    })
    logger.info(
        "After filter 3 (≥%d revenue quarters): %d firms (%d dropped)",
        MIN_PRE_SHOCK_QUARTERS, n3, d3,
    )

    # Step 4: Consumer-facing flag (flag only, no exclusion)
    for f in firms:
        name_lower = f.get("company_name", "").lower()
        f["consumer_flag"] = any(kw in name_lower for kw in CONSUMER_NAME_HINTS)

    consumer_count = sum(f["consumer_flag"] and f["_pass_3"] for f in firms)
    flagged_names = [
        f["ticker"] for f in firms if f["consumer_flag"] and f["_pass_3"]
    ]
    log["steps"].append({
        "step": 4,
        "description": "Consumer-facing flag (NOT excluded, flagged for review)",
        "firms_remaining": n3,
        "dropped": 0,
        "flagged_consumer": consumer_count,
        "flagged_firms": flagged_names,
    })
    logger.info(
        "After filter 4 (consumer flag): %d firms (%d flagged consumer-facing)",
        n3, consumer_count,
    )

    # Annotate every firm
    for f in firms:
        if f["_pass_3"]:
            f["meets_filters"] = True
            f["filter_failed_at"] = ""
        elif f["_pass_2"]:
            f["meets_filters"] = False
            f["filter_failed_at"] = "revenue_coverage"
        else:
            f["meets_filters"] = False
            f["filter_failed_at"] = "exchange"

    log["final_passing"] = n3
    log["total_sic"] = len(firms)

    return firms, log


# ---------------------------------------------------------------------------
# Phase C — Validate & write outputs
# ---------------------------------------------------------------------------
def validate(firms: list[dict], log: dict) -> str:
    n = log["final_passing"]
    lines = []
    if n < 120:
        lines.append(
            f"WARNING: Only {n} firms passed filters (expected 143-180). "
            "Check most restrictive filter."
        )
    elif n > 200:
        lines.append(
            f"WARNING: {n} firms passed filters (expected 143-180). "
            "Filters may be too loose."
        )
    else:
        lines.append(f"OK: {n} firms (expected range 143-180).")

    for step in log["steps"]:
        if step.get("dropped", 0) > 0:
            lines.append(
                f"  Step {step['step']} ({step['description']}): "
                f"dropped {step['dropped']}"
            )
    return "\n".join(lines)


def write_outputs(
    firms: list[dict], log: dict, validation: str, *, dry_run: bool
):
    passing = sorted(
        [f for f in firms if f["meets_filters"]],
        key=lambda f: f.get("ticker", ""),
    )

    if dry_run:
        logger.info("[DRY RUN] — not writing files.")
        logger.info("Validation:\n%s", validation)
        logger.info("Firms passing all filters:")
        for f in passing:
            logger.info(
                "  %-8s %-35s SIC=%s exch=%-6s q=%2d",
                f["ticker"], f["company_name"][:35], f["sic"],
                f["exchange"], f.get("pre_shock_quarters", 0),
            )
        failed = [f for f in firms if not f["meets_filters"]]
        if failed:
            logger.info("Firms FAILED:")
            for f in failed:
                logger.info(
                    "  %-8s %-35s failed_at=%-18s q=%d",
                    f["ticker"], f["company_name"][:35],
                    f.get("filter_failed_at", "?"),
                    f.get("pre_shock_quarters", 0),
                )
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(BUILD_LOG), exist_ok=True)

    # CSV — only passing firms
    cols = [
        "ticker", "cik", "company_name", "sic", "exchange",
        "meets_filters", "consumer_flag", "filter_failed_at",
        "pre_shock_quarters",
    ]
    df = pd.DataFrame(passing)
    df[[c for c in cols if c in df.columns]].to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d firms → %s", len(passing), OUTPUT_CSV)

    # Build log JSON (all firms, including dropped)
    with open(BUILD_LOG, "w") as fh:
        json.dump(log, fh, indent=2, default=str)
    logger.info("Wrote build log → %s", BUILD_LOG)

    # Summary text
    summary = _build_summary(passing, log, validation)
    with open(SUMMARY_TXT, "w") as fh:
        fh.write(summary)
    logger.info("Wrote summary → %s", SUMMARY_TXT)
    print("\n" + summary)


def _build_summary(passing: list[dict], log: dict, validation: str) -> str:
    lines = [
        "=" * 60,
        "FIRM UNIVERSE BUILD SUMMARY",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60, "",
    ]

    for step in log["steps"]:
        lines.append(f"Step {step['step']}: {step['description']}")
        lines.append(f"  Firms remaining: {step['firms_remaining']}")
        if step.get("dropped"):
            lines.append(f"  Dropped: {step['dropped']}")
        if step.get("flagged_consumer"):
            lines.append(f"  Flagged consumer-facing: {step['flagged_consumer']}")
        lines.append("")

    lines += ["-" * 60, f"FINAL COUNT: {len(passing)} firms", ""]

    # SIC breakdown
    lines.append("Breakdown by SIC code:")
    sic_counts: dict[str, int] = {}
    for f in passing:
        s = f.get("sic", "?")
        sic_counts[s] = sic_counts.get(s, 0) + 1
    for sic, count in sorted(sic_counts.items()):
        lines.append(f"  SIC {sic}: {count}")
    lines.append("")

    # Exchange breakdown
    lines.append("Breakdown by exchange:")
    ex_counts: dict[str, int] = {}
    for f in passing:
        ex = f.get("exchange", "?")
        ex_counts[ex] = ex_counts.get(ex, 0) + 1
    for ex, count in sorted(ex_counts.items()):
        lines.append(f"  {ex}: {count}")
    lines.append("")

    lines += ["-" * 60, "VALIDATION", validation, ""]

    # Consumer flags
    consumer = [f for f in passing if f.get("consumer_flag")]
    if consumer:
        lines.append("Consumer-flagged firms (review manually):")
        for f in consumer:
            lines.append(f"  {f['ticker']} — {f['company_name']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build firm universe from SEC EDGAR (SIC 7370-7379)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test with 7 known firms; do not write output files",
    )
    args = parser.parse_args()

    session = requests.Session()

    logger.info("=" * 60)
    logger.info("BUILDING FIRM UNIVERSE — SEC EDGAR SIC 7370-7379")
    logger.info("Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    firms = phase_a_get_sic_firms(session, dry_run=args.dry_run)
    firms, log = phase_b_apply_filters(firms)
    validation = validate(firms, log)
    logger.info("\n%s", validation)
    write_outputs(firms, log, validation, dry_run=args.dry_run)

    n = sum(1 for f in firms if f["meets_filters"])
    logger.info("Done. %d firms in final universe.", n)


if __name__ == "__main__":
    main()
