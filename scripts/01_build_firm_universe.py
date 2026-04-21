"""
01_build_firm_universe.py — Build three-tier firm universe from SEC EDGAR.

TIER 1 — primary_software : SIC 7370-7379, NYSE/NASDAQ/NYSE American, ≥6 pre-shock quarters.
TIER 2 — primary_knowledge: Manual ticker list (info services, publishing, education, consulting).
TIER 3 — placebo          : Manual ticker list (pharma, energy, manufacturing, payments).
                            Used as construct-validity / placebo test in H4.

Output column 'tier' ∈ {primary_software, primary_knowledge, placebo, placebo_both}.
  placebo_both: ticker is in placebo list AND has SIC 7370-7379 (unexpected — flagged).

NAICS NOTE: Auto SIC→NAICS crosswalk for universe membership is intentionally DISABLED.
  The crosswalk (SIC 7389 → NAICS 5191) pulls in 60+ payments/BPO/consumer firms that
  are not knowledge-product firms. Manual tier 2/3 lists are authoritative.
  The _SIC_TO_NAICS4 dict is retained only to populate the 'naics' output column.

Usage:
  python3 scripts/01_build_firm_universe.py --dry-run
  python3 scripts/01_build_firm_universe.py --tier primary_software
  python3 scripts/01_build_firm_universe.py --tier primary_knowledge
  python3 scripts/01_build_firm_universe.py --tier placebo
  python3 scripts/01_build_firm_universe.py

DO NOT RUN THE FULL BUILD without explicit authorization — it fetches submissions +
companyfacts from SEC EDGAR. With warm cache: <2 min. Cold: ~60 min.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from utils.edgar import get_company_tickers, get_submissions, get_companyfacts, pad_cik
from utils.xbrl import REVENUE_TAGS, _extract_quarterly_raw, _extract_annual_raw, _compute_q4
from utils import get_logger

logger = get_logger("universe", "01_build_firm_universe")


def _count_pre_shock_quarters(cf: dict, cutoff: str) -> int:
    """
    Count pre-shock quarterly revenue observations, taking the MAX across all
    REVENUE_TAGS.

    Standard extract_quarterly_revenue() stops at the first tag with ANY data.
    Some firms (e.g. EA, OSPN) adopted a new XBRL tag post-shock, so that tag
    exists with post-shock data only — the function never falls through to the
    older tag that has the full pre-shock history.

    For universe-building we only need the count (≥ MIN_PRE_SHOCK_QUARTERS),
    not the tag-consistent series, so using the best-across-tags count is correct.
    The financial panel builder uses strict tag priority (first-tag-wins) for
    reproducibility; this function is intentionally separate.
    """
    us_gaap = cf.get("facts", {}).get("us-gaap", {})
    best = 0
    for tag in REVENUE_TAGS:
        quarters = _extract_quarterly_raw(us_gaap, tag)
        annual   = _extract_annual_raw(us_gaap, tag)
        if annual:
            q4s = _compute_q4(quarters, annual)
            quarters.update(q4s)
        n = sum(1 for v in quarters.values() if v["period_end"] < cutoff)
        best = max(best, n)
    return best


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
OUTPUT_CSV          = Path("data/raw/firm_universe.csv")
BUILD_LOG           = Path("logs/universe_build.jsonl")
SUMMARY_TXT         = Path("data/raw/universe_build_summary.txt")
CONFIG_PATH         = Path("config/universe_filters.yaml")
UNIVERSE_TICKERS_PATH = Path("config/universe_tickers.yaml")

PRE_SHOCK_CUTOFF       = "2022-11-01"
MIN_PRE_SHOCK_QUARTERS = 6

VALID_EXCHANGES = frozenset({"NYSE", "NASDAQ", "NYSE AMERICAN", "NYSE MKT"})
ANNUAL_FORMS    = frozenset({"10-K", "10-K/A", "10-K405"})
EXCLUDED_TICKERS = frozenset({"IIIV"})

# Legacy consumer-name heuristic — lowercase substring match on company_name
CONSUMER_NAME_HINTS: list[str] = [
    "match group", "bumble", "spotify", "dropbox", "grubhub",
    "shutterstock", "yelp", "zillow", "trivago", "cargurus",
    "angi homeservices", "angi inc", "iac/", "iac inc", "vimeo",
    "fiverr", "etsy", "wix.com", "godaddy",
    # Consumer gaming / entertainment
    "electronic arts", "sports entertainment gaming", "motorsport games",
    "inspired entertainment",
]

DRY_RUN_TICKERS = ["ZS", "VEEV", "DDOG", "MCO", "PFE"]

# ---------------------------------------------------------------------------
# Census SIC-to-NAICS crosswalk — used ONLY to populate the 'naics' output
# column. NOT used to determine universe membership (auto SIC→NAICS discovery
# is intentionally disabled — see module docstring).
# ---------------------------------------------------------------------------
_SIC_TO_NAICS4: dict[int, str] = {
    # Primary software SIC range
    7370: "5415", 7371: "5415", 7372: "5112", 7373: "5415",
    7374: "5182", 7375: "5191", 7376: "5415", 7377: "5415",
    7378: "5415", 7379: "5191",
    # Publishing
    2711: "5111", 2712: "5111", 2721: "5111", 2731: "5111", 2741: "5111",
    # Education
    8200: "6114", 8220: "6113", 8221: "6113",
    8244: "6114", 8249: "6114", 8299: "6114",
    # Consulting
    8742: "5416", 8748: "5416",
    # Financial data / info services
    6199: "5239", 6282: "5239", 6411: "5239", 6159: "5239",
    # Other
    7389: "5191",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_universe_tickers() -> dict[str, dict]:
    """
    Load config/universe_tickers.yaml.

    Returns dict: ticker (uppercase) → {tier: str, sector: str,
                                         company: str, expected_rho: int|None}
    Tiers present: primary_knowledge, placebo.
    """
    with open(UNIVERSE_TICKERS_PATH) as f:
        raw = yaml.safe_load(f)

    result: dict[str, dict] = {}

    for entry in raw.get("primary_knowledge_tickers", []):
        ticker = str(entry["ticker"]).upper()
        result[ticker] = {
            "tier":         "primary_knowledge",
            "sector":       str(entry.get("sector", "")),
            "company":      entry.get("company", ""),
            "expected_rho": entry.get("expected_rho"),
        }

    for entry in raw.get("placebo_tickers", []):
        ticker = str(entry["ticker"]).upper()
        result[ticker] = {
            "tier":         "placebo",
            "sector":       str(entry.get("sector", "")),
            "company":      entry.get("company", ""),
            "expected_rho": entry.get("expected_rho"),
        }

    return result


# ---------------------------------------------------------------------------
# Phase A/B helpers
# ---------------------------------------------------------------------------

def get_all_listed_ciks() -> list[dict]:
    """Return all CIKs from SEC's company_tickers_exchange.json."""
    raw = get_company_tickers()
    fields    = raw["fields"]
    cik_idx   = fields.index("cik")
    name_idx  = fields.index("name")
    tick_idx  = fields.index("ticker")
    exch_idx  = fields.index("exchange")

    seen: set[int] = set()
    firms = []
    for row in raw["data"]:
        cik = int(row[cik_idx])
        if cik in seen:
            continue
        seen.add(cik)
        firms.append({
            "cik":      cik,
            "name":     row[name_idx],
            "ticker":   str(row[tick_idx]),
            "exchange": str(row[exch_idx]),
        })
    return firms


def enrich_from_submissions(cik: int) -> Optional[dict]:
    """
    Fetch submissions JSON; return metadata dict or None on failure.
    Fields: sic, company_name, tickers, exchanges, first_10k_date,
            last_10k_date, has_annual_10k.
    """
    try:
        data = get_submissions(cik)
    except RuntimeError:
        return None

    sic_raw      = data.get("sic", "")
    sic          = int(sic_raw) if str(sic_raw).isdigit() else None
    company_name = data.get("name", "")
    tickers      = data.get("tickers", [])
    exchanges    = data.get("exchanges", [])

    filings      = data.get("filings", {}).get("recent", {})
    forms        = filings.get("form", [])
    filing_dates = filings.get("filingDate", [])

    ten_k_dates = [d for f, d in zip(forms, filing_dates) if f in ANNUAL_FORMS]
    first_10k   = min(ten_k_dates) if ten_k_dates else None
    last_10k    = max(ten_k_dates) if ten_k_dates else None

    return {
        "sic":            sic,
        "company_name":   company_name,
        "tickers":        tickers,
        "exchanges":      exchanges,
        "first_10k_date": first_10k,
        "last_10k_date":  last_10k,
        "has_annual_10k": bool(ten_k_dates),
    }


# ---------------------------------------------------------------------------
# Phase C: Apply filters
# ---------------------------------------------------------------------------

def apply_filters(
    candidates: list[dict],
    *,
    primary_sic_range: set[int],
    manual_tickers: dict[str, dict],
    tier_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply exchange, exclusion, and pre-shock quarter filters.

    Tier assignment (priority order, non-overlapping except placebo_both):
      1. primary_software : SIC in primary_sic_range
         (if also in placebo list → placebo_both)
      2. primary_knowledge: ticker in manual_tickers with tier=primary_knowledge
         and NOT in primary_sic_range
      3. placebo          : ticker in manual_tickers with tier=placebo
         and NOT in primary_sic_range

    Filters applied to ALL tiers:
      - Ticker not in EXCLUDED_TICKERS
      - exchange.upper() in VALID_EXCHANGES
      - n_pre_shock_quarters >= MIN_PRE_SHOCK_QUARTERS
      - For primary_knowledge and placebo: has_annual_10k=True (10-K filers only)

    Auto SIC→NAICS crosswalk is intentionally NOT used for tier membership.
    """
    rows  = []
    total = len(candidates)

    for i, c in enumerate(candidates, 1):
        cik          = c["cik"]
        ticker       = c.get("ticker", "").upper()
        company_name = c.get("company_name", "")
        sic          = c.get("sic")
        exchange_raw = c.get("exchange") or ""
        first_10k    = c.get("first_10k_date")
        last_10k     = c.get("last_10k_date")
        has_10k      = c.get("has_annual_10k", False)

        if (i % 100) == 0:
            logger.info("[%d/%d] filtering…", i, total)

        # --- Tier determination ---
        in_software = (sic is not None) and (sic in primary_sic_range)
        manual_info = manual_tickers.get(ticker)
        in_knowledge = (manual_info is not None) and (manual_info["tier"] == "primary_knowledge")
        in_placebo   = (manual_info is not None) and (manual_info["tier"] == "placebo")

        if in_software and in_placebo:
            tier = "placebo_both"
        elif in_software:
            tier = "primary_software"
        elif in_knowledge:
            tier = "primary_knowledge"
        elif in_placebo:
            tier = "placebo"
        else:
            continue  # not in any tier

        # --- Optional tier filter (--tier flag) ---
        if tier_filter and tier != tier_filter:
            if not (tier_filter == "primary_software" and tier == "placebo_both"):
                continue

        # --- Exclusions ---
        if ticker in EXCLUDED_TICKERS:
            logger.info("  %s: excluded (hardcoded)", ticker)
            continue

        # --- Exchange filter ---
        if exchange_raw.upper() not in VALID_EXCHANGES:
            continue

        # --- 10-K filer requirement for manual tiers ---
        if tier in ("primary_knowledge", "placebo") and not has_10k:
            continue

        # --- Pre-shock quarter count ---
        try:
            cf = get_companyfacts(cik)
        except RuntimeError:
            logger.warning("  %s (CIK %s): companyfacts fetch failed", ticker, cik)
            continue

        n_pre_shock = _count_pre_shock_quarters(cf, PRE_SHOCK_CUTOFF)
        if n_pre_shock < MIN_PRE_SHOCK_QUARTERS:
            logger.info("  %s: only %d pre-shock quarters (< %d), skipped",
                        ticker, n_pre_shock, MIN_PRE_SHOCK_QUARTERS)
            continue

        # --- Consumer flag ---
        name_lower    = company_name.lower()
        consumer_flag = any(hint in name_lower for hint in CONSUMER_NAME_HINTS)

        # --- NAICS (informational only) ---
        naics4 = _SIC_TO_NAICS4.get(sic, "") if sic else ""
        if not naics4 and manual_info:
            naics4 = str(manual_info.get("sector", ""))[:4]

        # --- Sector and expected_rho from manual config ---
        sector_code  = manual_info["sector"]       if manual_info else ""
        expected_rho = manual_info["expected_rho"] if manual_info else None

        rows.append({
            "ticker":               ticker,
            "cik":                  cik,
            "company_name":         company_name,
            "sic":                  sic,
            "naics":                naics4,
            "exchange":             exchange_raw,
            "tier":                 tier,
            "sector_code":          sector_code,
            "expected_rho":         expected_rho,
            "first_10k_date":       first_10k,
            "last_10k_date":        last_10k,
            "n_pre_shock_quarters": n_pre_shock,
            "consumer_flag":        consumer_flag,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase D: Write outputs
# ---------------------------------------------------------------------------

def write_outputs(df: pd.DataFrame, log_entries: list[dict], *, dry_run: bool) -> None:
    if dry_run:
        print("\n[DRY RUN] — no files written")
        print(df.to_string(index=False))
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Wrote %d firms → %s", len(df), OUTPUT_CSV)

    tier_counts = df["tier"].value_counts().to_dict()
    n_consumer  = df["consumer_flag"].sum()

    lines = [
        "=" * 60,
        "FIRM UNIVERSE BUILD SUMMARY",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        f"Total firms: {len(df)}",
        "",
        "By tier:",
    ]
    for tier in ("primary_software", "primary_knowledge", "placebo", "placebo_both"):
        n = tier_counts.get(tier, 0)
        if n:
            lines.append(f"  {tier:22s}: {n}")
    lines += [
        f"  consumer_flagged       : {n_consumer}",
        "",
        "Primary software pre-shock quarter distribution:",
    ]
    prim = df[df["tier"] == "primary_software"]["n_pre_shock_quarters"]
    if len(prim):
        lines += [
            f"  Mean: {prim.mean():.1f}",
            f"  Min:  {prim.min()}",
            f"  Max:  {prim.max()}",
        ]

    if "sector_code" in df.columns:
        lines += ["", "Knowledge tier by sector:"]
        know = df[df["tier"] == "primary_knowledge"]
        for sec, grp in know.groupby("sector_code"):
            lines.append(f"  {sec:15s}: {len(grp):3d}  {', '.join(sorted(grp['ticker']))}")

        lines += ["", "Placebo tier by sector:"]
        plac = df[df["tier"] == "placebo"]
        for sec, grp in plac.groupby("sector_code"):
            lines.append(f"  {sec:15s}: {len(grp):3d}  {', '.join(sorted(grp['ticker']))}")

    lines += ["", "Primary software SIC breakdown:"]
    prim_df = df[df["tier"].isin({"primary_software", "placebo_both"})]
    for sic, cnt in prim_df["sic"].value_counts().sort_index().items():
        lines.append(f"  SIC {sic}: {cnt}")

    summary = "\n".join(lines)
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(summary)
    logger.info("Wrote summary → %s", SUMMARY_TXT)
    print("\n" + summary)

    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOG, "w") as fh:
        for entry in log_entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build three-tier firm universe from SEC EDGAR"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Test on 5 known tickers; do not write output files")
    parser.add_argument("--tier", choices=["primary_software", "primary_knowledge", "placebo"],
                        help="Build a single tier only")
    args = parser.parse_args()

    cfg            = load_config()
    manual_tickers = load_universe_tickers()
    logger.info("Loaded %d manual tickers (%d knowledge, %d placebo)",
                len(manual_tickers),
                sum(1 for v in manual_tickers.values() if v["tier"] == "primary_knowledge"),
                sum(1 for v in manual_tickers.values() if v["tier"] == "placebo"))

    primary_sic_range = set(cfg["tiers"]["primary_software"]["sic_codes"])

    log_entries: list[dict] = [
        {"ts": datetime.utcnow().isoformat(), "event": "start",
         "tier_filter": args.tier, "dry_run": args.dry_run,
         "manual_ticker_count": len(manual_tickers)}
    ]

    if args.dry_run:
        dry_tickers_all = set(DRY_RUN_TICKERS)
        # Also include any manual tickers in dry-run set
        dry_tickers_all |= set(list(manual_tickers.keys())[:5])
        logger.info("DRY RUN: %d tickers: %s", len(dry_tickers_all), sorted(dry_tickers_all))

        tickers_json = get_company_tickers()
        fields     = tickers_json["fields"]
        ticker_idx = fields.index("ticker")
        cik_idx    = fields.index("cik")
        exch_idx   = fields.index("exchange")

        candidates = []
        found: set[str] = set()
        for row in tickers_json["data"]:
            t = str(row[ticker_idx]).upper()
            if t in dry_tickers_all and t not in found:
                found.add(t)
                cik = int(row[cik_idx])
                enriched = enrich_from_submissions(cik)
                if enriched is None:
                    continue
                candidates.append({
                    "cik":      cik,
                    "ticker":   t,
                    "exchange": str(row[exch_idx]),
                    **enriched,
                })
        logger.info("Fetched data for %d dry-run tickers", len(candidates))

    else:
        logger.info("Phase A: downloading company_tickers_exchange.json …")
        listed = get_all_listed_ciks()
        logger.info("  %d unique CIKs", len(listed))

        # Build ticker→row lookup for manual tickers
        ticker_to_row: dict[str, dict] = {
            str(r["ticker"]).upper(): r for r in listed
        }

        logger.info("Phase B: fetching submissions …")
        candidates = []
        seen_ciks: set[int] = set()

        for j, firm in enumerate(listed, 1):
            if j % 500 == 0:
                logger.info("  %d / %d scanned  candidates so far: %d",
                            j, len(listed), len(candidates))
            cik = firm["cik"]
            if cik in seen_ciks:
                continue

            enriched = enrich_from_submissions(cik)
            if enriched is None:
                continue

            sic       = enriched["sic"]
            ticker_up = str(firm["ticker"]).upper()

            in_software  = sic is not None and sic in primary_sic_range
            in_manual    = ticker_up in manual_tickers

            # Tier filter short-circuit
            if args.tier == "primary_software" and not in_software:
                continue
            if args.tier in ("primary_knowledge", "placebo"):
                if not in_manual:
                    continue
                if manual_tickers[ticker_up]["tier"] != args.tier:
                    continue
            if not in_software and not in_manual:
                continue

            seen_ciks.add(cik)
            ticker   = (enriched["tickers"] or [firm["ticker"]])[0]
            exchange = (enriched["exchanges"] or [firm["exchange"]])[0]

            candidates.append({
                "cik":      cik,
                "ticker":   ticker_up,
                "exchange": exchange or "",
                **enriched,
            })

        # Ensure all manual tickers are in candidates even if not in SIC scan
        if args.tier != "primary_software":
            for mticker in manual_tickers:
                row = ticker_to_row.get(mticker)
                if row is None or int(row["cik"]) in seen_ciks:
                    continue
                cik = int(row["cik"])
                enriched = enrich_from_submissions(cik)
                if enriched is None:
                    continue
                seen_ciks.add(cik)
                exchange = (enriched["exchanges"] or [row["exchange"]])[0]
                candidates.append({
                    "cik":      cik,
                    "ticker":   mticker,
                    "exchange": exchange or "",
                    **enriched,
                })
                logger.info("  Added manual ticker outside SIC scan: %s", mticker)

        logger.info("  %d candidates after submissions filter", len(candidates))
        log_entries.append({"ts": datetime.utcnow().isoformat(),
                            "event": "candidates_after_submissions",
                            "n": len(candidates)})

    logger.info("Phase C: applying filters …")
    df = apply_filters(
        candidates,
        primary_sic_range=primary_sic_range,
        manual_tickers=manual_tickers,
        tier_filter=args.tier,
    )

    log_entries.append({"ts": datetime.utcnow().isoformat(),
                        "event": "after_filters", "n": len(df)})

    for tier, grp in df.groupby("tier"):
        logger.info("  %s: %d firms", tier, len(grp))

    logger.info("Phase D: writing outputs …")
    write_outputs(df, log_entries, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
