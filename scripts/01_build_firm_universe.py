"""
01_build_firm_universe.py — Build PRIMARY and EXTENDED firm universes from SEC EDGAR.

PRIMARY universe  : SIC 7370-7379, NYSE/NASDAQ/NYSE American, ≥6 pre-shock revenue quarters.
EXTENDED universe : UNION of two discovery methods:
  (a) Manual ticker list  — config/extended_universe_tickers.yaml (authoritative)
  (b) Auto SIC→NAICS      — firms outside primary SIC range whose SIC maps to a target NAICS-4
  Output column 'extended_source' ∈ {manual, auto, both} tracks provenance.

NAICS NOTE: SEC EDGAR has NO native NAICS field in any standard endpoint.
  - company_tickers_exchange.json : ['cik', 'name', 'ticker', 'exchange'] only
  - submissions JSON              : has 'sic' (SEC SIC code) but not NAICS
  - companyfacts DEI              : no standard NAICS XBRL tag
NAICS codes in the output are derived from the bundled Census Bureau SIC-to-NAICS-4
crosswalk (_SIC_TO_NAICS4). The 'naics' column should be treated as approximate
(4-digit NAICS prefix, not official firm-level classification).

Usage:
  python3 scripts/01_build_firm_universe.py --dry-run
  python3 scripts/01_build_firm_universe.py --primary-only
  python3 scripts/01_build_firm_universe.py --universe-only primary
  python3 scripts/01_build_firm_universe.py --universe-only extended
  python3 scripts/01_build_firm_universe.py

DO NOT RUN THE FULL BUILD without explicit authorization — it fetches ~10k
submissions + ~800 companyfacts from SEC EDGAR (~10 min even with caching).
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
OUTPUT_CSV  = Path("data/raw/firm_universe.csv")
BUILD_LOG   = Path("logs/universe_build.jsonl")
SUMMARY_TXT = Path("data/raw/universe_build_summary.txt")
CONFIG_PATH = Path("config/universe_filters.yaml")
EXTENDED_TICKERS_PATH = Path("config/extended_universe_tickers.yaml")

# Pre-shock cutoff: count revenue quarters with period_end strictly before this
PRE_SHOCK_CUTOFF       = "2022-11-01"
MIN_PRE_SHOCK_QUARTERS = 6

# Exchange matching — normalize to uppercase for comparison
VALID_EXCHANGES = frozenset({"NYSE", "NASDAQ", "NYSE AMERICAN", "NYSE MKT"})

ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-K405"})

# IIIV: hardcoded exclusion (post-privatization accounting restatements)
EXCLUDED_TICKERS = frozenset({"IIIV"})

# Legacy consumer-name heuristic — lowercase substring match on company_name
# Preserved exactly from build_firm_universe.py
CONSUMER_NAME_HINTS: list[str] = [
    "match group", "bumble", "spotify", "dropbox", "grubhub",
    "shutterstock", "yelp", "zillow", "trivago", "cargurus",
    "angi homeservices", "angi inc", "iac/", "iac inc", "vimeo",
    "fiverr", "etsy", "wix.com", "godaddy",
    # Consumer gaming / entertainment
    "electronic arts", "sports entertainment gaming", "motorsport games",
    "inspired entertainment",
]

# Dry-run tickers (as specified in master prompt)
DRY_RUN_TICKERS = ["ZS", "VEEV", "DDOG", "ZIP", "HUBS"]

# ---------------------------------------------------------------------------
# Census SIC-to-NAICS crosswalk (bundled — EDGAR has no native NAICS)
# Source: Census Bureau 2022 SIC-to-NAICS concordance tables.
# Key: SIC int, Value: 4-digit NAICS prefix string.
# ---------------------------------------------------------------------------
_NAICS_TO_SIC: dict[str, set[int]] = {
    # Software Publishers
    "5112": {7372},
    # Computer Systems Design and Related Services
    "5415": {7371, 7372, 7373, 7374, 7376, 7377, 7378, 7379},
    # Data Processing, Hosting, Related Services
    "5182": {7374},
    # Other Information Services (internet publishing, web portals)
    "5191": {7375, 7379, 7389},
    # Newspaper, Book, and Directory Publishers
    "5111": {2711, 2712, 2721, 2731, 2741},
    # Colleges, Universities, and Professional Schools
    "6113": {8220, 8221},
    # Business and Secretarial/Vocational Schools
    "6114": {8200, 8244, 8249, 8299},
    # Management, Scientific, and Technical Consulting
    "5416": {8742, 8748},
    # Other Financial Activities / Financial Data Services
    "5239": {6199, 6282, 6411, 6159},
}

# Reverse: SIC → NAICS-4 (first entry in _NAICS_TO_SIC order wins on collision)
_SIC_TO_NAICS4: dict[int, str] = {}
for _naics4, _sics in _NAICS_TO_SIC.items():
    for _sic in _sics:
        _SIC_TO_NAICS4.setdefault(_sic, _naics4)

# Also assign NAICS to primary SIC range (7370-7379) that aren't already covered
_SIC_TO_NAICS4.setdefault(7370, "5415")
_SIC_TO_NAICS4.setdefault(7371, "5415")
_SIC_TO_NAICS4.setdefault(7372, "5112")
_SIC_TO_NAICS4.setdefault(7373, "5415")
_SIC_TO_NAICS4.setdefault(7374, "5182")
_SIC_TO_NAICS4.setdefault(7375, "5191")
_SIC_TO_NAICS4.setdefault(7376, "5415")
_SIC_TO_NAICS4.setdefault(7377, "5415")
_SIC_TO_NAICS4.setdefault(7378, "5415")
_SIC_TO_NAICS4.setdefault(7379, "5191")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_manual_extended_tickers() -> dict[str, dict]:
    """
    Load config/extended_universe_tickers.yaml.

    Returns dict: ticker (uppercase) → {naics: str, company: str}.
    """
    with open(EXTENDED_TICKERS_PATH) as f:
        raw = yaml.safe_load(f)
    result: dict[str, dict] = {}
    for entry in raw.get("knowledge_product_tickers", []):
        ticker = str(entry["ticker"]).upper()
        result[ticker] = {
            "naics":   str(entry["naics"])[:4],
            "company": entry.get("company", ""),
        }
    return result


# ---------------------------------------------------------------------------
# Phase A/B: Identify candidates from EDGAR
# ---------------------------------------------------------------------------

def get_all_listed_ciks() -> list[dict]:
    """
    Return all CIKs from SEC's company_tickers_exchange.json.
    Fields: cik, name, ticker, exchange (no SIC, no NAICS).
    """
    raw = get_company_tickers()
    fields = raw["fields"]
    cik_idx      = fields.index("cik")
    name_idx     = fields.index("name")
    ticker_idx   = fields.index("ticker")
    exchange_idx = fields.index("exchange")

    seen_ciks: set[int] = set()
    firms = []
    for row in raw["data"]:
        cik = int(row[cik_idx])
        if cik in seen_ciks:
            continue
        seen_ciks.add(cik)
        firms.append({
            "cik":      cik,
            "name":     row[name_idx],
            "ticker":   str(row[ticker_idx]),
            "exchange": str(row[exchange_idx]),
        })
    return firms


def enrich_from_submissions(cik: int) -> Optional[dict]:
    """
    Fetch submissions JSON for a CIK and extract metadata.

    Returns dict with: sic (int), company_name (str), tickers (list[str]),
    exchanges (list[str]), first_10k_date (str|None), last_10k_date (str|None),
    has_annual_10k (bool, True if any 10-K/10-K/A filing exists).
    Returns None on fetch failure.
    """
    try:
        data = get_submissions(cik)
    except RuntimeError:
        return None

    sic_raw = data.get("sic", "")
    sic = int(sic_raw) if str(sic_raw).isdigit() else None
    company_name = data.get("name", "")
    tickers   = data.get("tickers", [])
    exchanges = data.get("exchanges", [])

    filings = data.get("filings", {}).get("recent", {})
    forms        = filings.get("form", [])
    filing_dates = filings.get("filingDate", [])

    ten_k_dates = [
        d for f, d in zip(forms, filing_dates)
        if f in ANNUAL_FORMS
    ]
    first_10k = min(ten_k_dates) if ten_k_dates else None
    last_10k  = max(ten_k_dates) if ten_k_dates else None

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
    target_naics_codes: set[str],
    primary_sic_range: set[int],
    manual_extended: dict[str, dict],
    extended_only: bool = False,
    primary_only: bool = False,
) -> pd.DataFrame:
    """
    For each candidate firm, apply exchange, quarter, and exclusion filters.
    Assigns universe label: 'primary', 'extended', or 'both'.
    For extended firms, assigns extended_source: 'manual', 'auto', or 'both'.

    Extended universe is the UNION of:
      (a) manual_extended   — tickers in config/extended_universe_tickers.yaml
      (b) auto SIC→NAICS    — firms whose SIC maps to a target NAICS and are
                              outside the primary SIC range

    Quarter counting logic:
      - Fetch companyfacts for each candidate
      - Run extract_quarterly_revenue (from utils.xbrl)
      - Count observations with period_end strictly < PRE_SHOCK_CUTOFF ("2022-11-01")
      - Firm passes if count >= MIN_PRE_SHOCK_QUARTERS (6)

    Exchange filter:
      - Normalize exchange string to uppercase
      - Must be in VALID_EXCHANGES

    Extended-universe additional filter:
      - Must have at least one 10-K filing (has_annual_10k=True)
      - Excludes 20-F and 40-F filers (non-US domicile annual reports)
    """
    rows = []
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

        # --- Determine universe membership ---
        in_primary = (sic is not None) and (sic in primary_sic_range)
        naics4     = _SIC_TO_NAICS4.get(sic, "") if sic else ""

        in_auto_extended   = (naics4 in target_naics_codes) and (not in_primary)
        in_manual_extended = ticker in manual_extended
        in_extended = in_auto_extended or in_manual_extended

        if primary_only and not in_primary:
            continue
        if extended_only and not in_extended:
            continue
        if not in_primary and not in_extended:
            continue

        # --- IIIV exclusion ---
        if ticker in EXCLUDED_TICKERS:
            logger.info("  %s: excluded (IIIV hardcoded)", ticker)
            continue

        # --- Exchange filter ---
        exchange_norm = exchange_raw.upper()
        if exchange_norm not in VALID_EXCHANGES:
            continue

        # --- Extended: must be 10-K filer (no 20-F / 40-F) ---
        if in_extended and not in_primary and not has_10k:
            continue

        # --- Pre-shock quarter count ---
        try:
            cf = get_companyfacts(cik)
        except RuntimeError:
            logger.warning("  %s (CIK %s): companyfacts fetch failed", ticker, cik)
            continue

        n_pre_shock = _count_pre_shock_quarters(cf, PRE_SHOCK_CUTOFF)
        if n_pre_shock < MIN_PRE_SHOCK_QUARTERS:
            continue

        # --- consumer_flag ---
        name_lower    = company_name.lower()
        consumer_flag = any(hint in name_lower for hint in CONSUMER_NAME_HINTS)

        # --- universe label ---
        if in_primary and in_extended:
            universe = "both"
        elif in_primary:
            universe = "primary"
        else:
            universe = "extended"

        # --- extended_source (only meaningful for extended firms) ---
        if universe in ("extended", "both"):
            if in_auto_extended and in_manual_extended:
                extended_source = "both"
            elif in_manual_extended:
                extended_source = "manual"
            else:
                extended_source = "auto"
        else:
            extended_source = ""

        # --- NAICS: prefer manual config value for extended firms ---
        if in_manual_extended and naics4 == "":
            naics4 = manual_extended[ticker]["naics"]

        rows.append({
            "ticker":               ticker,
            "cik":                  cik,
            "company_name":         company_name,
            "sic":                  sic,
            "naics":                naics4,
            "exchange":             exchange_raw,
            "universe":             universe,
            "extended_source":      extended_source,
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

    n_primary  = (df["universe"].isin({"primary", "both"})).sum()
    n_extended = (df["universe"].isin({"extended", "both"})).sum()
    n_both     = (df["universe"] == "both").sum()
    n_consumer = df["consumer_flag"].sum()

    # Extended source breakdown
    ext_df = df[df["universe"].isin({"extended", "both"})]
    n_manual = (ext_df["extended_source"].isin({"manual", "both"})).sum()
    n_auto   = (ext_df["extended_source"].isin({"auto",   "both"})).sum()
    n_ext_both = (ext_df["extended_source"] == "both").sum()

    lines = [
        "=" * 60,
        "FIRM UNIVERSE BUILD SUMMARY",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        f"Total passing firms:   {len(df)}",
        f"  Primary (SIC 7370-7379): {n_primary}",
        f"  Extended (total):        {n_extended}",
        f"    of which manual-only:  {n_manual - n_ext_both}",
        f"    of which auto-only:    {n_auto - n_ext_both}",
        f"    of which both:         {n_ext_both}",
        f"  In both universes:       {n_both}",
        f"  Consumer-flagged:        {n_consumer}",
        "",
        "Pre-shock quarter distribution (primary):",
    ]
    prim = df[df["universe"].isin({"primary", "both"})]["n_pre_shock_quarters"]
    if len(prim):
        lines += [
            f"  Mean: {prim.mean():.1f}",
            f"  Min:  {prim.min()}",
            f"  Max:  {prim.max()}",
        ]
    lines += ["", "SIC breakdown (primary):"]
    prim_df = df[df["universe"].isin({"primary", "both"})]
    for sic, cnt in prim_df["sic"].value_counts().sort_index().items():
        lines.append(f"  SIC {sic}: {cnt}")

    if len(ext_df):
        lines += ["", "Extended firms (manual list):"]
        manual_firms = ext_df[ext_df["extended_source"].isin({"manual", "both"})][
            ["ticker", "naics", "extended_source"]
        ].sort_values("ticker")
        for _, row in manual_firms.iterrows():
            lines.append(f"  {row['ticker']:8s}  NAICS {row['naics']}  [{row['extended_source']}]")

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
        description="Build primary and extended firm universes from SEC EDGAR"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Test on 5 known tickers; do not write output files")
    parser.add_argument("--primary-only", action="store_true",
                        help="Build primary SIC universe only (faster)")
    parser.add_argument("--universe-only", choices=["primary", "extended"],
                        help="Build a single universe")
    args = parser.parse_args()

    primary_only  = args.primary_only or (args.universe_only == "primary")
    extended_only = (args.universe_only == "extended")

    cfg            = load_config()
    manual_extended = load_manual_extended_tickers()
    logger.info("Loaded %d manual extended tickers from config", len(manual_extended))

    primary_sic_range = set(cfg["primary"]["sic_codes"])
    target_naics_raw  = [str(n) for n in cfg["extended"]["naics_codes"]]
    target_naics4     = {n[:4] for n in target_naics_raw}

    log_entries: list[dict] = [
        {"ts": datetime.utcnow().isoformat(), "event": "start",
         "primary_only": primary_only, "extended_only": extended_only,
         "dry_run": args.dry_run,
         "manual_extended_count": len(manual_extended)}
    ]

    if args.dry_run:
        # Dry-run: test 5 known tickers (primary) + a sample of manual extended tickers
        dry_extended_sample = ["MCO", "CHGG", "COUR", "ACN", "NYT"]
        dry_tickers_all = set(DRY_RUN_TICKERS) | set(dry_extended_sample)
        logger.info("DRY RUN: fetching %d tickers (%s)",
                    len(dry_tickers_all), sorted(dry_tickers_all))

        tickers_json = get_company_tickers()
        fields     = tickers_json["fields"]
        ticker_idx = fields.index("ticker")
        cik_idx    = fields.index("cik")
        exch_idx   = fields.index("exchange")
        name_idx   = fields.index("name")

        candidates = []
        found_tickers: set[str] = set()
        for row in tickers_json["data"]:
            t = str(row[ticker_idx]).upper()
            if t in dry_tickers_all and t not in found_tickers:
                found_tickers.add(t)
                cik = int(row[cik_idx])
                enriched = enrich_from_submissions(cik)
                if enriched is None:
                    continue
                candidates.append({
                    "cik":     cik,
                    "ticker":  t,
                    "exchange": str(row[exch_idx]),
                    **enriched,
                })
        logger.info("Fetched enriched data for %d dry-run tickers", len(candidates))

    else:
        logger.info("Phase A: downloading company_tickers_exchange.json …")
        listed = get_all_listed_ciks()
        logger.info("  %d unique CIKs in company_tickers_exchange.json", len(listed))

        # Build a ticker→CIK lookup for manual extended tickers not in SIC range
        ticker_to_cik_map: dict[str, int] = {
            str(r["ticker"]).upper(): int(r["cik"])
            for r in listed
        }

        logger.info("Phase B: fetching submissions for SIC candidates + manual extended …")
        candidates = []
        seen_ciks: set[int] = set()

        for j, firm in enumerate(listed, 1):
            if j % 500 == 0:
                logger.info("  Submissions fetched: %d / %d  candidates so far: %d",
                            j, len(listed), len(candidates))
            cik = firm["cik"]
            if cik in seen_ciks:
                continue

            enriched = enrich_from_submissions(cik)
            if enriched is None:
                continue

            sic    = enriched["sic"]
            naics4 = _SIC_TO_NAICS4.get(sic, "") if sic else ""

            in_primary    = sic is not None and sic in primary_sic_range
            in_auto_ext   = (naics4 in target_naics4) and not in_primary
            ticker_upper  = str(firm["ticker"]).upper()
            in_manual_ext = ticker_upper in manual_extended

            if primary_only and not in_primary:
                continue
            if extended_only and not (in_auto_ext or in_manual_ext):
                continue
            if not in_primary and not in_auto_ext and not in_manual_ext:
                continue

            seen_ciks.add(cik)
            ticker   = (enriched["tickers"] or [firm["ticker"]])[0]
            exchange = (enriched["exchanges"] or [firm["exchange"]])[0]

            candidates.append({
                "cik":      cik,
                "ticker":   ticker,
                "exchange": exchange,
                **enriched,
            })

        # Add any manual extended tickers not yet seen (e.g. listed under different SIC)
        if not primary_only:
            for mticker, minfo in manual_extended.items():
                cik = ticker_to_cik_map.get(mticker)
                if cik is None or cik in seen_ciks:
                    continue
                enriched = enrich_from_submissions(cik)
                if enriched is None:
                    continue
                seen_ciks.add(cik)
                exchange = (enriched["exchanges"] or ["UNKNOWN"])[0]
                candidates.append({
                    "cik":      cik,
                    "ticker":   mticker,
                    "exchange": exchange,
                    **enriched,
                })
                logger.info("  Added manual extended ticker not in SIC scan: %s", mticker)

        logger.info("  %d candidates after submissions filter", len(candidates))
        log_entries.append({"ts": datetime.utcnow().isoformat(),
                            "event": "candidates_after_submissions",
                            "n": len(candidates)})

    logger.info("Phase C: applying filters (exchange + pre-shock quarters) …")
    df = apply_filters(
        candidates,
        target_naics_codes=target_naics4,
        primary_sic_range=primary_sic_range,
        manual_extended=manual_extended,
        extended_only=extended_only,
        primary_only=primary_only,
    )

    log_entries.append({"ts": datetime.utcnow().isoformat(),
                        "event": "after_filters", "n": len(df)})

    n_primary  = int((df["universe"].isin({"primary", "both"})).sum())
    n_extended = int((df["universe"].isin({"extended", "both"})).sum())
    logger.info("Passing firms: %d total  (%d primary, %d extended)",
                len(df), n_primary, n_extended)

    # Sanity check: primary should reproduce legacy firms
    if not args.dry_run and not extended_only:
        legacy_path = Path("data/raw/firm_universe.csv")
        if legacy_path.exists():
            legacy_df = pd.read_csv(legacy_path)
            if "meets_filters" in legacy_df.columns:
                legacy_passing = set(legacy_df[legacy_df["meets_filters"] == True]["ticker"])
            else:
                legacy_passing = set(legacy_df["ticker"])
            new_primary = set(df[df["universe"].isin({"primary", "both"})]["ticker"])
            only_in_legacy = legacy_passing - new_primary
            only_in_new    = new_primary - legacy_passing
            if only_in_legacy or only_in_new:
                logger.warning("Primary universe diff vs legacy CSV:")
                logger.warning("  Only in legacy (%d): %s",
                               len(only_in_legacy), sorted(only_in_legacy)[:20])
                logger.warning("  Only in new    (%d): %s",
                               len(only_in_new),    sorted(only_in_new)[:20])
            else:
                logger.info("Primary universe matches legacy CSV exactly (%d firms)",
                            len(new_primary))

    logger.info("Phase D: writing outputs …")
    write_outputs(df, log_entries, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
