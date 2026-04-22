"""
Phase 2.10 — AI mention panel from 10-K and 10-Q full filing text.

Per-firm-filing count of AI-related term mentions using position-based
deduplication (longest match wins at any text position). Covers all
10-K and 10-Q filings 2019-01-01 through 2025-12-31.

Input:   data/raw/firm_universe.csv
         config/ai_mention_lexicon.yaml
         EDGAR submissions cache (data/raw/edgar_cache/submissions/)
Output:  data/processed/ai_mention_panel.csv
         data/processed/ai_mention_panel_qa.json
Text cache: text_data/10k10q_extracts/{TICKER}_{ACCESSION_FLAT}.txt

Formula: mention_density = n_mentions_total / filing_words * 1000  (per 1k words)

Dedup algorithm:
  1. Collect all regex match spans across all 22 patterns.
  2. Sort by (start_pos ASC, match_length DESC) — longest wins at same position.
  3. Walk left-to-right, skip any match whose start < previous match's end.

Usage:
    python3 scripts/07_build_ai_mention_panel.py                     # full run
    python3 scripts/07_build_ai_mention_panel.py --dry-run           # 5 firms, no write
    python3 scripts/07_build_ai_mention_panel.py --tickers ZS MSFT   # specific firms
    python3 scripts/07_build_ai_mention_panel.py --no-skip-existing  # re-process all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path to scripts/utils/ (sibling directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from utils.edgar import get_submissions, get_filing_text, find_primary_doc
from utils.text_sections import _html_to_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_UNIVERSE = Path("data/raw/firm_universe.csv")
LEXICON_PATH   = Path("config/ai_mention_lexicon.yaml")
OUTPUT_PATH    = Path("data/processed/ai_mention_panel.csv")
QA_PATH        = Path("data/processed/ai_mention_panel_qa.json")
TEXT_CACHE_DIR = Path("text_data/10k10q_extracts")

TARGET_FORMS = frozenset({"10-K", "10-K/A", "10-K405", "10-Q", "10-Q/A"})
DATE_START   = "2019-01-01"
DATE_END     = "2025-12-31"
SHOCK_DATE   = "2022-10-01"

DRY_RUN_TICKERS = ["MSFT", "ZS", "PFE", "GE", "NYT"]


# ---------------------------------------------------------------------------
# Lexicon loading and pattern compilation
# ---------------------------------------------------------------------------

def _load_patterns(path: Path) -> list[tuple[re.Pattern, str, str]]:
    """Load YAML lexicon; compile all inclusion patterns once.

    Returns list of (compiled_re, pattern_str, category) tuples.
    """
    with open(path, encoding="utf-8") as f:
        lex = yaml.safe_load(f)

    compiled = []
    for entry in lex["inclusion_terms"]:
        pattern  = entry["pattern"]
        category = entry["category"]
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), pattern, category))
        except re.error as exc:
            print(f"WARNING: could not compile pattern '{pattern}': {exc}")
    return compiled


# ---------------------------------------------------------------------------
# Mention counting — position-based deduplication
# ---------------------------------------------------------------------------

def count_mentions(
    text: str,
    patterns: list[tuple[re.Pattern, str, str]],
) -> dict:
    """Count unique non-overlapping AI term mentions.

    Returns dict with n_mentions_total, n_mentions_{category}×3, filing_words.
    """
    raw: list[tuple[int, int, str]] = []
    for regex, _pstr, category in patterns:
        for m in regex.finditer(text):
            raw.append((m.start(), m.end(), category))

    # Sort: start ASC, then length DESC — longest match wins at tie
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    kept: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, cat in raw:
        if start >= last_end:
            kept.append((start, end, cat))
            last_end = end

    return {
        "n_mentions_total":         len(kept),
        "n_mentions_core":          sum(1 for *_, c in kept if c == "core"),
        "n_mentions_named_product": sum(1 for *_, c in kept if c == "named_product"),
        "n_mentions_technical":     sum(1 for *_, c in kept if c == "technical"),
        "filing_words":             len(text.split()),
    }


# ---------------------------------------------------------------------------
# Filing discovery from submissions JSON
# ---------------------------------------------------------------------------

def _iter_filings(subs: dict) -> list[dict]:
    """Extract all in-range 10-K/10-Q entries from a submissions JSON blob."""
    recent       = subs.get("filings", {}).get("recent", {})
    forms        = recent.get("form",            [])
    filing_dates = recent.get("filingDate",      [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate",      [])

    results = []
    for i, form in enumerate(forms):
        form = form.strip()
        if form not in TARGET_FORMS:
            continue
        fdate = filing_dates[i] if i < len(filing_dates) else ""
        if not fdate or not (DATE_START <= fdate <= DATE_END):
            continue
        acc  = accessions[i]   if i < len(accessions)   else ""
        pdoc = primary_docs[i] if i < len(primary_docs) else ""
        rdate = report_dates[i] if i < len(report_dates) else ""
        if not acc:
            continue
        results.append({
            "accession":   acc,
            "filing_date": fdate,
            "report_date": rdate,
            "primary_doc": pdoc,
            "form_type":   form,
        })
    return results


# ---------------------------------------------------------------------------
# period_end helpers
# ---------------------------------------------------------------------------

def _period_to_fq(period_end: str) -> tuple[int, int]:
    """Map YYYY-MM-DD period_end to (fiscal_quarter, fiscal_year) by calendar quarter."""
    try:
        dt = datetime.strptime(period_end, "%Y-%m-%d")
    except ValueError:
        return (0, 0)
    q = (dt.month - 1) // 3 + 1
    return (q, dt.year)


def _estimate_period_end(filing_date: str, form_type: str) -> str:
    """Estimate period_end when reportDate is missing in submissions JSON."""
    try:
        dt = datetime.strptime(filing_date, "%Y-%m-%d")
    except ValueError:
        return filing_date
    offset = timedelta(days=75 if "10-K" in form_type else 35)
    return (dt - offset).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Per-filing plain-text fetch with two-tier cache
# ---------------------------------------------------------------------------

def get_plain_text(
    ticker: str,
    cik: int,
    accession: str,
    primary_doc: str,
) -> Optional[str]:
    """Return plain text for a filing.

    Tier 1: text_data/10k10q_extracts/{TICKER}_{ACCESSION_FLAT}.txt
    Tier 2: HTML from edgar_cache (or network), then _html_to_text → save Tier 1.
    Falls back to find_primary_doc() when the primaryDocument hint is empty.
    """
    accession_flat = accession.replace("-", "")
    cache_path = TEXT_CACHE_DIR / f"{ticker}_{accession_flat}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    # Resolve missing primary_doc hint via filing index
    if not primary_doc:
        primary_doc = find_primary_doc(cik, accession)
    if not primary_doc:
        return None

    html = get_filing_text(accession, cik, primary_doc)
    if html is None:
        return None

    text = _html_to_text(html)
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8", errors="replace")
    return text


# ---------------------------------------------------------------------------
# QA record constructor
# ---------------------------------------------------------------------------

def _qa_record(
    ticker: str,
    cik: int,
    n_in_range: int,
    n_processed: int,
    n_failed: int,
    total_mentions: int,
    pre_mentions: int,
    post_mentions: int,
    warnings: list[str],
) -> dict:
    if n_in_range == 0:
        status = "no_filings_in_range"
    elif n_processed == 0 and n_failed > 0:
        status = "all_failed"
    elif n_failed > 0:
        status = "partial_failure"
    else:
        status = "ok"

    return {
        "ticker":                          ticker,
        "cik":                             cik,
        "n_filings_in_range":              n_in_range,
        "n_filings_processed":             n_processed,
        "n_filings_failed":                n_failed,
        "total_mentions_across_all_filings": total_mentions,
        "pre_shock_mentions":              pre_mentions,
        "post_shock_mentions":             post_mentions,
        "status":                          status,
        "warnings":                        warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.10: AI mention panel from 10-K/10-Q filings"
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Process only these tickers",
    )
    parser.add_argument(
        "--skip-existing", dest="skip_existing",
        action="store_true", default=True,
        help="Skip firms already present in output (default: on)",
    )
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing",
        action="store_false",
        help="Re-process firms even if they already have output rows",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=f"Process {DRY_RUN_TICKERS} only; print stats, do not write files",
    )
    args = parser.parse_args()

    # --- Compile patterns once ---
    patterns = _load_patterns(LEXICON_PATH)
    print(f"Loaded {len(patterns)} patterns from {LEXICON_PATH}")
    print()

    # --- Load firm universe ---
    universe = pd.read_csv(INPUT_UNIVERSE)
    all_firms = universe[["ticker", "cik"]].drop_duplicates().copy()

    # Determine which tickers are explicitly requested
    if args.dry_run:
        requested = set(DRY_RUN_TICKERS)
        print(f"DRY RUN — {len(requested)} firms: {DRY_RUN_TICKERS}")
    elif args.tickers:
        requested = {t.upper() for t in args.tickers}
        print(f"Targeted run — {len(requested)} firm(s): {sorted(requested)}")
    else:
        requested = set(all_firms["ticker"].tolist())
        print(f"Full universe — {len(requested)} firms")

    # --- Resolve existing output; decide what to skip and what to carry over ---
    existing_rows:    list[dict] = []
    skip_tickers:     set[str]  = set()

    if not args.dry_run and OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        if args.skip_existing:
            # Carry all existing rows; skip firms in output that we were about to process
            skip_tickers = set(existing_df["ticker"].unique()) & requested
            existing_rows = existing_df.to_dict("records")
        else:
            # Drop rows for tickers we will re-process; keep everything else
            existing_rows = (
                existing_df[~existing_df["ticker"].isin(requested)]
                .to_dict("records")
            )

    if skip_tickers:
        print(f"Skipping {len(skip_tickers)} already-processed firm(s)")

    firms_to_process = all_firms[
        all_firms["ticker"].isin(requested - skip_tickers)
    ].reset_index(drop=True)

    print(f"To process: {len(firms_to_process)} firm(s)")
    print()

    # --- Main loop ---
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    new_rows:   list[dict] = []
    qa_records: list[dict] = []
    total_n     = len(firms_to_process)

    for idx, row in firms_to_process.iterrows():
        ticker = str(row["ticker"])
        cik    = int(row["cik"])

        print(f"[{idx + 1}/{total_n}] {ticker} (CIK {cik})", end="  ", flush=True)

        # Fetch submissions
        try:
            subs = get_submissions(cik)
        except Exception as exc:
            print(f"FAILED submissions: {exc}")
            qa_records.append(_qa_record(ticker, cik, 0, 0, 0, 0, 0, 0,
                                         [f"submissions error: {exc}"]))
            continue

        filings = _iter_filings(subs)
        n_in_range = len(filings)

        if n_in_range == 0:
            print("0 in-range filings")
            qa_records.append(_qa_record(ticker, cik, 0, 0, 0, 0, 0, 0, []))
            continue

        n_processed = n_failed = total_mentions = pre_mentions = post_mentions = 0
        warnings: list[str] = []

        for f in filings:
            acc        = f["accession"]
            fdate      = f["filing_date"]
            rdate      = f["report_date"]
            pdoc       = f["primary_doc"]
            form_type  = f["form_type"]

            period_end = rdate if rdate else _estimate_period_end(fdate, form_type)
            fq, fy     = _period_to_fq(period_end)

            text = get_plain_text(ticker, cik, acc, pdoc)
            if text is None:
                n_failed += 1
                warnings.append(f"fetch failed: {acc} ({fdate})")
                continue

            counts  = count_mentions(text, patterns)
            n_total = counts["n_mentions_total"]
            words   = counts["filing_words"]
            density = round(n_total / words * 1000, 4) if words > 0 else 0.0

            new_rows.append({
                "ticker":                   ticker,
                "cik":                      cik,
                "period_end":               period_end,
                "fiscal_quarter":           fq,
                "fiscal_year":              fy,
                "form_type":                form_type,
                "filing_date":              fdate,
                "accession":                acc,
                "n_mentions_total":         n_total,
                "n_mentions_core":          counts["n_mentions_core"],
                "n_mentions_named_product": counts["n_mentions_named_product"],
                "n_mentions_technical":     counts["n_mentions_technical"],
                "filing_words":             words,
                "mention_density":          density,
            })

            n_processed    += 1
            total_mentions += n_total
            if period_end < SHOCK_DATE:
                pre_mentions  += n_total
            else:
                post_mentions += n_total

        qa_records.append(_qa_record(
            ticker, cik, n_in_range, n_processed, n_failed,
            total_mentions, pre_mentions, post_mentions, warnings,
        ))

        print(
            f"{n_in_range} filings | {n_processed} ok | {n_failed} failed"
            f" | {total_mentions} mentions (pre={pre_mentions} post={post_mentions})"
        )

    # --- Assemble and write outputs ---
    out_df = pd.DataFrame(existing_rows + new_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["ticker", "period_end", "filing_date"]).reset_index(drop=True)

    if not args.dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUTPUT_PATH, index=False)

        # Merge QA with any existing QA for tickers not processed in this run
        merged_qa = qa_records
        if QA_PATH.exists():
            try:
                old_qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
                new_tickers = {q["ticker"] for q in qa_records}
                merged_qa = [q for q in old_qa if q["ticker"] not in new_tickers] + qa_records
            except Exception:
                pass

        QA_PATH.parent.mkdir(parents=True, exist_ok=True)
        QA_PATH.write_text(json.dumps(merged_qa, indent=2), encoding="utf-8")

        print()
        print(f"Output written to         : {OUTPUT_PATH}  ({len(out_df):,} rows)")
        print(f"QA log written to         : {QA_PATH}  ({len(merged_qa)} firms)")
    else:
        print()
        print("DRY RUN — no files written.")

    # --- Validation report ---
    this_run_qa = qa_records  # only firms processed this run

    print()
    print("=" * 60)
    n_firms_done   = sum(1 for q in this_run_qa if q["n_filings_processed"] > 0)
    n_filings_done = sum(q["n_filings_processed"] for q in this_run_qa)
    n_pre_total    = sum(q["pre_shock_mentions"]   for q in this_run_qa)
    n_post_total   = sum(q["post_shock_mentions"]  for q in this_run_qa)
    n_all_mentions = sum(q["total_mentions_across_all_filings"] for q in this_run_qa)

    print(f"Firms with ≥1 filing processed : {n_firms_done}")
    print(f"Total filings processed        : {n_filings_done:,}")
    print(f"Total mentions (this run)      : {n_all_mentions:,}")
    print(f"  Pre-shock  (<2022Q4)         : {n_pre_total:,}")
    print(f"  Post-shock (≥2022Q4)         : {n_post_total:,}")
    if n_pre_total > 0:
        print(f"  Post/pre ratio               : {n_post_total / n_pre_total:.1f}x  (expect >> 1)")
    print()

    if not out_df.empty:
        # Top 20 by post-shock mentions (across full accumulated output)
        post_df = out_df[out_df["period_end"] >= SHOCK_DATE]
        if not post_df.empty:
            top20 = (
                post_df.groupby("ticker")["n_mentions_total"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
            )
            print("Top 20 firms by post-shock mentions:")
            for tkr, n in top20.items():
                print(f"  {tkr:<8}: {n:,}")
            print()

        # Zero-mention firms
        firm_totals = out_df.groupby("ticker")["n_mentions_total"].sum()
        zero_firms  = sorted(firm_totals[firm_totals == 0].index.tolist())
        print(f"Firms with 0 total mentions   : {len(zero_firms)}")
        if zero_firms:
            print(f"  {zero_firms[:30]}")
        print()

        # MSFT 2019 vs 2023 annual 10-K spot check
        if "MSFT" in out_df["ticker"].values:
            msft_10k = out_df[
                (out_df["ticker"] == "MSFT") &
                (out_df["form_type"].isin(["10-K", "10-K405"]))
            ].sort_values("filing_date")
            msft_2019 = msft_10k[msft_10k["filing_date"].str.startswith("2019")]
            msft_2023 = msft_10k[msft_10k["filing_date"].str.startswith("2023")]
            print("MSFT annual 10-K spot check (should be >> in 2023 vs 2019):")
            for label, subset in [("FY2019", msft_2019), ("FY2023", msft_2023)]:
                if not subset.empty:
                    r = subset.iloc[0]
                    print(f"  {label} ({r['filing_date']}): "
                          f"{r['n_mentions_total']} mentions | "
                          f"density={r['mention_density']:.4f}/1kw | "
                          f"words={r['filing_words']:,}")
            print()

        # Mention density by fiscal year (median across firms)
        if "fiscal_year" in out_df.columns:
            fy_density = (
                out_df.groupby("fiscal_year")["mention_density"]
                .median()
                .sort_index()
            )
            print("Median mention_density by fiscal year:")
            for fy, d in fy_density.items():
                bar = "#" * int(d * 20)
                print(f"  {fy}: {d:.4f}  {bar}")
            print()

    # QA status summary
    status_counts: dict[str, int] = {}
    for q in this_run_qa:
        status_counts[q["status"]] = status_counts.get(q["status"], 0) + 1
    print("QA status (this run):")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<25}: {count}")
    print()

    failed_firms = [q["ticker"] for q in this_run_qa if q["status"] == "all_failed"]
    if failed_firms:
        print(f"All-failed firms: {failed_firms}")


if __name__ == "__main__":
    main()
