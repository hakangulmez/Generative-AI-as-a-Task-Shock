"""
Phase A — collect pre-shock 10-K sections for supply-side (ρ) scoring.

For each firm in firm_universe.csv, finds the most recent 10-K filed before
2022-11-01, fetches the HTML, and extracts Item 1 / Item 1A / Item 7.

Input:   data/raw/firm_universe.csv        (321 firms)
Output:  text_data/10k_extracts/{TICKER}.txt
         data/processed/extraction_qa.json

Usage:
    python3 scripts/02_collect_10k_text.py                    # all firms, skip existing
    python3 scripts/02_collect_10k_text.py --tickers MSFT CRM
    python3 scripts/02_collect_10k_text.py --retry-failed     # re-run QA-failed firms
    python3 scripts/02_collect_10k_text.py --no-skip-existing # force full re-collection
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import edgar
from utils.text_sections import extract_sections

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = Path("data/raw/firm_universe.csv")
OUTPUT_DIR  = Path("text_data/10k_extracts")
QA_LOG_PATH = Path("data/processed/extraction_qa.json")
CUTOFF_DATE = "2022-11-01"

_FAILED_STATUSES = frozenset({
    # no_pre_shock_10k intentionally excluded — firm has no pre-shock 10-K (definite)
    "no_primary_doc",
    "fetch_failed",
    "extraction_failed",
})

# Statuses that permanently skip regardless of --retry-failed or missing .txt file.
# --tickers mode bypasses this to allow manual CIK-override re-runs.
_PERMANENT_SKIP_STATUSES = frozenset({
    "no_pre_shock_10k",
    "dropped_extraction_fail",
})


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_accession(acc: str) -> str:
    """Ensure hyphenated accession format: XXXXXXXXXX-YY-ZZZZZZ."""
    flat = acc.replace("-", "")
    if len(flat) == 18:
        return f"{flat[:10]}-{flat[10:12]}-{flat[12:]}"
    return acc


def _write_extract(path: Path, meta: dict, sections: dict) -> None:
    lines = [
        "---",
        f"ticker: {meta['ticker']}",
        f"cik: {meta['cik']}",
        f"company_name: {meta['company_name']}",
        f"filing_date: {meta['filing_date']}",
        f"accession_no: {meta['accession_no']}",
        f"form_type: {meta['form_type']}",
        "---",
    ]
    if sections.get("item_1"):
        lines += ["### ITEM_1_START ###", sections["item_1"]]
    if sections.get("item_1a"):
        lines += ["### ITEM_1A_START ###", sections["item_1a"]]
    if sections.get("item_7"):
        lines += ["### ITEM_7_START ###", sections["item_7"]]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# QA log helpers
# ---------------------------------------------------------------------------

def _load_qa(path: Path) -> dict[str, dict]:
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {e["ticker"]: e for e in entries}
    return {}


def _save_qa(path: Path, qa_map: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(qa_map.values()), indent=2), encoding="utf-8")
    tmp.replace(path)   # atomic on POSIX; on Windows falls back to non-atomic rename


# ---------------------------------------------------------------------------
# Per-firm processing
# ---------------------------------------------------------------------------

def _process_firm(ticker: str, cik: int, company_name: str) -> dict:
    """Fetch and extract pre-shock 10-K sections for one firm.

    Returns a QA record dict regardless of success or failure.
    """
    qa: dict = {
        "ticker":       ticker,
        "cik":          cik,
        "filing_date":  None,
        "accession":    None,
        "form_type":    None,
        "item_1_words":  0,
        "item_1a_words": 0,
        "item_7_words":  0,
        "status":       None,
        "warnings":     [],
    }

    # Step 1 — find pre-shock 10-K (cache hit; no HTTP after Phase 1)
    try:
        filing = edgar.find_pre_shock_10k(cik, CUTOFF_DATE)
    except Exception as exc:
        qa["status"] = "no_pre_shock_10k"
        qa["warnings"].append(f"submissions error: {exc}")
        return qa

    if filing is None:
        qa["status"] = "no_pre_shock_10k"
        return qa

    accession, filing_date, primary_doc_hint, form_type = filing
    qa["filing_date"] = filing_date
    qa["accession"]   = _fmt_accession(accession)
    qa["form_type"]   = form_type

    # Step 2 — resolve primary document filename
    doc_name = (primary_doc_hint or "").strip() or None
    if not doc_name:
        try:
            doc_name = edgar.find_primary_doc(cik, accession)
        except Exception as exc:
            qa["status"] = "no_primary_doc"
            qa["warnings"].append(f"index lookup error: {exc}")
            return qa
        if doc_name is None:
            qa["status"] = "no_primary_doc"
            return qa

    # Step 3 — fetch filing HTML (cached on first hit)
    try:
        html = edgar.get_filing_text(accession, cik, doc_name)
    except Exception as exc:
        qa["status"] = "fetch_failed"
        qa["warnings"].append(f"fetch error: {exc}")
        return qa

    if html is None:
        qa["status"] = "fetch_failed"
        return qa

    # Step 4 — extract sections
    try:
        extracted = extract_sections(html, form_type)
    except Exception as exc:
        qa["status"] = "extraction_failed"
        qa["warnings"].append(f"extraction error: {exc}")
        return qa

    item_1  = extracted.get("item_1")
    item_1a = extracted.get("item_1a")
    item_7  = extracted.get("item_7")
    wc      = extracted.get("length_words", {})

    qa["item_1_words"]  = wc.get("item_1",  0)
    qa["item_1a_words"] = wc.get("item_1a", 0)
    qa["item_7_words"]  = wc.get("item_7",  0)

    if item_1 is None:
        qa["status"] = "extraction_failed"
        qa["warnings"].append("item_1 extraction returned None")
        return qa

    if item_1a is None:
        qa["warnings"].append("item_1a missing — flagged for manual review")
    if item_7 is None:
        qa["warnings"].append("item_7 missing")

    if item_1 and item_1a and item_7:
        qa["status"] = "ok"
    elif item_1 and item_1a:
        qa["status"] = "ok_no_item7"
    else:
        qa["status"] = "ok_item1_only"

    # Step 5 — write structured text file
    meta = {
        "ticker":       ticker,
        "cik":          cik,
        "company_name": company_name,
        "filing_date":  filing_date,
        "accession_no": _fmt_accession(accession),
        "form_type":    form_type,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_extract(OUTPUT_DIR / f"{ticker}.txt", meta,
                   {"item_1": item_1, "item_1a": item_1a, "item_7": item_7})

    return qa


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _should_process(ticker: str, qa_map: dict, skip_existing: bool,
                    retry_failed: bool, explicit_tickers: bool = False) -> bool:
    # --tickers always overrides all skip logic (needed after manual CIK fixes)
    if explicit_tickers:
        return True

    status = qa_map.get(ticker, {}).get("status")
    # Permanent failures are never retried automatically
    if status in _PERMANENT_SKIP_STATUSES:
        return False

    out_path = OUTPUT_DIR / f"{ticker}.txt"
    if not out_path.exists():
        return True                              # new firm — always process
    if not skip_existing:
        return True                              # --no-skip-existing
    if retry_failed:
        return status in _FAILED_STATUSES        # re-run retryable failures only
    return False                                 # file exists, skip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A: collect pre-shock 10-K sections for ρ scoring"
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Re-collect specific tickers only")
    parser.add_argument("--skip-existing", dest="skip_existing",
                        action="store_true", default=True,
                        help="Skip firms that already have a .txt file (default)")
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false",
                        help="Force re-collection even if .txt already exists")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt firms with a failed status in the QA log")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    qa_map = _load_qa(QA_LOG_PATH)

    # --- build candidate list ---
    rows = list(df.itertuples(index=False))

    if args.tickers:
        requested = {t.upper() for t in args.tickers}
        rows = [r for r in rows if r.ticker in requested]
        missing = requested - {r.ticker for r in rows}
        if missing:
            print(f"WARNING: not in firm_universe.csv: {', '.join(sorted(missing))}")

    explicit = bool(args.tickers)
    to_process = [
        r for r in rows
        if _should_process(r.ticker, qa_map, args.skip_existing,
                           args.retry_failed, explicit_tickers=explicit)
    ]

    total = len(to_process)
    already_done = len(rows) - total
    print(f"Firm universe : {len(rows)}")
    print(f"Already done  : {already_done}  (skipped)")
    print(f"To collect    : {total}")
    if args.retry_failed:
        print("  (--retry-failed active: includes QA-failed firms)")
    if not args.skip_existing:
        print("  (--no-skip-existing: full re-collection)")
    print()

    if total == 0:
        print("Nothing to do.")
        return

    t0 = time.time()
    counts: dict[str, int] = {
        "ok": 0, "ok_no_item7": 0, "ok_item1_only": 0,
        "no_pre_shock_10k": 0, "no_primary_doc": 0,
        "fetch_failed": 0, "extraction_failed": 0,
    }

    for i, row in enumerate(to_process, start=1):
        ticker       = row.ticker
        cik          = int(row.cik)
        company_name = row.company_name

        qa = _process_firm(ticker, cik, company_name)
        qa_map[ticker] = qa
        counts[qa["status"]] = counts.get(qa["status"], 0) + 1

        status_tag = qa["status"]
        w1  = qa["item_1_words"]
        w1a = qa["item_1a_words"]
        w7  = qa["item_7_words"]

        if status_tag.startswith("ok"):
            print(f"  {ticker:<8}  {status_tag:<14}  "
                  f"Item1={w1:>6}  Item1A={w1a:>6}  Item7={w7:>6}"
                  f"  {qa['filing_date']}  {qa['form_type']}")
        else:
            warn = qa["warnings"][0] if qa["warnings"] else ""
            print(f"  {ticker:<8}  FAILED [{status_tag}]  {warn}")

        if i % 20 == 0 or i == total:
            elapsed   = time.time() - t0
            rate      = elapsed / i
            remaining = rate * (total - i)
            ok_count  = counts["ok"] + counts["ok_no_item7"] + counts["ok_item1_only"]
            fail_count = total - ok_count - (total - i)   # processed so far
            print(f"  [{i}/{total}] ok={ok_count}  "
                  f"elapsed={elapsed:.0f}s  remaining≈{remaining:.0f}s")

        # Flush QA log every 25 firms so progress survives interruption
        if i % 25 == 0:
            _save_qa(QA_LOG_PATH, qa_map)

    # Final QA log write
    _save_qa(QA_LOG_PATH, qa_map)

    elapsed = time.time() - t0
    ok_total = counts["ok"] + counts["ok_no_item7"] + counts["ok_item1_only"]
    fail_total = sum(v for k, v in counts.items() if k not in ("ok", "ok_no_item7", "ok_item1_only"))

    print()
    print("=" * 60)
    print(f"Collected this run   : {ok_total}")
    print(f"  ok (all 3 sections): {counts['ok']}")
    print(f"  ok_no_item7        : {counts['ok_no_item7']}")
    print(f"  ok_item1_only      : {counts['ok_item1_only']}")
    print(f"Failed this run      : {fail_total}")
    print(f"  no_pre_shock_10k   : {counts['no_pre_shock_10k']}")
    print(f"  no_primary_doc     : {counts['no_primary_doc']}")
    print(f"  fetch_failed       : {counts['fetch_failed']}")
    print(f"  extraction_failed  : {counts['extraction_failed']}")
    print(f"Time elapsed         : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"QA log written to    : {QA_LOG_PATH}")
    print(f"Extracts written to  : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
