"""
Phase 4 — score supply-side LLM replicability (ρ_i) for all firms.

For each firm:
  1. Load pre-shock 10-K Item 1 from text_data/10k_extracts/{TICKER}.txt
  2. Strip the frontmatter block written by 02_collect_10k_text.py
  3. Pass full Item 1 body to llm_client.score_firm() with SupplyScore schema
  4. Append one row to data/processed/lit_scores.csv

Modes:
  --test           : score the 14-15 anchor firms from anchor_firms.yaml + PFE
  --tickers T1 T2  : score specific tickers
  --limit N        : score first N tickers from firm_universe (debug helper)
  --skip-existing  : skip tickers already present in output CSV
  --output PATH    : override default output path
  --no-verify-cache : disable cache assertions (use for runs spanning
                      >5 minutes where TTL expiry is expected; cache
                      verification is on by default)

Anchor checks (--test mode only) are SOFT pattern warnings:
  - real-time security firms (ZS, DDOG, CRWD) should land low (≤ 30)
  - text-heavy firms (HUBS, LPSN, EGAN, ZIP) should land high (≥ 60)
  - placebo (PFE) should land near floor (≤ 15)

These are sanity heuristics for human review — they do NOT halt execution
or trigger prompt iteration on their own. The Phase 3 design principle is
that scores emerge deterministically from rubric matching; pulling toward
yaml numeric anchors would re-introduce researcher discretion.

Strict formula reconciliation lives in scripts/utils/schemas.py
(SupplyScore.check_formula_reconciliation). If the model adjusts a score
away from the (e1 + 0.5*e2)/n_tasks * 99 + 1 formula, the Pydantic
validator rejects and llm_client retries. By the time score_firm() returns
here, the score is formula-consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.llm_client import (
    score_firm, aggregate_costs, reset_cumulative_state,
    LLMScoringError, CacheVerificationError,
)
from utils.schemas import SupplyScore
from utils.logging_setup import get_logger

logger = get_logger("supply_score", "08_score_supply_rho")

# =============================================================================
# Paths
# =============================================================================
PROMPT_PATH    = Path("prompts/supply_rho_system.txt")
UNIVERSE_PATH  = Path("data/raw/firm_universe.csv")
TEXT_DIR       = Path("text_data/10k_extracts")
OUTPUT_PATH    = Path("data/processed/lit_scores.csv")
ANCHOR_PATH    = Path("config/anchor_firms.yaml")
ERROR_LOG_PATH = Path("data/processed/lit_scores_errors.jsonl")

# =============================================================================
# Output schema
# =============================================================================
CSV_FIELDS = [
    "ticker", "cik", "tier", "sector_code",
    "n_tasks", "e0_count", "e1_count", "e2_count",
    "raw_exposure", "normalized_score",
    "reasoning", "tasks_json",
    "input_tokens", "cache_creation_tokens", "cache_read_tokens", "output_tokens",
    "cost_usd", "retries_used",
    "scored_at",
]

# =============================================================================
# Tool definition (passed to llm_client)
# =============================================================================
TOOL_NAME = "submit_supply_score"
TOOL_DESCRIPTION = (
    "Submit the supply-side LLM replicability score for the firm based on "
    "E0/E1/E2 task classification. The normalized_score must be computed "
    "deterministically from the formula (e1 + 0.5*e2) / n_tasks * 99 + 1. "
    "Do not adjust manually."
)


# =============================================================================
# Helpers
# =============================================================================
def load_prompt() -> str:
    """Load supply prompt from disk. Verifies non-empty."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"{PROMPT_PATH} is empty")
    return text


def load_firm_text(ticker: str) -> str:
    """Load Item 1 Business Description text for one ticker.

    The 02_collect_10k_text.py output format wraps content in:
        ---
        ticker: XXX
        cik: ...
        ...
        ---
        ### ITEM_1_START ###
        <Item 1 body>
        ### ITEM_1A_START ###
        <Item 1A body>
        ### ITEM_7_START ###
        <Item 7 body>

    We extract ONLY the Item 1 section (between ### ITEM_1_START ### and
    the next section marker, or end of file if Item 1A is absent).

    Methodology rationale: Item 1 (Business Description) is the canonical
    source for product task identification under the Eloundou framework
    applied at the product level. Item 1A (Risk Factors) introduces
    "AI could disrupt our business"-style risk language that biases the
    model toward higher exposure classifications. Item 7 (MD&A) is
    financial commentary, not product description.

    Empirical evidence (diagnostic 2026-04-25):
        Item 1 mean: ~13K tokens (range: 1K to 174K)
        Full body mean: ~62K tokens
        Item 1 alone keeps full-run cost within ~$10 budget.

    No truncation within Item 1 itself. Even GE's 174K-char Item 1 is
    passed in full at ~$0.043/call uncached.
    """
    path = TEXT_DIR / f"{ticker}.txt"
    if not path.exists():
        raise FileNotFoundError(f"10-K extract missing: {path}")

    full = path.read_text(encoding="utf-8")

    item1_marker = "### ITEM_1_START ###"
    item1a_marker = "### ITEM_1A_START ###"
    item7_marker = "### ITEM_7_START ###"

    start_idx = full.find(item1_marker)
    if start_idx == -1:
        # No Item 1 marker found. Fall back to post-frontmatter body to
        # avoid breaking firms with unusual extract formats.
        lines = full.split("\n")
        if lines and lines[0].strip() == "---":
            try:
                second_marker = lines.index("---", 1)
                body = "\n".join(lines[second_marker + 1:])
            except ValueError:
                body = full
        else:
            body = full
        if not body.strip():
            raise RuntimeError(f"{path} has no Item 1 marker and empty body")
        return body

    # Item 1 starts after its marker line
    body_start = start_idx + len(item1_marker)

    # Find the earliest subsequent section marker
    end_candidates = []
    for marker in (item1a_marker, item7_marker):
        idx = full.find(marker, body_start)
        if idx != -1:
            end_candidates.append(idx)
    body_end = min(end_candidates) if end_candidates else len(full)

    item1_body = full[body_start:body_end].strip()

    if not item1_body:
        raise RuntimeError(f"{path} Item 1 section is empty")

    return item1_body


def load_anchor_tickers() -> list[str]:
    """Load supply anchor tickers from config/anchor_firms.yaml.

    Reads keys only — numeric `rho` values are deliberately ignored
    (Phase 3 made them obsolete). PFE is added explicitly because the
    prompt includes it as the 15th example but the yaml is missing it.
    PFE is skipped if its 10-K extract doesn't exist on disk.
    """
    with open(ANCHOR_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tickers = list(data.get("supply_anchors", {}).keys())

    if "PFE" not in tickers:
        if (TEXT_DIR / "PFE.txt").exists():
            tickers.append("PFE")
        else:
            logger.info("PFE 10-K extract not found; skipping in test mode")

    return tickers


def load_existing_tickers(path: Path) -> set[str]:
    """Return set of tickers already in output CSV (for --skip-existing)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["ticker"])
    return set(df["ticker"].astype(str).str.upper())


def append_row(path: Path, row: dict) -> None:
    """Append one row to the output CSV. Writes header if file is new.

    Uses 'a' mode for atomic-ish per-row append. If the script is
    interrupted mid-row, that row is lost but earlier rows are intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_error(path: Path, ticker: str, error_type: str, message: str) -> None:
    """Append one JSONL row to the error log (no header, line-delimited JSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ticker": ticker,
            "error_type": error_type,
            "message": message,
            "ts": datetime.now(timezone.utc).isoformat(),
        }) + "\n")


# =============================================================================
# Per-firm scoring
# =============================================================================
def score_one_firm(
    ticker: str,
    cik: str,
    tier: str,
    sector_code: str,
    system_prompt: str,
    output_path: Path,
    verify_cache: bool,
) -> tuple[bool, dict | None, dict | None]:
    """Score one firm. Returns (success, row, usage_record).

    On success: appends row to output_path, returns (True, row, usage_record).
    On error: logs to ERROR_LOG_PATH, returns (False, None, None).

    The row dict is the same dict written to CSV — caller can use it for
    downstream summaries (anchor pattern checks, console printing) without
    re-reading the CSV from disk.
    """
    try:
        firm_text = load_firm_text(ticker)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning(f"{ticker}: text load failed: {exc}")
        append_error(ERROR_LOG_PATH, ticker, "text_load_failed", str(exc))
        return False, None, None

    try:
        result, usage = score_firm(
            system_prompt=system_prompt,
            firm_text=firm_text,
            ticker=ticker,
            schema=SupplyScore,
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            verify_cache=verify_cache,
        )
    except CacheVerificationError as exc:
        # Cache failures are programmer errors, not data errors — surface them
        logger.error(f"{ticker}: cache verification failed: {exc}")
        append_error(ERROR_LOG_PATH, ticker, "cache_verification_failed", str(exc))
        raise  # halt execution; user must investigate
    except LLMScoringError as exc:
        logger.warning(f"{ticker}: scoring failed after retries: {exc}")
        append_error(ERROR_LOG_PATH, ticker, "scoring_failed", str(exc))
        return False, None, None
    except Exception as exc:
        logger.error(f"{ticker}: unexpected error: {type(exc).__name__}: {exc}")
        append_error(ERROR_LOG_PATH, ticker, type(exc).__name__, str(exc))
        return False, None, None

    # Build CSV row from validated SupplyScore + usage_record
    row = {
        "ticker": result.ticker,
        "cik": cik,
        "tier": tier,
        "sector_code": sector_code,
        "n_tasks": result.n_tasks,
        "e0_count": result.e0_count,
        "e1_count": result.e1_count,
        "e2_count": result.e2_count,
        "raw_exposure": round(result.raw_exposure, 4),
        "normalized_score": round(result.normalized_score, 1),
        "reasoning": result.reasoning,
        "tasks_json": json.dumps([t.model_dump() for t in result.tasks]),
        "input_tokens": usage["input_tokens"],
        "cache_creation_tokens": usage["cache_creation_input_tokens"],
        "cache_read_tokens": usage["cache_read_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": usage["cost_usd"],
        "retries_used": usage["retries_used"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    append_row(output_path, row)

    return True, row, usage


# =============================================================================
# Soft anchor checks
# =============================================================================
def print_anchor_pattern_checks(rows: list[dict]) -> None:
    """Print SOFT pattern warnings for anchor results.

    These are sanity heuristics for human review — they do NOT halt
    execution, do NOT trigger prompt iteration, and do NOT compute
    Spearman/MAD/outlier counts against yaml `rho` values.

    Phase 3 design principle: scores emerge from rubric matching, not
    from numeric anchor matching. The yaml `rho` values are stale
    Phase 3 holdover and were deliberately removed from the prompt.
    """
    print()
    print("=" * 60)
    print("ANCHOR PATTERN CHECKS (informational, not gating)")
    print("=" * 60)

    by_ticker = {r["ticker"]: r for r in rows}

    # Pattern 1: real-time security infrastructure (low scores expected)
    realtime_security = ["ZS", "DDOG", "CRWD"]
    for tkr in realtime_security:
        if tkr in by_ticker:
            score = by_ticker[tkr]["normalized_score"]
            verdict = "OK" if score <= 30 else "WARN"
            print(f"  [{verdict}] {tkr}: {score} (real-time security; expected ≤ 30)")

    # Pattern 2: text-heavy products (high scores expected)
    text_heavy = ["HUBS", "LPSN", "EGAN", "ZIP"]
    for tkr in text_heavy:
        if tkr in by_ticker:
            score = by_ticker[tkr]["normalized_score"]
            verdict = "OK" if score >= 60 else "WARN"
            print(f"  [{verdict}] {tkr}: {score} (text-heavy E1; expected ≥ 60)")

    # Pattern 3: placebo (physical product, near floor)
    if "PFE" in by_ticker:
        score = by_ticker["PFE"]["normalized_score"]
        verdict = "OK" if score <= 15 else "WARN"
        print(f"  [{verdict}] PFE: {score} (placebo physical product; expected ≤ 15)")

    # Pattern 4: knowledge-intensive services (mid-range, varies)
    knowledge = ["MCO", "SPGI", "MSCI", "COUR", "CHGG"]
    for tkr in knowledge:
        if tkr in by_ticker:
            score = by_ticker[tkr]["normalized_score"]
            print(f"  [INFO] {tkr}: {score} (knowledge-intensive; sectoral variation expected)")

    # Pattern 5: VEEV (regulated software, mid-range)
    if "VEEV" in by_ticker:
        score = by_ticker["VEEV"]["normalized_score"]
        verdict = "OK" if 25 <= score <= 55 else "WARN"
        print(f"  [{verdict}] VEEV: {score} (regulated mixed E1/E2; expected 25–55)")

    # Pattern 6: PAYC (text + compliance)
    if "PAYC" in by_ticker:
        score = by_ticker["PAYC"]["normalized_score"]
        verdict = "OK" if 35 <= score <= 75 else "WARN"
        print(f"  [{verdict}] PAYC: {score} (HCM/payroll text-heavy; expected 35–75)")

    print()
    print("These are heuristic pattern checks for human review.")
    print("WARNs are flags for sanity inspection, not failures.")
    print("Phase 4 Day 3 will add Eloundou aggregate cross-validation as the")
    print("literature-grounded informational benchmark (see phase6_notes.md).")


# =============================================================================
# Cost summary
# =============================================================================
def print_cost_summary(usage_records: list[dict], elapsed_sec: float) -> None:
    """Print final cost / cache / timing summary."""
    if not usage_records:
        print("\nNo successful calls — no usage to report.")
        return

    totals = aggregate_costs(usage_records)

    print()
    print("=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"  Calls:                   {totals['n_calls']}")
    print(f"  Total cost:              ${totals['total_cost_usd']:.4f}")
    print(f"  Mean per call:           ${totals['mean_cost_per_call']:.5f}")
    print(f"  Cache hit ratio:         {totals['cache_hit_ratio_overall']:.3f}")
    print(f"  Input tokens (uncached): {totals['total_input_tokens']:,}")
    print(f"  Cache write tokens:      {totals['total_cache_write_tokens']:,}")
    print(f"  Cache read tokens:       {totals['total_cache_read_tokens']:,}")
    print(f"  Output tokens:           {totals['total_output_tokens']:,}")
    print(f"  Elapsed:                 {elapsed_sec:.1f} sec ({elapsed_sec/60:.1f} min)")
    print()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: score supply-side LLM replicability (ρ_i)"
    )
    parser.add_argument("--test", action="store_true",
                        help="Score anchor firms only (~$0.50, soft pattern checks)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers already in output CSV")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Score specific tickers (overrides --test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score first N tickers from firm_universe (debug helper)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output CSV path (default: {OUTPUT_PATH})")
    parser.add_argument("--no-verify-cache", action="store_true",
                        help="Disable cache assertions (use for >5min runs)")
    args = parser.parse_args()

    output_path = args.output
    verify_cache = not args.no_verify_cache

    # Determine ticker list
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        mode = "explicit"
    elif args.test:
        tickers = load_anchor_tickers()
        mode = "test"
    else:
        df = pd.read_csv(UNIVERSE_PATH)
        tickers = df["ticker"].astype(str).tolist()
        if args.limit is not None:
            tickers = tickers[:args.limit]
        mode = "full" if args.limit is None else f"limited({args.limit})"

    # Skip existing if requested
    if args.skip_existing:
        existing = load_existing_tickers(output_path)
        before = len(tickers)
        tickers = [t for t in tickers if t not in existing]
        skipped = before - len(tickers)
        if skipped:
            print(f"Skipping {skipped} ticker(s) already in {output_path}")

    if not tickers:
        print("Nothing to score.")
        return

    # Load metadata for tier / sector / cik enrichment
    universe = pd.read_csv(UNIVERSE_PATH).set_index("ticker")
    universe.index = universe.index.astype(str)

    # Load prompt
    system_prompt = load_prompt()

    # Print run banner
    print(f"Mode:        {mode}")
    print(f"Tickers:     {len(tickers)}")
    print(f"Output:      {output_path}")
    print(f"Cache check: {'enabled' if verify_cache else 'DISABLED'}")
    if mode == "test":
        print(f"Anchors:     {tickers}")
    print()

    reset_cumulative_state()
    rows_for_anchor_check: list[dict] = []
    usage_records: list[dict] = []
    n_failed = 0

    t0 = time.time()
    for i, ticker in enumerate(tickers, start=1):
        # Look up universe metadata (graceful if absent)
        if ticker in universe.index:
            meta = universe.loc[ticker]
            cik = str(meta["cik"])
            tier = str(meta.get("tier", ""))
            sector_code = str(meta.get("sector_code", ""))
        else:
            cik, tier, sector_code = "", "", ""
            logger.warning(f"{ticker} not in firm_universe.csv")

        print(f"[{i}/{len(tickers)}] {ticker}", end="  ", flush=True)

        success, row, usage = score_one_firm(
            ticker=ticker,
            cik=cik,
            tier=tier,
            sector_code=sector_code,
            system_prompt=system_prompt,
            output_path=output_path,
            verify_cache=verify_cache,
        )

        if not success:
            n_failed += 1
            print("FAILED (see error log)")
            continue

        usage_records.append(usage)

        # Console line: read directly from the row dict (pure Python types,
        # no CSV round-trip, no numpy.float64 contamination)
        print(f"ρ={row['normalized_score']}  cost=${row['cost_usd']:.5f}  "
              f"retries={row['retries_used']}")

        # Anchor mode: collect rows for soft pattern check
        if mode == "test":
            rows_for_anchor_check.append(row)

    elapsed = time.time() - t0

    # Anchor pattern check (soft, mode == 'test' only)
    if mode == "test" and rows_for_anchor_check:
        print_anchor_pattern_checks(rows_for_anchor_check)

    # Cost summary
    print_cost_summary(usage_records, elapsed)

    # Final tallies
    print(f"  Successful:              {len(usage_records)}")
    print(f"  Failed:                  {n_failed}")
    print()
    print(f"Output written to: {output_path}")
    if n_failed > 0:
        print(f"Errors logged to:  {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()
