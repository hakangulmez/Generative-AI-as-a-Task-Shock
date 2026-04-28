"""
Phase 4 — score supply-side LLM replicability (ρ_i) for all firms.

For each firm:
  1. Load pre-shock 10-K Item 1 from text_data/10k_extracts/{TICKER}.txt
  2. Strip the frontmatter block written by 02_collect_10k_text.py
  3. Pass full Item 1 body to llm_client.score_firm() with SupplyScore schema
  4. Append one row to data/processed/supply_rho.csv (default output)

Modes:
  --test           : score the 14-15 anchor firms from anchor_firms.yaml + PFE
                     (defaults to --iterations 3 unless overridden)
  --tickers T1 T2  : score specific tickers
  --limit N        : score first N tickers from firm_universe (debug helper)
  --skip-existing  : skip tickers already present in output CSV (rho_mean non-null)
  --output PATH    : override default output path
  --iterations N   : number of independent scoring iterations per firm (1-3,
                     default 1 for non-test modes; 3 for --test mode)
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

The model returns only ticker, tasks, and overall_reasoning. Counts and
rho score are computed deterministically by compute_aggregates() from
the validated task labels — the model is never asked to self-report
summary counts, which eliminated a self-consistency failure mode
observed in Step 6a smoke (ZS: count mismatches across all 3 retries).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from utils.schemas import SupplyScore, compute_aggregates
from utils.logging_setup import get_logger

logger = get_logger("supply_score", "08_score_supply_rho")

# =============================================================================
# Paths
# =============================================================================
PROMPT_PATH    = Path("prompts/supply_rho_system.txt")
UNIVERSE_PATH  = Path("data/raw/firm_universe.csv")
TEXT_DIR       = Path("text_data/10k_extracts")
OUTPUT_PATH    = Path("data/processed/supply_rho.csv")
ANCHOR_PATH    = Path("config/anchor_firms.yaml")
ERROR_LOG_PATH = Path("data/processed/supply_rho_errors.jsonl")

# =============================================================================
# Output schema
# =============================================================================
CSV_FIELDS = [
    # Identity
    "ticker", "cik", "tier", "sector_code",
    # Iteration 1 (always present)
    "n_tasks_iter1", "r0_count_iter1", "r1_count_iter1", "r2_count_iter1",
    "raw_exposure_iter1", "rho_iter1",
    "tasks_json_iter1", "overall_reasoning_iter1",
    # Iteration 2 (null when --iterations < 2)
    "n_tasks_iter2", "r0_count_iter2", "r1_count_iter2", "r2_count_iter2",
    "raw_exposure_iter2", "rho_iter2",
    "tasks_json_iter2", "overall_reasoning_iter2",
    # Iteration 3 (null when --iterations < 3)
    "n_tasks_iter3", "r0_count_iter3", "r1_count_iter3", "r2_count_iter3",
    "raw_exposure_iter3", "rho_iter3",
    "tasks_json_iter3", "overall_reasoning_iter3",
    # Aggregates
    "rho_mean", "rho_std", "n_iterations_completed",
    # Cost (summed across iterations)
    "input_tokens", "cache_creation_tokens", "cache_read_tokens", "output_tokens",
    "cost_usd", "retries_used_total",
    "scored_at",
]

# =============================================================================
# Tool definition (passed to llm_client)
# =============================================================================
TOOL_NAME = "submit_supply_score"
TOOL_DESCRIPTION = (
    "Submit a list of 6-12 software product tasks for the firm, each "
    "classified per the R-rubric (R0 / R1 / R2) with a brief reasoning "
    "field. Also submit a 2-3 sentence overall_reasoning describing the "
    "product. Do NOT include count summaries or scores — those are "
    "computed deterministically from your task labels."
)


# =============================================================================
# Exceptions
# =============================================================================
class IncompatibleCSVError(RuntimeError):
    """Raised when output CSV uses an incompatible (pre-Step-5) column schema."""
    pass


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
    """Return set of tickers already in output CSV with a non-null rho_mean.

    A row is considered 'existing' only when rho_mean is non-null, meaning
    all iterations completed successfully.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=["ticker", "rho_mean"])
    mask = df["rho_mean"].notna()
    return set(df.loc[mask, "ticker"].astype(str).str.upper())


def check_csv_compatibility(path: Path) -> None:
    """Raise IncompatibleCSVError if the existing CSV uses the old column schema.

    Old format (pre-Step-5) has column 'e0_count'.
    New format (Step-5+) has column 'r0_count_iter1'.
    """
    if not path.exists() or path.stat().st_size == 0:
        return  # no file or empty file — will write header fresh

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return  # empty file

    if "e0_count" in header:
        raise IncompatibleCSVError(
            f"Output CSV at {path} uses the old E0/E1/E2 column schema "
            f"('e0_count' detected in header). Cannot append Step-5 format rows. "
            f"Options: (1) delete the old file and rerun, "
            f"(2) specify a new output path with --output."
        )


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
# Per-firm scoring (multi-iteration)
# =============================================================================
def score_one_firm_multi_iter(
    ticker: str,
    cik: str,
    tier: str,
    sector_code: str,
    system_prompt: str,
    output_path: Path,
    verify_cache: bool,
    iterations: int,
) -> tuple[bool, dict | None, list[dict] | None]:
    """Score one firm across `iterations` independent runs.

    Returns:
        (success: bool, csv_row: dict | None, per_iter_usage: list[dict] | None)

    On success: appends row to output_path, returns (True, csv_row, per_iter_usage).
    On error (any iteration): logs to ERROR_LOG_PATH, returns (False, None, None).
    No partial rows are written — all iterations must succeed.
    """
    try:
        firm_text = load_firm_text(ticker)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning(f"{ticker}: text load failed: {exc}")
        append_error(ERROR_LOG_PATH, ticker, "text_load_failed", str(exc))
        return False, None, None

    iter_results: list[dict] = []   # parsed field dicts, one per iteration
    per_iter_usage: list[dict] = [] # raw usage dicts from score_firm()

    for i in range(1, iterations + 1):
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
            logger.error(f"{ticker} iter{i}: cache verification failed: {exc}")
            append_error(ERROR_LOG_PATH, ticker, "cache_verification_failed", str(exc))
            raise  # halt execution; user must investigate
        except LLMScoringError as exc:
            logger.warning(f"{ticker} iter{i}: scoring failed after retries: {exc}")
            append_error(ERROR_LOG_PATH, ticker, "scoring_failed", str(exc))
            return False, None, None
        except Exception as exc:
            logger.error(f"{ticker} iter{i}: unexpected error: {type(exc).__name__}: {exc}")
            append_error(ERROR_LOG_PATH, ticker, type(exc).__name__, str(exc))
            return False, None, None

        aggregates = compute_aggregates(result)
        iter_results.append({
            "n_tasks": aggregates["n_tasks"],
            "r0_count": aggregates["r0_count"],
            "r1_count": aggregates["r1_count"],
            "r2_count": aggregates["r2_count"],
            "raw_exposure": aggregates["raw_exposure"],
            "rho": aggregates["normalized_score"],
            "tasks_json": json.dumps([t.model_dump() for t in result.tasks]),
            "overall_reasoning": result.overall_reasoning,
        })
        per_iter_usage.append(usage)

    # Compute aggregates
    rho_values = [d["rho"] for d in iter_results]
    rho_mean = round(sum(rho_values) / len(rho_values), 2)
    rho_std = round(statistics.stdev(rho_values), 2) if len(rho_values) >= 2 else None

    # Aggregate cost / token totals across iterations
    total_input = sum(u["input_tokens"] for u in per_iter_usage)
    total_cache_creation = sum(u["cache_creation_input_tokens"] for u in per_iter_usage)
    total_cache_read = sum(u["cache_read_input_tokens"] for u in per_iter_usage)
    total_output = sum(u["output_tokens"] for u in per_iter_usage)
    total_cost = sum(u["cost_usd"] for u in per_iter_usage)
    total_retries = sum(u["retries_used"] for u in per_iter_usage)

    # Build CSV row — slots beyond `iterations` get empty strings
    row: dict = {
        "ticker": ticker,
        "cik": cik,
        "tier": tier,
        "sector_code": sector_code,
    }

    for i in range(1, 4):
        if i <= iterations:
            d = iter_results[i - 1]
            row[f"n_tasks_iter{i}"] = d["n_tasks"]
            row[f"r0_count_iter{i}"] = d["r0_count"]
            row[f"r1_count_iter{i}"] = d["r1_count"]
            row[f"r2_count_iter{i}"] = d["r2_count"]
            row[f"raw_exposure_iter{i}"] = d["raw_exposure"]
            row[f"rho_iter{i}"] = d["rho"]
            row[f"tasks_json_iter{i}"] = d["tasks_json"]
            row[f"overall_reasoning_iter{i}"] = d["overall_reasoning"]
        else:
            row[f"n_tasks_iter{i}"] = ""
            row[f"r0_count_iter{i}"] = ""
            row[f"r1_count_iter{i}"] = ""
            row[f"r2_count_iter{i}"] = ""
            row[f"raw_exposure_iter{i}"] = ""
            row[f"rho_iter{i}"] = ""
            row[f"tasks_json_iter{i}"] = ""
            row[f"overall_reasoning_iter{i}"] = ""

    row["rho_mean"] = rho_mean
    row["rho_std"] = rho_std if rho_std is not None else ""
    row["n_iterations_completed"] = len(rho_values)
    row["input_tokens"] = total_input
    row["cache_creation_tokens"] = total_cache_creation
    row["cache_read_tokens"] = total_cache_read
    row["output_tokens"] = total_output
    row["cost_usd"] = round(total_cost, 6)
    row["retries_used_total"] = total_retries
    row["scored_at"] = datetime.now(timezone.utc).isoformat()

    # Inline sanity assertions
    assert row["n_iterations_completed"] == iterations, (
        f"{ticker}: completed {row['n_iterations_completed']} but {iterations} requested"
    )
    if iterations > 1:
        assert rho_std is not None, f"{ticker}: rho_std missing despite multi-iter"
        if rho_std > 25:
            sys.stderr.write(
                f"WARN {ticker}: rho_std={rho_std:.2f} > 25 — high cross-iteration variance. "
                f"Iterations: {rho_values}\n"
            )

    append_row(output_path, row)
    return True, row, per_iter_usage


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

    In multi-iter mode, checks run on rho_mean.
    """
    print()
    print("=" * 60)
    print("ANCHOR PATTERN CHECKS (informational, not gating)")
    print("=" * 60)

    by_ticker = {r["ticker"]: r for r in rows}

    def get_score(tkr: str) -> float:
        row = by_ticker[tkr]
        val = row.get("rho_mean")
        if val is not None and val != "":
            return float(val)
        return float(row.get("rho_iter1", 0))

    # Pattern 1: real-time security infrastructure (R0-dominant, low scores)
    realtime_security = ["ZS", "DDOG", "CRWD"]
    for tkr in realtime_security:
        if tkr in by_ticker:
            score = get_score(tkr)
            verdict = "OK" if score <= 30 else "WARN"
            print(f"  [{verdict}] {tkr}: {score} (R0-dominant real-time; expected ≤ 30)")

    # Pattern 2: text-heavy products (R1-dominant, high scores)
    text_heavy = ["HUBS", "LPSN", "EGAN", "ZIP"]
    for tkr in text_heavy:
        if tkr in by_ticker:
            score = get_score(tkr)
            verdict = "OK" if score >= 60 else "WARN"
            print(f"  [{verdict}] {tkr}: {score} (R1-dominant text-heavy; expected ≥ 60)")

    # Pattern 3: placebo (physical product, near floor)
    if "PFE" in by_ticker:
        score = get_score("PFE")
        verdict = "OK" if score <= 15 else "WARN"
        print(f"  [{verdict}] PFE: {score} (placebo physical product; expected ≤ 15)")

    # Pattern 4: knowledge-intensive services (mid-range, varies)
    knowledge = ["MCO", "SPGI", "MSCI", "COUR", "CHGG"]
    for tkr in knowledge:
        if tkr in by_ticker:
            score = get_score(tkr)
            print(f"  [INFO] {tkr}: {score} (knowledge-intensive; sectoral variation expected)")

    # Pattern 5: VEEV (regulated software, mid-range)
    if "VEEV" in by_ticker:
        score = get_score("VEEV")
        verdict = "OK" if 25 <= score <= 55 else "WARN"
        print(f"  [{verdict}] VEEV: {score} (regulated R0/R1/R2 mix; expected 25–55)")

    # Pattern 6: PAYC (text + compliance)
    if "PAYC" in by_ticker:
        score = get_score("PAYC")
        verdict = "OK" if 35 <= score <= 75 else "WARN"
        print(f"  [{verdict}] PAYC: {score} (HCM/payroll R1-heavy; expected 35–75)")

    print()
    print("These are heuristic pattern checks for human review.")
    print("WARNs are flags for sanity inspection, not failures.")
    print("Step 6 will compute ICC(3,1) on rho_iter1/2/3 as the reliability gate.")


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
                        help="Score anchor firms only (soft pattern checks; default --iterations 3)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers already in output CSV (rho_mean non-null)")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Score specific tickers (overrides --test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score first N tickers from firm_universe (debug helper)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output CSV path (default: {OUTPUT_PATH})")
    parser.add_argument("--no-verify-cache", action="store_true",
                        help="Disable cache assertions (use for >5min runs)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help=(
            "Number of independent scoring iterations per firm. "
            "Use 1 for full sample scoring (default for non-test modes). "
            "Use 3 for anchor reliability validation (Step 6). "
            "Each iteration is a separate API call with the same prompt and "
            "firm text but a fresh model invocation; iteration variance is "
            "the noise we are measuring. Defaults to 3 in --test mode."
        ),
    )
    args = parser.parse_args()

    output_path = args.output
    verify_cache = not args.no_verify_cache

    # Resolve iterations: test mode defaults to 3 unless explicitly set
    if args.iterations is not None:
        iterations = args.iterations
    elif args.test:
        iterations = 3
    else:
        iterations = 1

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

    # Check CSV compatibility before any file I/O
    check_csv_compatibility(output_path)

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
    print(f"Iterations:  {iterations}")
    print(f"Output:      {output_path}")
    print(f"Cache check: {'enabled' if verify_cache else 'DISABLED'}")
    if mode == "test":
        print(f"Anchors:     {tickers}")
    print()

    reset_cumulative_state()
    rows_for_anchor_check: list[dict] = []
    all_iter_usage: list[dict] = []  # flattened per-API-call usage records
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

        success, row, per_iter_usage = score_one_firm_multi_iter(
            ticker=ticker,
            cik=cik,
            tier=tier,
            sector_code=sector_code,
            system_prompt=system_prompt,
            output_path=output_path,
            verify_cache=verify_cache,
            iterations=iterations,
        )

        if not success:
            n_failed += 1
            print("FAILED (see error log)")
            continue

        all_iter_usage.extend(per_iter_usage)

        # Console line: show rho_mean in multi-iter mode, rho_iter1 in single-iter
        if iterations > 1:
            rho_display = f"ρ_mean={row['rho_mean']}  std={row['rho_std']}"
        else:
            rho_display = f"ρ={row['rho_iter1']}"
        print(f"{rho_display}  cost=${row['cost_usd']:.5f}  "
              f"retries={row['retries_used_total']}")

        if mode == "test":
            rows_for_anchor_check.append(row)

    elapsed = time.time() - t0

    # Anchor pattern check (soft, mode == 'test' only)
    if mode == "test" and rows_for_anchor_check:
        print_anchor_pattern_checks(rows_for_anchor_check)

    # Cost summary
    print_cost_summary(all_iter_usage, elapsed)

    # Final tallies
    n_successful = len(tickers) - n_failed
    print(f"  Successful:              {n_successful}")
    print(f"  Failed:                  {n_failed}")
    print()
    print(f"Output written to: {output_path}")
    if n_failed > 0:
        print(f"Errors logged to:  {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()
