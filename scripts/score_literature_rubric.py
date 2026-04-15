"""
Score firms using the Literature Rubric via Claude API.

Reads Item 1 text from 10-K extracts, sends to Claude with the
literature rubric system prompt, parses JSON scores.

Usage:
  python3 scripts/score_literature_rubric.py --test           # 5 test firms
  python3 scripts/score_literature_rubric.py --tickers MSFT CRM
  python3 scripts/score_literature_rubric.py                  # all firms
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Optional

import anthropic
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTRACTS_DIR = "text_data/10k_extracts"
SYSTEM_PROMPT_PATH = "prompts/literature_rubric_system.txt"
OUTPUT_CSV = "data/processed/lit_scores.csv"

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2048
RATE_LIMIT = 2.0  # seconds between API calls

ITEM1A_SEPARATOR = "### ITEM_1A_START ###"
TEST_TICKERS = ["ASAN", "CRWD", "DDOG", "NOW", "ZIP"]

CRITERIA = [f"C{i}" for i in range(1, 11)]
WEIGHTS = {
    "C1": 3, "C2": 3, "C3": 2, "C4": 2, "C5": 2,
    "C6": 3, "C7": 3, "C8": 2, "C9": 2, "C10": 2,
}
POSITIVE = {"C1", "C2", "C3", "C4", "C5"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_item1(ticker: str) -> Optional[str]:
    """Load Item 1 text (before Item 1A separator), skip header."""
    path = os.path.join(EXTRACTS_DIR, f"{ticker}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    # Split at Item 1A separator
    if ITEM1A_SEPARATOR in text:
        text = text.split(ITEM1A_SEPARATOR)[0]

    # Remove header block (between --- delimiters)
    lines = text.split("\n")
    content_start = 0
    dash_count = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            dash_count += 1
            if dash_count == 2:
                content_start = i + 1
                break

    content = "\n".join(lines[content_start:]).strip()
    return content if content else None


def compute_scores(scores: dict) -> tuple[int, float]:
    """Compute raw and normalized scores from criterion dict."""
    raw = 0
    for c in POSITIVE:
        raw += scores.get(c, 0) * WEIGHTS[c]
    for c in set(CRITERIA) - POSITIVE:
        raw -= scores.get(c, 0) * WEIGHTS[c]

    normalized = ((raw + 24) / 48) * 99 + 1
    normalized = max(1.0, min(100.0, normalized))
    return raw, normalized


def parse_response(text: str, ticker: str) -> Optional[dict]:
    """Parse JSON from Claude response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("  %s: JSON parse failed: %s", ticker, e)
        logger.error("  Response: %s", text[:200])
        return None

    scores = data.get("scores", {})
    reasoning = data.get("reasoning", {})

    # Validate all criteria present
    for c in CRITERIA:
        if c not in scores:
            logger.warning("  %s: missing criterion %s", ticker, c)
            return None
        if scores[c] not in (-1, 0, 1):
            logger.warning("  %s: invalid score for %s: %s", ticker, c, scores[c])
            return None

    # Recompute scores (don't trust model's arithmetic)
    raw, normalized = compute_scores(scores)

    row = {"ticker": ticker, "raw_score": raw, "normalized_score": round(normalized, 1)}
    for c in CRITERIA:
        row[c] = scores[c]
    for c in CRITERIA:
        row[f"{c}_reasoning"] = reasoning.get(c, "")

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score firms using Literature Rubric via Claude API"
    )
    parser.add_argument("--test", action="store_true",
                        help="Score 5 test firms only")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Score specific tickers")
    parser.add_argument("--model", default=MODEL,
                        help=f"Claude model (default: {MODEL})")
    args = parser.parse_args()

    # Load system prompt
    with open(SYSTEM_PROMPT_PATH) as f:
        system_prompt = f.read().strip()

    # Determine which tickers to score
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.test:
        tickers = TEST_TICKERS
    else:
        # All extracted firms
        tickers = sorted(
            f.replace(".txt", "")
            for f in os.listdir(EXTRACTS_DIR)
            if f.endswith(".txt")
        )

    logger.info("Scoring %d firms with model=%s", len(tickers), args.model)

    client = anthropic.Anthropic()
    rows = []
    failures = []

    for i, ticker in enumerate(tickers):
        item1 = load_item1(ticker)
        if not item1:
            logger.warning("  %s: no Item 1 text found", ticker)
            failures.append(ticker)
            continue

        # Truncate if extremely long (>15k words → ~60k tokens)
        words = item1.split()
        if len(words) > 15000:
            item1 = " ".join(words[:15000])
            logger.info("  %s: truncated to 15k words", ticker)

        user_msg = f"Ticker: {ticker}\n\n{item1}"

        time.sleep(RATE_LIMIT)
        try:
            resp = client.messages.create(
                model=args.model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
        except Exception as e:
            logger.error("  %s: API error: %s", ticker, e)
            failures.append(ticker)
            continue

        row = parse_response(text, ticker)
        if row is None:
            failures.append(ticker)
            continue

        rows.append(row)
        logger.info(
            "  [%d/%d] %s: raw=%+d normalized=%.1f",
            i + 1, len(tickers), ticker, row["raw_score"], row["normalized_score"],
        )

    # Save
    if rows:
        df = pd.DataFrame(rows)
        cols = (
            ["ticker", "raw_score", "normalized_score"]
            + CRITERIA
            + [f"{c}_reasoning" for c in CRITERIA]
        )
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df[cols].to_csv(OUTPUT_CSV, index=False)
        logger.info("Wrote %d scores → %s", len(df), OUTPUT_CSV)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("SCORING SUMMARY")
    logger.info("=" * 50)
    logger.info("Scored: %d / %d", len(rows), len(tickers))
    if failures:
        logger.info("Failed: %d — %s", len(failures), ", ".join(failures))

    if rows:
        scores = [r["normalized_score"] for r in rows]
        logger.info("Normalized score distribution:")
        logger.info("  Mean: %.1f", sum(scores) / len(scores))
        logger.info("  Min:  %.1f (%s)", min(scores),
                     next(r["ticker"] for r in rows if r["normalized_score"] == min(scores)))
        logger.info("  Max:  %.1f (%s)", max(scores),
                     next(r["ticker"] for r in rows if r["normalized_score"] == max(scores)))


if __name__ == "__main__":
    main()
