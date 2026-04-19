"""
Score firms using the C1-C10 Literature Rubric (PRIMARY treatment variable).

Three-framework grounding:
  Eloundou et al. (2024) — E1/E0 task exposure
  Acemoglu & Restrepo (2022) — task substitution
  Brynjolfsson (2023) — LLM suitability

Score computation (performed by Claude Opus, not pipeline):
  C1-C5 weighted sum (E1-increasing) minus C6-C10 weighted sum (E0-decreasing)
  raw ∈ [-24, +24] → normalized = round((raw + 24) / 48 * 99 + 1) ∈ [1, 100]

Usage:
  python3 scripts/score_literature_rubric.py --test
  python3 scripts/score_literature_rubric.py --tickers MSFT CRM
  python3 scripts/score_literature_rubric.py --skip-existing
  python3 scripts/score_literature_rubric.py
"""

import os
import sys
import json
import time
import argparse
import logging

import anthropic
import pandas as pd

EXTRACTS_DIR       = "text_data/10k_extracts"
SYSTEM_PROMPT_PATH = "prompts/literature_rubric_system.txt"
OUTPUT_CSV         = "data/processed/lit_scores.csv"
LOG_FILE           = "logs/literature_rubric_scoring.log"

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 1024
RATE_LIMIT = 3.0

ITEM1A_SEPARATOR = "### ITEM_1A_START ###"
TEST_TICKERS     = ["ZS", "NET", "ASAN", "ZIP", "LPSN"]

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def load_item1(ticker: str):
    path = os.path.join(EXTRACTS_DIR, f"{ticker}.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    if ITEM1A_SEPARATOR in text:
        text = text.split(ITEM1A_SEPARATOR)[0]
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


def parse_response(text: str, ticker: str):
    import re
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
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
        logger.error("  Response: %s", text[:300])
        return None

    # B2C / Mixed — excluded
    customer_type = data.get("customer_type", "B2B")
    if customer_type != "B2B":
        logger.info("  %s: customer_type=%s — excluded", ticker, customer_type)
        return {"ticker": ticker, "normalized_score": None, "customer_type": customer_type,
                "e0_count": 0, "e1_count": 0, "e2_count": 0, "n_tasks": 0,
                "integration_depth_penalty": 0, "raw_exposure": None,
                "adjusted_exposure": None, "reasoning": data.get("reasoning", "")}

    # Task counts
    e0 = int(data.get("e0_count", 0))
    e1 = int(data.get("e1_count", 0))
    e2 = int(data.get("e2_count", 0))
    n  = int(data.get("n_tasks", e0 + e1 + e2))
    penalty = int(data.get("integration_depth_penalty", 0))

    if n == 0:
        logger.warning("  %s: n_tasks=0", ticker)
        return None

    # Recompute scores from task counts (pipeline validation)
    raw_exp  = (e1 + 0.5 * e2) / n
    adj_exp  = max(0.0, raw_exp + penalty / 10)
    computed = round(adj_exp * 99 + 1, 1)
    computed = max(1.0, min(100.0, computed))

    # Model's own normalized_score (for comparison)
    model_score = data.get("normalized_score")
    if model_score is not None:
        try:
            model_score = round(float(model_score), 1)
            if abs(model_score - computed) > 3:
                logger.warning("  %s: model score %.1f vs computed %.1f (diff=%.1f)",
                               ticker, model_score, computed, abs(model_score - computed))
        except (TypeError, ValueError):
            pass

    reasoning = data.get("reasoning", "")

    return {
        "ticker": ticker,
        "normalized_score": computed,
        "customer_type": customer_type,
        "e0_count": e0, "e1_count": e1, "e2_count": e2,
        "n_tasks": n,
        "integration_depth_penalty": penalty,
        "raw_exposure": round(raw_exp, 4),
        "adjusted_exposure": round(adj_exp, 4),
        "reasoning": reasoning,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    with open(SYSTEM_PROMPT_PATH) as f:
        system_prompt = f.read().strip()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.test:
        tickers = TEST_TICKERS
    else:
        tickers = sorted(
            f.replace(".txt", "")
            for f in os.listdir(EXTRACTS_DIR)
            if f.endswith(".txt")
        )

    existing = set()
    if args.skip_existing and os.path.exists(OUTPUT_CSV):
        existing = set(pd.read_csv(OUTPUT_CSV)["ticker"].tolist())
        tickers = [t for t in tickers if t not in existing]
        logger.info("Skipping %d already scored firms", len(existing))

    logger.info("Scoring %d firms with model=%s", len(tickers), args.model)

    client = anthropic.Anthropic()
    failures = []

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV) or not existing
    cols = ["ticker", "normalized_score", "customer_type",
            "e0_count", "e1_count", "e2_count", "n_tasks",
            "integration_depth_penalty", "raw_exposure", "adjusted_exposure",
            "reasoning"]

    for i, ticker in enumerate(tickers):
        item1 = load_item1(ticker)
        if not item1:
            logger.warning("  %s: no Item 1 text found", ticker)
            failures.append(ticker)
            continue

        words = item1.split()
        if len(words) > 15047:
            item1 = " ".join(words[:15047])
            logger.info("  %s: truncated to 15,047 words", ticker)

        user_msg = f"Ticker: {ticker}\n\n{item1}"

        time.sleep(RATE_LIMIT)
        try:
            resp = client.messages.create(
                model=args.model,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
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

        df_row = pd.DataFrame([row])[cols]
        df_row.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        write_header = False

        score_str = f"{row['normalized_score']:.1f}" if row["normalized_score"] is not None else "B2C"
        logger.info("  [%d/%d] %s: score=%s  E0=%d E1=%d E2=%d  penalty=%d  %s",
                    i + 1, len(tickers), ticker, score_str,
                    row["e0_count"], row["e1_count"], row["e2_count"],
                    row["integration_depth_penalty"], row["reasoning"][:80])

    logger.info("")
    logger.info("=" * 50)
    logger.info("Scored: %d / %d", len(tickers) - len(failures), len(tickers))
    if failures:
        logger.info("Failed: %d — %s", len(failures), ", ".join(failures))

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        s = pd.to_numeric(df["normalized_score"], errors="coerce").dropna()
        logger.info("Score distribution (B2B only):")
        logger.info("  n=%d  mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
                    len(s), s.mean(), s.std(), s.min(), s.max())


if __name__ == "__main__":
    main()
