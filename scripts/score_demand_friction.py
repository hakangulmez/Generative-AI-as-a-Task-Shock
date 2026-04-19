"""
Score firms on demand-side friction (δ_i) — the degree to which customers are
insulated from switching to LLM-based alternatives.

Three sub-scores (each ∈ [0,1] in 0.1 steps):
  δ_switch  — switching cost (contractual + technical + organizational + data portability)
  δ_error   — error cost (liability, regulatory, irreversibility of AI mistakes)
  δ_data    — data/network moat (proprietary data or network effects)

  δ_composite = (δ_switch + δ_error + δ_data) / 3  ∈ [0, 1]

Theoretical grounding:
  Farrell & Klemperer (2007) — switching cost theory
  Agrawal, Gans & Goldfarb (2018) — prediction machines / error cost
  Katz & Shapiro (1985), Rochet & Tirole (2003) — network effects / data moats

Usage:
  python3 scripts/score_demand_friction.py --test
  python3 scripts/score_demand_friction.py --tickers MSFT CRM
  python3 scripts/score_demand_friction.py --skip-existing
  python3 scripts/score_demand_friction.py
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
SYSTEM_PROMPT_PATH = "prompts/demand_friction_system.txt"
OUTPUT_CSV         = "data/processed/demand_friction.csv"
LOG_FILE           = "logs/demand_friction_scoring.log"

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 1024
RATE_LIMIT = 3.0

ITEM1A_SEPARATOR = "### ITEM_1A_START ###"
TEST_TICKERS     = ["ZS", "DDOG", "VEEV", "PAYC", "HUBS", "LPSN", "ZIP"]

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
        # Include Item 1A (Risk Factors) — relevant for error cost scoring
        text = text  # keep full text including 1A
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

    try:
        d_switch = round(float(data["delta_switch"]), 1)
        d_error  = round(float(data["delta_error"]), 1)
        d_data   = round(float(data["delta_data"]), 1)
    except (KeyError, TypeError, ValueError) as e:
        logger.error("  %s: missing/invalid sub-score: %s", ticker, e)
        return None

    # Validate range
    for name, val in [("delta_switch", d_switch), ("delta_error", d_error), ("delta_data", d_data)]:
        if not (0.0 <= val <= 1.0):
            logger.error("  %s: %s=%.1f out of [0,1]", ticker, name, val)
            return None

    computed_composite = round((d_switch + d_error + d_data) / 3, 3)

    model_composite = data.get("delta_composite")
    if model_composite is not None:
        try:
            diff = abs(float(model_composite) - computed_composite)
            if diff > 0.01:
                logger.warning("  %s: model composite %.3f vs computed %.3f",
                               ticker, float(model_composite), computed_composite)
        except (TypeError, ValueError):
            pass

    return {
        "ticker": ticker,
        "delta_switch": d_switch,
        "delta_error": d_error,
        "delta_data": d_data,
        "delta_composite": computed_composite,
        "switching_cost_reasoning": data.get("switching_cost_reasoning", ""),
        "error_cost_reasoning": data.get("error_cost_reasoning", ""),
        "data_moat_reasoning": data.get("data_moat_reasoning", ""),
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
    cols = ["ticker", "delta_switch", "delta_error", "delta_data", "delta_composite",
            "switching_cost_reasoning", "error_cost_reasoning", "data_moat_reasoning"]

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

        logger.info(
            "  [%d/%d] %s: switch=%.1f error=%.1f data=%.1f composite=%.3f",
            i + 1, len(tickers), ticker,
            row["delta_switch"], row["delta_error"], row["delta_data"],
            row["delta_composite"],
        )

    logger.info("")
    logger.info("=" * 50)
    logger.info("Scored: %d / %d", len(tickers) - len(failures), len(tickers))
    if failures:
        logger.info("Failed: %d — %s", len(failures), ", ".join(failures))

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        c = pd.to_numeric(df["delta_composite"], errors="coerce").dropna()
        logger.info("δ_composite distribution:")
        logger.info("  n=%d  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                    len(c), c.mean(), c.std(), c.min(), c.max())


if __name__ == "__main__":
    main()
