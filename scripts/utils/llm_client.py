"""
LLM client for Phase 4 scoring (supply ρ_i and demand δ_i).

Wraps claude-haiku-4-5-20251001 with:
  - Tool-use forced structured output (Pydantic schema → tool input_schema)
  - Prompt caching on system prompts (cache_control=ephemeral)
  - Cache verification assertions:
      * first call for a given prompt: cache_creation_input_tokens > 0
      * subsequent calls: cache_read_ratio > 0.85
  - Pydantic validation with retry-on-failure (model gets 3 attempts)
  - Per-call cost tracking with Haiku 4.5 pricing (verified Apr 2026)

Caller contract: score_firm() returns (validated_result, usage_record).
The caller is responsible for writing usage_record to its own CSV/log.
This client never writes to disk.

API key is loaded from .env at module import time via python-dotenv.
The .env file lives at the repo root and is gitignored. The repo's
existing scripts (e.g. 01_build_firm_universe.py) use the same pattern.

Gemini 2.5 Flash fallback is documented as a skeleton at the bottom of
the file (NotImplementedError). It will be implemented only if Haiku
anchor validation fails — never for cost reasons.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Type

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from utils.logging_setup import get_logger

logger = get_logger("llm_client", "llm_client")

# Load .env from repo root. python-dotenv searches upward from cwd by default,
# but for safety we explicitly point at the repo root (two levels up from
# scripts/utils/llm_client.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")


# =============================================================================
# Constants
# =============================================================================
MODEL_ID = "claude-haiku-4-5-20251001"

# Haiku 4.5 pricing — verified against Anthropic platform docs (Apr 2026).
# Do NOT replace with Haiku 3.5 values ($0.80 / $4.00).
PRICE_INPUT_PER_MTOK          = 1.00
PRICE_OUTPUT_PER_MTOK         = 5.00
PRICE_CACHE_WRITE_5M_PER_MTOK = 1.25
PRICE_CACHE_READ_PER_MTOK     = 0.10

_RETRY_DELAYS_API = (1, 4, 16)   # API errors: 429, 5xx
_DEFAULT_MAX_TOKENS = 2000
_DEFAULT_TEMPERATURE = 0.0
_CACHE_READ_RATIO_FLOOR = 0.85   # subsequent-call cache assertion threshold


# =============================================================================
# Exceptions
# =============================================================================
class LLMScoringError(RuntimeError):
    """Raised when validation retries are exhausted."""
    pass


class CacheVerificationError(RuntimeError):
    """Raised when prompt caching is silently failing."""
    pass


# =============================================================================
# API key validation (at import time)
# =============================================================================
def _validate_api_key(key: str | None, var_name: str) -> str:
    if not key or key.strip() == "":
        raise RuntimeError(
            f"{var_name} is not set. "
            f"Add it to {_REPO_ROOT}/.env: {var_name}=sk-ant-..."
        )
    if key.startswith("REPLACE-ME"):
        raise RuntimeError(
            f"{var_name} contains a placeholder value '{key[:20]}...'. "
            f"Set a real API key in {_REPO_ROOT}/.env"
        )
    if key.endswith("..."):
        raise RuntimeError(
            f"{var_name} appears truncated (ends with '...'). "
            f"Copy the full key from console.anthropic.com into {_REPO_ROOT}/.env"
        )
    return key


_API_KEY = _validate_api_key(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY")
_client = anthropic.Anthropic(api_key=_API_KEY)


# =============================================================================
# Module-level state
# =============================================================================
# Tracks whether we've seen a first call for each unique system prompt hash.
# Single-process scoring is the only intended use; this is not thread-safe.
_cache_state: dict[str, bool] = {}

# Cumulative cost across all score_firm() calls in this process.
_cumulative_cost: float = 0.0
_n_calls: int = 0


# =============================================================================
# Public: score_firm()
# =============================================================================
def score_firm(
    *,
    system_prompt: str,
    firm_text: str,
    ticker: str,
    schema: Type[BaseModel],
    tool_name: str,
    tool_description: str,
    max_retries: int = 3,
    verify_cache: bool = True,
) -> tuple[BaseModel, dict]:
    """
    Score one firm with forced structured output + prompt caching + validation.

    Args:
        system_prompt: Full text of prompts/{supply,demand}_*.txt.
        firm_text:     Pre-shock 10-K Item 1 text (full, no truncation).
        ticker:        Firm ticker (for logging and inclusion in prompt).
        schema:        Pydantic BaseModel subclass (SupplyScore or DemandScore).
        tool_name:     "submit_supply_score" or "submit_demand_score".
        tool_description: Human-readable description of the tool's purpose.
        max_retries:   Retries for ValidationError (per-call). Default 3.
        verify_cache:  If True, raise CacheVerificationError on cache misses.
                       Set False for full-run loops longer than 5 min where
                       TTL expiry is expected.

    Returns:
        (validated_result, usage_record) where:
          validated_result : instance of `schema`, fully validated
          usage_record     : dict with keys
            ticker, input_tokens, cache_creation_input_tokens,
            cache_read_input_tokens, output_tokens, cost_usd,
            cache_hit_ratio, retries_used, tool_use_id, attempt_count

    Raises:
        LLMScoringError       if validation fails after max_retries
        CacheVerificationError if cache assertions fail (verify_cache=True)
        anthropic.APIError    on persistent non-retryable API errors
    """
    global _cumulative_cost, _n_calls

    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    is_first_call_for_prompt = prompt_hash not in _cache_state

    user_message = (
        f"Score the following firm based on its pre-shock 10-K Item 1 text.\n\n"
        f"Ticker: {ticker}\n\n"
        f"=== 10-K ITEM 1 BUSINESS DESCRIPTION ===\n\n"
        f"{firm_text}"
    )

    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": schema.model_json_schema(),
    }

    last_validation_error: ValidationError | None = None

    for attempt in range(1, max_retries + 1):
        if attempt > 1 and last_validation_error is not None:
            retry_message = (
                user_message
                + f"\n\n=== PREVIOUS ATTEMPT FAILED VALIDATION ===\n\n"
                + f"Error: {last_validation_error}\n\n"
                + "Please re-score the firm and submit a valid tool call."
            )
        else:
            retry_message = user_message

        response = _api_call_with_retry(
            system_prompt=system_prompt,
            user_message=retry_message,
            tool_def=tool_def,
            tool_name=tool_name,
        )

        u = response.usage
        input_tokens = getattr(u, "input_tokens", 0)
        cache_creation = getattr(u, "cache_creation_input_tokens", 0)
        cache_read = getattr(u, "cache_read_input_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0)

        cost = (
            input_tokens    * PRICE_INPUT_PER_MTOK          / 1_000_000
            + cache_creation * PRICE_CACHE_WRITE_5M_PER_MTOK / 1_000_000
            + cache_read    * PRICE_CACHE_READ_PER_MTOK     / 1_000_000
            + output_tokens  * PRICE_OUTPUT_PER_MTOK         / 1_000_000
        )
        _cumulative_cost += cost
        _n_calls += 1

        if verify_cache:
            _verify_cache(
                prompt_hash=prompt_hash,
                is_first_call=is_first_call_for_prompt,
                input_tokens=input_tokens,
                cache_creation=cache_creation,
                cache_read=cache_read,
                ticker=ticker,
                system_prompt_chars=len(system_prompt),
            )
        _cache_state[prompt_hash] = True
        is_first_call_for_prompt = False

        total_input_side = input_tokens + cache_creation + cache_read
        cache_hit_ratio = (
            cache_read / total_input_side if total_input_side > 0 else 0.0
        )

        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                tool_use_block = block
                break

        if tool_use_block is None:
            logger.warning(
                f"{ticker}: forced tool_use but no tool_use block in response; "
                f"attempt {attempt}/{max_retries}"
            )
            last_validation_error = ValueError(
                "Response contained no tool_use block despite forced tool_choice"
            )
            continue

        try:
            validated = schema.model_validate(tool_use_block.input)
        except ValidationError as exc:
            last_validation_error = exc
            logger.warning(
                f"{ticker}: validation failed on attempt {attempt}/{max_retries}: "
                f"{str(exc)[:200]}"
            )
            continue

        usage_record = {
            "ticker": ticker,
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "cache_hit_ratio": round(cache_hit_ratio, 4),
            "retries_used": attempt - 1,
            "tool_use_id": tool_use_block.id,
            "attempt_count": attempt,
        }

        logger.info(
            f"{ticker}  cw={cache_creation} cr={cache_read} in={input_tokens} "
            f"out={output_tokens}  cost=${cost:.5f}  cumulative=${_cumulative_cost:.4f}  "
            f"calls={_n_calls}  retries={attempt - 1}"
        )

        return validated, usage_record

    raise LLMScoringError(
        f"{ticker}: validation failed after {max_retries} attempts. "
        f"Last error: {last_validation_error}"
    )


# =============================================================================
# Helper: _api_call_with_retry
# =============================================================================
def _api_call_with_retry(
    *,
    system_prompt: str,
    user_message: str,
    tool_def: dict,
    tool_name: str,
):
    """Single API call with exponential backoff for 429 / 5xx / connection errors.

    Returns the anthropic Message object on success. Raises on persistent failure
    or non-retryable error (4xx other than 429).
    """
    last_exc: Exception | None = None

    for attempt, delay in enumerate((*_RETRY_DELAYS_API, None), start=1):
        try:
            return _client.messages.create(
                model=MODEL_ID,
                max_tokens=_DEFAULT_MAX_TOKENS,
                temperature=_DEFAULT_TEMPERATURE,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": tool_name},
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.RateLimitError as exc:
            last_exc = exc
            if delay is None:
                break
            logger.warning(f"Rate limit hit; sleeping {delay}s before retry {attempt}")
            time.sleep(delay)
            continue
        except anthropic.APIStatusError as exc:
            if exc.status_code in (500, 502, 503, 504):
                last_exc = exc
                if delay is None:
                    break
                logger.warning(
                    f"5xx error {exc.status_code}; sleeping {delay}s before retry {attempt}"
                )
                time.sleep(delay)
                continue
            raise
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            if delay is None:
                break
            logger.warning(f"Connection error; sleeping {delay}s before retry {attempt}")
            time.sleep(delay)
            continue

    raise RuntimeError(
        f"Anthropic API failed after {len(_RETRY_DELAYS_API) + 1} attempts. "
        f"Last error: {last_exc}"
    )


# =============================================================================
# Helper: _verify_cache
# =============================================================================
def _verify_cache(
    *,
    prompt_hash: str,
    is_first_call: bool,
    input_tokens: int,
    cache_creation: int,
    cache_read: int,
    ticker: str,
    system_prompt_chars: int,
) -> None:
    """Assert that prompt caching is working as expected.

    First call for a prompt:
        cache_creation_input_tokens MUST be > 0.
        If 0, the system prompt is below Haiku 4.5's 4,096-token cache minimum
        and caching silently failed.

    Subsequent calls:
        cache_read_input_tokens / total_input_side MUST be > 0.85.
        If lower, the cache was silently invalidated (TTL expiry, prompt drift,
        or workspace-level isolation issue from the Feb 2026 Anthropic change).

    Mid-run cache write detection:
        Subsequent calls with cache_creation > 0 are logged as warnings.
        This indicates cache invalidation between calls — costly but not fatal.
    """
    if is_first_call:
        if cache_creation == 0:
            raise CacheVerificationError(
                f"Cache write expected on first call for prompt_hash {prompt_hash} "
                f"(ticker={ticker}) but cache_creation_input_tokens=0. "
                f"Likely cause: system prompt below Haiku 4.5 minimum (4,096 tokens). "
                f"Measured: input_tokens={input_tokens}, system_prompt chars={system_prompt_chars}. "
                f"Fix: pad the system prompt in prompts/*.txt — Sonnet's lower 1,024 minimum "
                f"is not an option per repo rules (Haiku 4.5 only)."
            )
        logger.info(
            f"{ticker}  CACHE WRITE OK  prompt_hash={prompt_hash}  "
            f"cache_creation={cache_creation} tokens"
        )
    else:
        total_input_side = input_tokens + cache_creation + cache_read
        if total_input_side == 0:
            raise CacheVerificationError(
                f"{ticker}: zero total input-side tokens reported by API"
            )
        cache_read_ratio = cache_read / total_input_side

        if cache_read_ratio < _CACHE_READ_RATIO_FLOOR:
            raise CacheVerificationError(
                f"Cache miss on subsequent call for prompt_hash {prompt_hash} "
                f"(ticker={ticker}). cache_read_ratio={cache_read_ratio:.3f} "
                f"< floor={_CACHE_READ_RATIO_FLOOR}. "
                f"Tokens: cache_read={cache_read} cache_write={cache_creation} "
                f"input={input_tokens}. "
                f"Likely cause: 5-min TTL expired between calls (>300s gap), "
                f"or system prompt content changed between calls. "
                f"For full runs >5 min, pass verify_cache=False to score_firm()."
            )

        if cache_creation > 0:
            logger.warning(
                f"{ticker}: subsequent call wrote {cache_creation} cache tokens "
                f"(prompt_hash {prompt_hash}). Cache may have been invalidated "
                f"mid-run. Cost impact: 12.5x vs cache read."
            )


# =============================================================================
# Helpers: aggregate_costs and module utilities
# =============================================================================
def aggregate_costs(usage_records: list[dict]) -> dict:
    """Sum per-firm cost records into a totals dict."""
    if not usage_records:
        return {
            "n_calls": 0,
            "total_input_tokens": 0,
            "total_cache_write_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "mean_cost_per_call": 0.0,
            "cache_hit_ratio_overall": 0.0,
        }

    total_input = sum(r["input_tokens"] for r in usage_records)
    total_write = sum(r["cache_creation_input_tokens"] for r in usage_records)
    total_read = sum(r["cache_read_input_tokens"] for r in usage_records)
    total_output = sum(r["output_tokens"] for r in usage_records)
    total_cost = sum(r["cost_usd"] for r in usage_records)
    n = len(usage_records)
    total_input_side = total_input + total_write + total_read

    return {
        "n_calls": n,
        "total_input_tokens": total_input,
        "total_cache_write_tokens": total_write,
        "total_cache_read_tokens": total_read,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 5),
        "mean_cost_per_call": round(total_cost / n, 6),
        "cache_hit_ratio_overall": (
            round(total_read / total_input_side, 4) if total_input_side > 0 else 0.0
        ),
    }


def get_cumulative_cost() -> float:
    """Return module-level cumulative cost across all score_firm() calls."""
    return _cumulative_cost


def reset_cumulative_state() -> None:
    """Reset module-level cumulative cost, call counter, and cache state."""
    global _cumulative_cost, _n_calls, _cache_state
    _cumulative_cost = 0.0
    _n_calls = 0
    _cache_state = {}


# =============================================================================
# Gemini 2.5 Flash fallback — NOT YET IMPLEMENTED
# =============================================================================
# Trigger: only if Haiku anchor validation fails systematically.
# Per CLAUDE.md, Gemini is a capability fallback, NOT a cost-driven choice.

def score_firm_gemini(*args, **kwargs):
    """Gemini fallback — not implemented in Phase 4 initial pass."""
    raise NotImplementedError(
        "Gemini fallback is a Phase 4 capability backup, triggered only on "
        "Haiku anchor validation failure. Implement in scripts/utils/llm_client.py "
        "after consulting the anchor test results. Per CLAUDE.md, never use "
        "Gemini for cost reasons — it's a quality fallback only."
    )
