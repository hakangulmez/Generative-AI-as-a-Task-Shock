"""
Task matching utility for Phase 4 supply scoring pivot.

Given an LLM-extracted product task description, finds the best-matching
ONET task from the Eloundou (2024) corpus and returns the published
gpt4_exposure label with similarity score.

The corpus (data/external/eloundou_task_embeddings.parquet) is built once
by scripts/build_eloundou_corpus.py. This module loads it lazily on first
call and caches the embeddings + metadata in module-level state.

Embeddings are unit-normalized at corpus build time, so cosine similarity
reduces to a dot product.

Public API:
    match_task(task_text: str, top_k: int = 5) -> list[Match]
        Returns top-K matches sorted by similarity (descending).

    label_from_matches(matches: list[Match], strategy: str = "top1") -> Label
        Aggregates matches into a final label.
        strategy="top1": use the highest-similarity match's label.
        strategy="weighted_majority": similarity-weighted average of beta values.

    LOW_CONFIDENCE_THRESHOLD = 0.5
        Top-1 similarity below this is flagged as a low-confidence match
        (logged warning, but still returned — caller decides to exclude).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

CORPUS_PATH = Path("data/external/eloundou_task_embeddings.parquet")
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.5

# Eloundou label encoding (verified in Adım 2):
#   E0 → beta=0.0  (no LLM exposure)
#   E1 → beta=1.0  (direct LLM exposure)
#   E2 → beta=0.5  (LLM with standard tools)
LABEL_TO_BETA = {"E0": 0.0, "E1": 1.0, "E2": 0.5}


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class Match:
    """A single ONET task match for a product task query."""
    task_id: str
    onet_soc_code: str
    task_text: str
    title: str
    gpt4_exposure: Literal["E0", "E1", "E2"]
    beta: float       # 0.0, 0.5, or 1.0
    alpha: float      # E1 only (1.0 if E1, else 0.0)
    gamma: float      # E1 + E2 (1.0 if E1 or E2, else 0.0)
    similarity: float  # cosine similarity in [-1, 1] (typically [0, 1])


@dataclass(frozen=True)
class Label:
    """Aggregated label result from a list of matches."""
    gpt4_exposure: Literal["E0", "E1", "E2"]
    beta: float
    alpha: float
    gamma: float
    similarity_top1: float
    n_matches_used: int
    strategy: str
    low_confidence: bool  # True if similarity_top1 < LOW_CONFIDENCE_THRESHOLD


# =============================================================================
# Module-level lazy state
# =============================================================================

_corpus: pd.DataFrame | None = None
_corpus_embeddings: np.ndarray | None = None  # (N, 384), unit-normalized
_model: SentenceTransformer | None = None


def _load_corpus() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the Eloundou corpus and stack embeddings. Cached."""
    global _corpus, _corpus_embeddings
    if _corpus is None:
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"Corpus not found at {CORPUS_PATH}. "
                f"Run scripts/build_eloundou_corpus.py first."
            )
        logger.info(f"Loading Eloundou corpus from {CORPUS_PATH}")
        df = pd.read_parquet(CORPUS_PATH)
        embs = np.stack(df["embedding"].values).astype(np.float32)
        # Verify unit-normalization (build script should already have done this,
        # but defensive: re-normalize if any drift).
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.warning(
                f"Corpus embeddings not unit-normalized "
                f"(min={norms.min():.4f}, max={norms.max():.4f}). Renormalizing."
            )
            embs = embs / norms[:, np.newaxis]
        _corpus = df
        _corpus_embeddings = embs
        logger.info(f"Corpus loaded: {len(df)} tasks, embeddings shape {embs.shape}")
    return _corpus, _corpus_embeddings


def _load_model() -> SentenceTransformer:
    """Load the embedding model. Cached."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        if _model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model dimension mismatch: expected {EMBEDDING_DIM}, "
                f"got {_model.get_sentence_embedding_dimension()}"
            )
    return _model


# =============================================================================
# Public API
# =============================================================================

def match_task(task_text: str, top_k: int = DEFAULT_TOP_K) -> list[Match]:
    """Find the top-K Eloundou ONET tasks matching the given product task.

    Embeddings are unit-normalized at corpus build time, so cosine similarity
    is computed as a dot product.

    Args:
        task_text: Product task description (e.g., "Process internet traffic
                   in real-time across 250 billion daily requests").
        top_k: Number of matches to return (default 5).

    Returns:
        List of Match objects sorted by similarity descending.

    Raises:
        ValueError: if task_text is empty.
        FileNotFoundError: if corpus parquet not built yet.
    """
    if not task_text or not task_text.strip():
        raise ValueError("task_text cannot be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    df, corpus_embs = _load_corpus()
    model = _load_model()

    # Embed the query and normalize
    query_emb = model.encode(task_text, convert_to_numpy=True).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Cosine similarity = dot product (both unit-normalized)
    sims = corpus_embs @ query_emb  # shape (N,)

    # Top-K
    if top_k >= len(df):
        top_idx = np.argsort(sims)[::-1]
    else:
        # argpartition is faster than full argsort for large N
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    matches = []
    for idx in top_idx:
        row = df.iloc[int(idx)]
        matches.append(Match(
            task_id=str(row["task_id"]),
            onet_soc_code=str(row["onet_soc_code"]),
            task_text=str(row["task_text"]),
            title=str(row["title"]) if pd.notna(row["title"]) else "",
            gpt4_exposure=str(row["gpt4_exposure"]),
            beta=float(row["beta"]),
            alpha=float(row["alpha"]),
            gamma=float(row["gamma"]),
            similarity=float(sims[idx]),
        ))
    return matches


def label_from_matches(
    matches: list[Match],
    strategy: Literal["top1", "weighted_majority"] = "top1",
) -> Label:
    """Aggregate a list of matches into a single label.

    Strategy "top1" (default, primary):
        Returns the gpt4_exposure label of the highest-similarity match.
        Deterministic. Defensible. Simple.

    Strategy "weighted_majority" (Phase 6 robustness alternative):
        Returns a similarity-weighted average of beta values across all
        matches, then maps back to the discrete label whose beta value
        is closest. The averaged beta is also returned directly (continuous).

    Args:
        matches: List of Match objects (top-K from match_task).
        strategy: "top1" or "weighted_majority".

    Returns:
        Label object with gpt4_exposure, beta, alpha, gamma, similarity_top1,
        n_matches_used, strategy name, and low_confidence flag.

    Raises:
        ValueError: if matches is empty or strategy unknown.
    """
    if not matches:
        raise ValueError("matches list cannot be empty")
    if strategy not in ("top1", "weighted_majority"):
        raise ValueError(f"unknown strategy: {strategy!r}")

    sim_top1 = matches[0].similarity
    low_conf = sim_top1 < LOW_CONFIDENCE_THRESHOLD

    if strategy == "top1":
        m = matches[0]
        return Label(
            gpt4_exposure=m.gpt4_exposure,
            beta=m.beta,
            alpha=m.alpha,
            gamma=m.gamma,
            similarity_top1=sim_top1,
            n_matches_used=1,
            strategy="top1",
            low_confidence=low_conf,
        )

    # weighted_majority
    sims = np.array([m.similarity for m in matches], dtype=np.float64)
    # Negative similarities (semantic opposites) shouldn't pull the weighted
    # average. Clip to >= 0 before normalizing.
    sims = np.clip(sims, 0.0, None)
    if sims.sum() == 0:
        # All matches have zero or negative similarity — fall back to top1.
        m = matches[0]
        return Label(
            gpt4_exposure=m.gpt4_exposure,
            beta=m.beta,
            alpha=m.alpha,
            gamma=m.gamma,
            similarity_top1=sim_top1,
            n_matches_used=1,
            strategy="weighted_majority_fallback_top1",
            low_confidence=True,
        )
    weights = sims / sims.sum()

    betas = np.array([m.beta for m in matches], dtype=np.float64)
    alphas = np.array([m.alpha for m in matches], dtype=np.float64)
    gammas = np.array([m.gamma for m in matches], dtype=np.float64)

    weighted_beta = float(weights @ betas)
    weighted_alpha = float(weights @ alphas)
    weighted_gamma = float(weights @ gammas)

    # Map continuous beta back to discrete label by closest beta.
    # E0=0.0, E2=0.5, E1=1.0
    distances = {
        "E0": abs(weighted_beta - 0.0),
        "E2": abs(weighted_beta - 0.5),
        "E1": abs(weighted_beta - 1.0),
    }
    best_label = min(distances, key=distances.get)

    return Label(
        gpt4_exposure=best_label,
        beta=weighted_beta,
        alpha=weighted_alpha,
        gamma=weighted_gamma,
        similarity_top1=sim_top1,
        n_matches_used=len(matches),
        strategy="weighted_majority",
        low_confidence=low_conf,
    )


def match_and_label(
    task_text: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: Literal["top1", "weighted_majority"] = "top1",
) -> tuple[Label, list[Match]]:
    """Convenience: match + label in one call. Returns (label, top_k_matches).

    Useful for the scoring pipeline (08): one call per product task,
    Label feeds into the score, matches feed into the audit log.
    """
    matches = match_task(task_text, top_k=top_k)
    label = label_from_matches(matches, strategy=strategy)
    return label, matches


# =============================================================================
# Self-test (run as: python3 scripts/utils/task_matching.py)
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Hand-crafted test cases — each task has an expected exposure based
    # on Eloundou framework reasoning, and each tests a different region
    # of the embedding space.
    test_cases = [
        # (task_text, expected_exposure_class, rationale)
        (
            "Process and inspect internet traffic in real-time across billions of daily requests to enforce security policies and detect threats",
            "E0",
            "Real-time streaming, sub-second latency — hardware/network bound, not text",
        ),
        (
            "Generate quarterly financial reports summarizing revenue and expenses for executives",
            "E1",
            "Pure text generation from numeric data — direct LLM substitute",
        ),
        (
            "Identify and classify data loss prevention patterns using dictionaries and OCR",
            "E1",
            "Text classification task — direct LLM substitute",
        ),
        (
            "Discover cloud applications and generate access control policies based on user identity",
            "E2",
            "Database lookup + text generation — LLM with standard tools",
        ),
        (
            "Operate machinery on an automotive assembly line",
            "E0",
            "Physical operation — no LLM substitute possible",
        ),
        (
            "Translate customer support tickets from English to Spanish",
            "E1",
            "Translation — canonical LLM task",
        ),
        (
            "Audit financial statements for compliance with GAAP",
            "E2",
            "Document analysis + standard tools — LLM with reference data",
        ),
        (
            "Diagnose patients based on physical examination",
            "E0",
            "Physical examination — no LLM substitute",
        ),
        (
            "Write marketing copy for a new product launch",
            "E1",
            "Pure text generation — direct LLM substitute",
        ),
        (
            "Schedule appointments for a dental practice",
            "E2",
            "Calendar + text — LLM with tools",
        ),
    ]

    print("=" * 80)
    print("Task matching self-test (10 hand-crafted cases)")
    print("=" * 80)

    n_pass = 0
    n_fail = 0
    n_low_conf = 0

    for i, (task, expected, rationale) in enumerate(test_cases, 1):
        label, matches = match_and_label(task, top_k=5, strategy="top1")
        match_top1 = matches[0]

        passed = label.gpt4_exposure == expected
        if passed:
            n_pass += 1
            mark = "PASS"
        else:
            n_fail += 1
            mark = "FAIL"

        if label.low_confidence:
            n_low_conf += 1

        print(f"\n[{mark}] Test {i}: {task[:80]}")
        print(f"  Expected: {expected}  ({rationale})")
        print(f"  Got:      {label.gpt4_exposure} (beta={label.beta:.2f}, sim={label.similarity_top1:.3f})")
        if label.low_confidence:
            print(f"  WARNING: low confidence (sim < {LOW_CONFIDENCE_THRESHOLD})")
        print(f"  Top match: {match_top1.task_text[:90]}")
        print(f"             Title: {match_top1.title}")

    print("\n" + "=" * 80)
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_low_conf} low-confidence")
    print("=" * 80)

    # Test weighted_majority strategy on one case
    print("\n--- Bonus: weighted_majority strategy on test 1 ---")
    label_wm, matches_wm = match_and_label(test_cases[0][0], top_k=5, strategy="weighted_majority")
    print(f"top1 strategy:               beta={matches_wm[0].beta:.3f}  ({matches_wm[0].gpt4_exposure})")
    print(f"weighted_majority strategy:  beta={label_wm.beta:.3f}  ({label_wm.gpt4_exposure})")
    print(f"Top 5 betas + similarities:")
    for m in matches_wm:
        print(f"  sim={m.similarity:.3f}  [{m.gpt4_exposure}, beta={m.beta:.1f}]  {m.task_text[:70]}")

    # Failure handling: empty input, missing corpus path
    print("\n--- Edge cases ---")
    try:
        match_task("")
        print("FAIL: empty task should have raised ValueError")
        n_fail += 1
    except ValueError:
        print("PASS: empty task raises ValueError")

    try:
        match_task("x", top_k=0)
        print("FAIL: top_k=0 should have raised ValueError")
        n_fail += 1
    except ValueError:
        print("PASS: top_k=0 raises ValueError")

    if n_fail > 0:
        print(f"\n{n_fail} TEST(S) FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")
        sys.exit(0)
