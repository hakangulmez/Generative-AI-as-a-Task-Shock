"""
Build the Eloundou ONET task corpus with embeddings.

One-time setup script for Phase 4 pivot. Reads full_labelset.tsv from
data/external/, embeds the Task column with sentence-transformers/
all-MiniLM-L6-v2, and writes a parquet file with task_id, onet_soc_code,
task_text, exposure labels, and the 384-dim embedding vector.

Output: data/external/eloundou_task_embeddings.parquet

Run once. Re-run only if the source TSV changes or the embedding model
is upgraded. Output is gitignored under data/external/.

Source: github.com/openai/GPTs-are-GPTs/blob/main/data/full_labelset.tsv
README: 'alpha=E1, beta=E1+.5*E2, gamma=E1+E2'
        gpt4_exposure column = string label E0/E1/E2 (the GPT-4 annotator)
        beta column = numeric Eloundou-beta (0.0 / 0.5 / 1.0)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/external/eloundou_full_labelset.tsv")
OUTPUT_PATH = Path("data/external/eloundou_task_embeddings.parquet")
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run the curl download first.")
        sys.exit(1)

    print(f"Loading TSV from {INPUT_PATH}...")
    # The first column in the TSV is unnamed (row index from pandas).
    # Read with index_col=0 to drop it.
    df = pd.read_csv(INPUT_PATH, sep="\t", index_col=0, low_memory=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Check critical columns exist
    required = ["O*NET-SOC Code", "Task ID", "Task", "gpt4_exposure", "alpha", "beta", "gamma"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    # Drop rows with missing Task or labels (defensive)
    n_before = len(df)
    df = df.dropna(subset=["Task", "gpt4_exposure", "beta"])
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with missing Task/label")

    # Sanity check label distribution
    print(f"\nLabel distribution (gpt4_exposure):")
    print(df["gpt4_exposure"].value_counts().to_string())

    print(f"\nBeta value distribution:")
    print(df["beta"].value_counts().sort_index().to_string())

    # Verify beta encoding matches README: E0=0.0, E1=1.0, E2=0.5
    print(f"\nLabel ↔ beta consistency check:")
    consistency = df.groupby("gpt4_exposure")["beta"].agg(["min", "max", "mean", "count"])
    print(consistency.to_string())

    # Load model
    print(f"\nLoading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded. Dim: {model.get_sentence_embedding_dimension()}")
    assert model.get_sentence_embedding_dimension() == EMBEDDING_DIM, "dim mismatch"

    # Embed all tasks
    tasks = df["Task"].tolist()
    print(f"\nEmbedding {len(tasks)} task descriptions...")
    t0 = time.time()
    embeddings = model.encode(
        tasks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(tasks)/elapsed:.0f} tasks/sec)")
    print(f"  Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    assert embeddings.shape == (len(df), EMBEDDING_DIM), "shape mismatch"

    # Build output dataframe
    out = pd.DataFrame({
        "task_id": df["Task ID"].astype(str).values,
        "onet_soc_code": df["O*NET-SOC Code"].astype(str).values,
        "task_text": df["Task"].astype(str).values,
        "task_type": df["Task Type"].astype(str).values if "Task Type" in df.columns else "",
        "title": df["Title"].astype(str).values if "Title" in df.columns else "",
        "gpt4_exposure": df["gpt4_exposure"].astype(str).values,
        "alpha": df["alpha"].astype(float).values,
        "beta": df["beta"].astype(float).values,
        "gamma": df["gamma"].astype(float).values,
        "embedding": list(embeddings),
    })
    print(f"\nOutput dataframe: {len(out)} rows, {len(out.columns)} columns")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    out.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved. File size: {size_mb:.1f} MB")

    # Quick verification: reload and check
    print(f"\nVerifying...")
    reloaded = pd.read_parquet(OUTPUT_PATH)
    assert len(reloaded) == len(out), "row count mismatch on reload"
    assert reloaded["embedding"].iloc[0].shape == (EMBEDDING_DIM,), "embedding shape lost on save"
    print(f"  Reload OK: {len(reloaded)} rows, embedding[0].shape = {reloaded['embedding'].iloc[0].shape}")

    print(f"\n=== DONE ===")
    print(f"Corpus ready at {OUTPUT_PATH}")
    print(f"Use scripts/utils/task_matching.py (Adım 3) to match against this corpus.")


if __name__ == "__main__":
    main()
