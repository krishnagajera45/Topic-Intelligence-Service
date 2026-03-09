"""Backfill NMF metrics for the existing LDA/BERTopic batch history.

This script is needed when NMF was added to the pipeline AFTER the existing
batches were already processed.  It reconstructs per-batch NMF results by:

  1. Reading the existing LDA batch metadata (batch IDs, timestamps, window
     dates, and topic counts).
  2. Splitting the cumulative stored corpus into the same number of
     proportional cumulative slices (i.e. batch k gets the first k/N fraction
     of all documents, exactly as the live pipeline does).
  3. Running the NMF comparison flow for each slice and saving the results to
     ``outputs/metrics/nmf_metrics.json``.

Run from the project root:
    python scripts/backfill_nmf_metrics.py
"""
import sys
import json
from pathlib import Path

# ── ensure project root is on PYTHONPATH ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.etl.tasks.nmf_tasks import (
    preprocess_documents_for_nmf_task,
    train_nmf_model_task,
    calculate_nmf_coherence_task,
    calculate_nmf_diversity_task,
    calculate_nmf_silhouette_task,
    extract_nmf_metadata_task,
    save_nmf_metrics_task,
)
from src.utils import load_config

import time
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def run_nmf_for_batch(documents, num_topics, batch_id, window_start, window_end, config):
    """Run NMF tasks directly (no Prefect server required)."""
    print(f"\n{'─'*60}")
    print(f"  Batch   : {batch_id}")
    print(f"  Window  : {window_start}  →  {window_end}")
    print(f"  Docs    : {len(documents)}   Topics target: {num_topics}")

    flow_start = time.time()

    # 1. preprocess
    t0 = time.time()
    tfidf_matrix, vectorizer, tokenized_docs, dictionary = (
        preprocess_documents_for_nmf_task(documents)
    )
    preprocess_time = time.time() - t0
    print(f"  Preprocessed : {tfidf_matrix.shape[0]} × {tfidf_matrix.shape[1]} in {preprocess_time:.1f}s")

    if tfidf_matrix.shape[0] < 10:
        print("  ⚠️  Too few docs — skipping")
        return None

    # Clamp topics
    max_topics = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1])
    if num_topics > max_topics:
        print(f"  ⚠️  Clamping topics {num_topics} → {max_topics}")
        num_topics = max(2, max_topics)

    # 2. train
    max_iter = getattr(getattr(config, 'nmf', None), 'max_iter', 400)
    top_n    = getattr(getattr(config, 'model', None), 'top_n_words', 10)

    t0 = time.time()
    nmf_model = train_nmf_model_task(tfidf_matrix, num_topics,
                                     max_iter=max_iter,
                                     alpha_W=0.0, alpha_H=0.0)
    train_time = time.time() - t0
    recon_err = float(nmf_model.reconstruction_err_)
    print(f"  Trained NMF  : {num_topics} topics, err={recon_err:.4f} in {train_time:.1f}s")

    # 3. evaluate
    feature_names = vectorizer.get_feature_names_out()

    t0 = time.time()
    coherence = calculate_nmf_coherence_task(
        nmf_model, feature_names, tokenized_docs, dictionary, top_n=top_n)
    diversity  = calculate_nmf_diversity_task(nmf_model, feature_names, top_n=top_n)
    silhouette = calculate_nmf_silhouette_task(tfidf_matrix, nmf_model)
    eval_time  = time.time() - t0
    print(f"  Metrics      : coherence={coherence:.4f}  diversity={diversity:.4f}  "
          f"silhouette={silhouette:.4f}  ({eval_time:.1f}s)")

    # 4. metadata
    topic_metadata = extract_nmf_metadata_task(nmf_model, tfidf_matrix,
                                                feature_names, top_n=top_n)

    total_time = time.time() - flow_start

    # 5. build metrics dict
    metrics = {
        "status":                    "success",
        "batch_id":                  batch_id,
        "window_start":              window_start,
        "window_end":                window_end,
        "timestamp":                 datetime.utcnow().isoformat(),
        "num_topics":                num_topics,
        "num_documents":             len(documents),
        "vocabulary_size":           int(tfidf_matrix.shape[1]),
        "reconstruction_error":      recon_err,
        "coherence_c_v":             float(coherence),
        "diversity":                 float(diversity),
        "silhouette_score":          float(silhouette),
        "preprocessing_time_seconds": round(preprocess_time, 3),
        "training_time_seconds":      round(train_time, 3),
        "evaluation_time_seconds":    round(eval_time, 3),
        "total_time_seconds":         round(total_time, 3),
        "topics":                    topic_metadata,
        "nmf_config": {
            "max_iter":    max_iter,
            "alpha_W":     0.0,
            "alpha_H":     0.0,
            "top_n_words": top_n,
        },
    }

    # 6. save
    save_nmf_metrics_task(metrics)
    print(f"  Saved ✓   (total {total_time:.1f}s)")
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NMF Metrics Backfill")
    print("=" * 60)

    # Load config
    config = load_config()

    # ── 1. load batch metadata from LDA metrics ───────────────────────────────
    lda_path = ROOT / "outputs" / "metrics" / "lda_metrics.json"
    if not lda_path.exists():
        print(f"ERROR: LDA metrics not found at {lda_path}")
        sys.exit(1)

    with open(lda_path) as f:
        lda_data = json.load(f)

    lda_batches = lda_data.get("batches", [])
    if not lda_batches:
        print("ERROR: No LDA batches found — run the pipeline first.")
        sys.exit(1)

    print(f"Found {len(lda_batches)} LDA batches to backfill.")

    # ── 2. load full cumulative corpus ────────────────────────────────────────
    corpus_path = ROOT / "models" / "current" / "bertopic_model_corpus.json"
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        sys.exit(1)

    with open(corpus_path) as f:
        full_corpus = json.load(f)

    print(f"Corpus loaded: {len(full_corpus)} documents total.")

    # ── 3. check if nmf_metrics.json already has data ─────────────────────────
    nmf_path = ROOT / "outputs" / "metrics" / "nmf_metrics.json"
    existing_ids = set()
    if nmf_path.exists():
        try:
            with open(nmf_path) as f:
                existing = json.load(f)
            existing_ids = {b["batch_id"] for b in existing.get("batches", [])}
            print(f"Existing NMF batches: {len(existing_ids)} — will skip duplicates.")
        except Exception:
            pass

    # ── 4. replay each batch ──────────────────────────────────────────────────
    n = len(lda_batches)
    results = []

    for i, batch in enumerate(lda_batches, start=1):
        batch_id     = batch["batch_id"]
        window_start = batch.get("window_start", "")
        window_end   = batch.get("window_end", "")
        num_topics   = batch.get("num_topics") or 20

        if batch_id in existing_ids:
            print(f"\n  [{i}/{n}] Skipping {batch_id} (already exists)")
            continue

        # Cumulative corpus slice: first i/n fraction of the full corpus
        cutoff = max(50, int(len(full_corpus) * i / n))
        docs_slice = full_corpus[:cutoff]

        # Re-cap num_topics: don't request more than rough vocab estimate // 3
        # (exact vocab is unknown here, but short-text TF-IDF max_features=5000
        # so the real vocab for a small slice can be much smaller)
        est_vocab = min(5000, max(20, len(docs_slice) * 3))
        safe_max_topics = max(2, est_vocab // 3)
        if num_topics > safe_max_topics:
            num_topics = safe_max_topics

        result = run_nmf_for_batch(
            documents=docs_slice,
            num_topics=num_topics,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end,
            config=config,
        )
        if result:
            results.append(result)

    print("\n" + "=" * 60)
    print(f"  Backfill complete: {len(results)} NMF batches written.")
    print(f"  Output: {nmf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
