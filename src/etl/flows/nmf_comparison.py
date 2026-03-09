"""Prefect flow for NMF (Non-negative Matrix Factorization) model comparison.

NMF serves as the third evaluation baseline alongside LDA:
  BERTopic → neural / embedding-based  (primary system)
  LDA      → generative probabilistic  (baseline 1)
  NMF      → matrix-factorisation      (baseline 2, this module)
"""
from prefect import flow, get_run_logger
from typing import List, Dict, Any
import time
from datetime import datetime

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


@flow(name="nmf-modeling-flow", log_prints=True)
def nmf_comparison_flow(
    documents: List[str],
    num_topics: int,
    batch_id: str,
    window_start: str,
    window_end: str,
) -> Dict[str, Any]:
    """
    Train an NMF model and compute evaluation metrics for three-way comparison
    with BERTopic and LDA.

    Steps
    -----
    1. Preprocess documents → TF-IDF matrix + Gensim artefacts for coherence
    2. Train NMF with ``num_topics`` components
    3. Compute coherence (C_v), diversity, and silhouette score
    4. Extract per-topic metadata
    5. Save metrics to ``outputs/metrics/nmf_metrics.json``

    Args:
        documents   : Document texts (same cumulative corpus as BERTopic / LDA).
        num_topics  : Number of topics (matched to BERTopic's auto-detected count).
        batch_id    : Unique batch identifier string.
        window_start: ISO-format window start date.
        window_end  : ISO-format window end date.

    Returns:
        Dictionary with all NMF metrics (mirrors LDA schema).
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info("=" * 80)
    logger.info("Starting NMF Comparison Flow")
    logger.info("=" * 80)
    logger.info(f"Batch         : {batch_id}")
    logger.info(f"Window        : {window_start}  →  {window_end}")
    logger.info(f"Documents     : {len(documents)}")
    logger.info(f"Target topics : {num_topics}")

    try:
        config = load_config()

        # ── Step 1 : preprocess ────────────────────────────────────────────────
        logger.info("Step 1: Preprocessing documents for NMF (TF-IDF)")
        t0 = time.time()
        tfidf_matrix, vectorizer, tokenized_docs, dictionary = (
            preprocess_documents_for_nmf_task(documents)
        )
        step1_duration = time.time() - t0
        logger.info(f"Preprocessing done in {step1_duration:.2f}s  "
                    f"(matrix: {tfidf_matrix.shape[0]} × {tfidf_matrix.shape[1]})")

        if tfidf_matrix.shape[0] < 10:
            logger.warning(f"Too few documents ({tfidf_matrix.shape[0]}) — skipping NMF")
            return {
                'status': 'skipped',
                'reason': 'insufficient_documents',
                'documents_processed': tfidf_matrix.shape[0],
            }

        # Guard: NMF H-matrix collapses when num_topics ≈ n_features.
        # Cap at min(n_samples, n_features) // 3 for well-conditioned factors.
        n_s, n_f = tfidf_matrix.shape
        max_topics = max(2, min(n_s, n_f) // 3)
        if num_topics > max_topics:
            logger.warning(f"Clamping num_topics {num_topics} → {max_topics} (vocab={n_f})")
            num_topics = max_topics

        # ── Step 2 : train NMF ────────────────────────────────────────────────
        logger.info("Step 2: Training NMF model")
        t0 = time.time()

        # Pull NMF config with safe defaults
        if hasattr(config, 'nmf'):
            max_iter = getattr(config.nmf, 'max_iter', 400)
            alpha_W = getattr(config.nmf, 'alpha_W', 0.1)
            alpha_H = getattr(config.nmf, 'alpha_H', 0.1)
        else:
            logger.warning("NMF config not found — using defaults")
            max_iter, alpha_W, alpha_H = 400, 0.1, 0.1

        nmf_model = train_nmf_model_task(
            tfidf_matrix=tfidf_matrix,
            num_topics=num_topics,
            max_iter=max_iter,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
        )
        step2_duration = time.time() - t0
        logger.info(f"NMF training done in {step2_duration:.2f}s")

        feature_names = vectorizer.get_feature_names_out().tolist()
        top_n = getattr(config.model, 'top_n_words', 10)

        # ── Step 3 : metrics ──────────────────────────────────────────────────
        logger.info("Step 3: Computing evaluation metrics")
        t0 = time.time()

        logger.info("  Coherence (C_v) …")
        coherence_cv = calculate_nmf_coherence_task(
            model=nmf_model,
            feature_names=feature_names,
            tokenized_docs=tokenized_docs,
            dictionary=dictionary,
            top_n=top_n,
            coherence_type='c_v',
        )

        logger.info("  Diversity …")
        diversity = calculate_nmf_diversity_task(
            model=nmf_model,
            feature_names=feature_names,
            top_n=top_n,
        )

        logger.info("  Silhouette …")
        silhouette = calculate_nmf_silhouette_task(
            tfidf_matrix=tfidf_matrix,
            model=nmf_model,
        )

        step3_duration = time.time() - t0
        logger.info(f"Metrics computed in {step3_duration:.2f}s")

        # ── Step 4 : metadata ─────────────────────────────────────────────────
        logger.info("Step 4: Extracting model metadata")
        metadata = extract_nmf_metadata_task(
            model=nmf_model,
            tfidf_matrix=tfidf_matrix,
            feature_names=feature_names,
            top_n=top_n,
        )

        # ── Step 5 : compile & save ───────────────────────────────────────────
        flow_duration = time.time() - flow_start
        metrics: Dict[str, Any] = {
            'status': 'success',
            'batch_id': batch_id,
            'window_start': window_start,
            'window_end': window_end,
            'timestamp': datetime.now().isoformat(),

            # Model info
            'num_topics': num_topics,
            'num_documents': int(tfidf_matrix.shape[0]),
            'vocabulary_size': int(tfidf_matrix.shape[1]),
            'reconstruction_error': float(nmf_model.reconstruction_err_),

            # Evaluation metrics
            'coherence_c_v': float(coherence_cv),
            'diversity': float(diversity),
            'silhouette_score': float(silhouette),

            # Timing
            'preprocessing_time_seconds': step1_duration,
            'training_time_seconds': step2_duration,
            'evaluation_time_seconds': step3_duration,
            'total_time_seconds': flow_duration,

            # Per-topic data
            'topics': metadata['topics'],

            # Config snapshot
            'nmf_config': {
                'max_iter': max_iter,
                'alpha_W': alpha_W,
                'alpha_H': alpha_H,
                'top_n_words': top_n,
            },
        }

        logger.info("Step 5: Saving metrics")
        save_nmf_metrics_task(metrics)

        logger.info("=" * 80)
        logger.info("NMF Comparison Flow — COMPLETE")
        logger.info(f"  Duration      : {flow_duration:.2f}s")
        logger.info(f"  Topics        : {num_topics}")
        logger.info(f"  Coherence C_v : {coherence_cv:.4f}")
        logger.info(f"  Diversity     : {diversity:.4f}")
        logger.info(f"  Silhouette    : {silhouette:.4f}")
        logger.info("=" * 80)

        return metrics

    except Exception as exc:
        logger.error(f"NMF comparison flow failed: {exc}", exc_info=True)
        return {
            'status': 'error',
            'error_message': str(exc),
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
        }


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    _test_docs = [
        "I need help with my billing and payment method",
        "The app keeps crashing on startup every time",
        "How do I reset my password for account recovery",
        "Network connectivity issues with the firewall settings",
        "Database timeout errors during peak usage",
        "Mobile interface not responsive on iOS devices",
        "Email notifications stopped working after update",
        "Login authentication failing repeatedly today",
        "Subscription renewal failed payment declined",
        "Security alert suspicious login detected account",
    ] * 15  # Enough docs for meaningful metrics

    result = nmf_comparison_flow(
        documents=_test_docs,
        num_topics=5,
        batch_id="test_nmf_batch",
        window_start="2024-01-01",
        window_end="2024-01-02",
    )
    print("\nNMF Metrics:")
    for k in ('coherence_c_v', 'diversity', 'silhouette_score', 'num_topics'):
        print(f"  {k}: {result.get(k, 'N/A')}")
