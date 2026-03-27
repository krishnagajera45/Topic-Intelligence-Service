"""Prefect flow for BERTopic modeling (training + evaluation metrics)."""
from prefect import flow, get_run_logger
from datetime import datetime
from typing import List, Tuple
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.etl.tasks.model_tasks import (
    train_seed_model_task,
    train_batch_and_merge_models_task,
)
from src.etl.tasks.bertopic_metrics import (
    calculate_bertopic_coherence_task,
    calculate_bertopic_silhouette_task,
    save_bertopic_metrics_task,
)
from src.etl.tasks.lda_tasks import preprocess_documents_for_lda_task
from src.utils import load_config, StorageManager
from src.dashboard.utils.api_client import APIClient


@flow(name="bertopic-modeling-flow", log_prints=True)
def bertopic_modeling_flow(
    documents: List[str],
    doc_ids: List[str],
    batch_id: str,
    window_start: str,
    window_end: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BERTopic modeling flow: train/merge model + calculate evaluation metrics.
    
    All BERTopic-related tasks in one organized flow:
    1. Model training (seed or batch retrain + merge_models)
    2. Document assignments & corpus management
    3. Evaluation metrics (coherence, diversity, silhouette)
    
    Args:
        documents: Document texts
        doc_ids: Document IDs
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    
    logger.info("=" * 70)
    logger.info("BERTopic Modeling Flow")
    logger.info("=" * 70)
    logger.info(f"Batch: {batch_id}")
    logger.info(f"Window: {window_start} to {window_end}")
    logger.info(f"Documents: {len(documents)}")
    
    config = load_config()
    storage = StorageManager(config)
    
    # ────────────────────────────────────────────────────────────────────
    # Step 1: Train or Merge Model
    # ────────────────────────────────────────────────────────────────────
    train_start = time.time()
    model_exists = Path(config.storage.current_model_path).exists()
    
    if model_exists:
        logger.info("📦 Existing model found → Batch retrain + merge_models")
        topics, probs = train_batch_and_merge_models_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    else:
        logger.info("🌱 No model found → Training seed model")
        topics, probs = train_seed_model_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    
    training_duration = time.time() - train_start
    logger.info(f"✅ Model training complete ({training_duration:.2f}s)")
    logger.info(f"   Topics discovered: {len(set(topics))}")
    
    # ────────────────────────────────────────────────────────────────────
    # Step 2: Save Document Assignments & Corpus
    # ────────────────────────────────────────────────────────────────────
    logger.info("💾 Saving document assignments and corpus")
    
    # Calculate confidence scores
    if probs is None:
        confidence = [0.0] * len(topics)
    elif len(probs.shape) == 2:
        confidence = [p.max() if len(p) > 0 else 0.0 for p in probs]
    else:
        confidence = probs.tolist()
    
    # Validate inputs
    if not doc_ids or not documents:
        raise ValueError("doc_ids and documents must be provided")
    if len(doc_ids) != len(documents):
        raise ValueError(f"Length mismatch: {len(doc_ids)} doc_ids vs {len(documents)} documents")
    if len(topics) != len(doc_ids):
        raise ValueError(f"Length mismatch: {len(topics)} topics vs {len(doc_ids)} doc_ids")
    
    # Create batch assignments
    batch_assignments = pd.DataFrame({
        'doc_id': doc_ids,
        'topic_id': topics,
        'timestamp': window_start,
        'batch_id': batch_id,
        'confidence': confidence
    })
    
    # Maintain cumulative assignments and corpus
    assignments_path = config.storage.doc_assignments_path
    model_corpus_path = str(Path(config.storage.current_model_path).parent / (Path(config.storage.current_model_path).stem + "_corpus.json"))
    
    try:
        # Load and append assignments
        if Path(assignments_path).exists():
            existing_assignments = pd.read_csv(assignments_path)
            cumulative_assignments = pd.concat([existing_assignments, batch_assignments], ignore_index=True)
            logger.info(f"   Appended {len(batch_assignments)} assignments (total: {len(cumulative_assignments)})")
        else:
            cumulative_assignments = batch_assignments
            logger.info(f"   Created new assignments: {len(batch_assignments)} rows")
        
        cumulative_assignments.to_csv(assignments_path, index=False)
        
        # Load and append corpus
        existing_corpus = []
        if Path(model_corpus_path).exists():
            with open(model_corpus_path, 'r') as f:
                existing_corpus = json.load(f)
        
        cumulative_corpus = existing_corpus + documents
        with open(model_corpus_path, 'w') as f:
            json.dump(cumulative_corpus, f)
        
        logger.info(f"   Corpus: {len(cumulative_corpus)} documents ({len(documents)} new)")
        
        # Verify alignment
        if len(cumulative_corpus) == len(cumulative_assignments):
            logger.info(f"   ✅ Perfect alignment: {len(cumulative_corpus)} documents")
        else:
            error_msg = f"❌ Alignment mismatch: {len(cumulative_corpus)} corpus vs {len(cumulative_assignments)} assignments"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Failed to save cumulative data: {e}", exc_info=True)
        raise
    
    # ────────────────────────────────────────────────────────────────────
    # Step 3: Calculate Evaluation Metrics
    # ────────────────────────────────────────────────────────────────────
    logger.info("📊 Calculating evaluation metrics")
    
    config = load_config()
    
    # Step 2: Calculate evaluation metrics from merged model
    logger.info("Calculating BERTopic evaluation metrics (from merged model)")
    try:
        from src.utils import load_bertopic_model
        model = load_bertopic_model(config.storage.current_model_path)

        # ── Shared tokenisation (same as LDA) for fair coherence comparison ──
        # Uses simple_preprocess + WordNetLemmatizer + no_below=5 / no_above=0.5
        try:
            shared_texts, shared_dict, _ = preprocess_documents_for_lda_task(documents)
            logger.info(
                f"Shared coherence tokens built: {len(shared_texts)} docs, "
                f"vocab={len(shared_dict)}"
            )
        except Exception as tok_err:
            logger.warning(f"Shared tokenisation failed ({tok_err}); coherence will use internal fallback")
            shared_texts, shared_dict = None, None

        coherence_metrics = calculate_bertopic_coherence_task(
            model, documents, topics,
            pre_texts=shared_texts,
            pre_dictionary=shared_dict,
        )
        try:
            silhouette_score = calculate_bertopic_silhouette_task(model, documents, topics)
        except Exception as e:
            logger.warning(f"Could not calculate silhouette: {e}")
            silhouette_score = 0.0
        
        # Diversity from topic keywords
        try:
            api_client = APIClient()
            live_stats = api_client.get_topics()
            all_keywords = []
            for t in live_stats:
                all_keywords.extend([w for w in t.get("top_words", [])])
            unique_kw = len(set(all_keywords))
            total_kw = len(all_keywords) if all_keywords else 1
            diversity = unique_kw / total_kw if total_kw else 0.0
        except Exception:
            diversity = 0.0
        
        bertopic_metrics = {
            'coherence_c_v': coherence_metrics.get('coherence_c_v', 0.0),
            'silhouette_score': silhouette_score,
            'num_topics': len(set(topics)) - (1 if -1 in topics else 0),
            'diversity': diversity,
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_duration,
        }
        save_bertopic_metrics_task(bertopic_metrics)
        logger.info(f"   Coherence: {bertopic_metrics['coherence_c_v']:.4f}")
        logger.info(f"   Diversity: {diversity:.4f}")
        logger.info(f"   Silhouette: {silhouette_score:.4f}")
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}", exc_info=True)
        logger.warning("Pipeline continues without metrics")
    
    logger.info("=" * 70)
    logger.info("✅ BERTopic modeling flow complete")
    logger.info("=" * 70)
    
    return topics, probs
