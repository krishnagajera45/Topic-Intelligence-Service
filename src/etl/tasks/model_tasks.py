"""Prefect tasks for model training (with extracted BERTopic logic)."""
from prefect import task, get_run_logger
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from src.utils import load_config, StorageManager
from src.utils.ollama_client import generate_topic_label
from src.utils.model_versioning import ModelVersionManager


@task(name="initialize-bertopic-model", retries=1)
def initialize_bertopic_model_task(config: Any = None) -> BERTopic:
    """
    Initialize BERTopic model with all components (logic moved from BERTopicOnlineWrapper).
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized BERTopic model
    """
    logger = get_run_logger()
    logger.info("Initializing BERTopic model components")
    
    if config is None:
        config = load_config()
    
    # Sentence transformer for embeddings
    sentence_model = SentenceTransformer(config.model.embedding_model)
    logger.info(f"Loaded embedding model: {config.model.embedding_model}")
    
    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=config.model.umap_n_neighbors,
        n_components=config.model.umap_n_components,
        min_dist=config.model.umap_min_dist,
        metric=config.model.umap_metric,
        random_state=42
    )
    logger.info(f"Initialized UMAP with {config.model.umap_n_components} components")
    
    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.model.min_cluster_size,
        min_samples=config.model.min_samples,
        metric=config.model.hdbscan_metric,
        prediction_data=True
    )
    logger.info(f"Initialized HDBSCAN with min_cluster_size={config.model.min_cluster_size}")
    
    # CountVectorizer for bag-of-words representation
    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=config.model.min_df,
        max_df=config.model.max_df,
        ngram_range=tuple(config.model.ngram_range)
    )
    logger.info(f"Initialized CountVectorizer with ngram_range={config.model.ngram_range}")
    
    # C-TF-IDF for topic representation
    # c-TF-IDF = (word frequency in topic) × log(total topics / topics containing word)
    ctfidf_model = ClassTfidfTransformer()
    logger.info("Initialized C-TF-IDF transformer")
    
    # Create BERTopic model
    model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    logger.info("BERTopic model initialized successfully")
    return model


@task(name="fit-seed-model", retries=1)
def fit_seed_model_task(
    model: BERTopic,
    documents: List[str]
) -> Tuple[np.ndarray, np.ndarray, BERTopic]:
    """
    Fit BERTopic model on documents (seed training).
    
    Args:
        model: Initialized BERTopic model
        documents: List of document texts
        
    Returns:
        Tuple of (topics, probabilities, fitted_model)
    """
    logger = get_run_logger()
    logger.info(f"Fitting seed model on {len(documents)} documents")
    
    topics, probs = model.fit_transform(documents)
    
    logger.info(f"Model training complete. Found {len(set(topics))} topics")
    return topics, probs, model


@task(name="load-bertopic-model", retries=1)
def load_bertopic_model_task(model_path: str) -> BERTopic:
    """
    Load BERTopic model from disk.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded BERTopic model
    """
    logger = get_run_logger()
    logger.info(f"Loading BERTopic model from {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = BERTopic.load(model_path)
    logger.info(f"Model loaded successfully")
    return model


@task(name="save-bertopic-model", retries=1)
def save_bertopic_model_task(model: BERTopic, model_path: str) -> str:
    """
    Save BERTopic model to disk.
    
    Args:
        model: BERTopic model to save
        model_path: Path to save model
        
    Returns:
        Model path
    """
    logger = get_run_logger()
    logger.info(f"Saving BERTopic model to {model_path}")
    
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, serialization="pickle")
    
    logger.info(f"Model saved successfully")
    return model_path


@task(name="transform-documents", retries=1)
def transform_documents_task(
    model: BERTopic,
    documents: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform documents using existing BERTopic model (logic moved from BERTopicOnlineWrapper).
    
    Args:
        model: Fitted BERTopic model
        documents: List of document texts
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    logger.info(f"Transforming {len(documents)} documents with existing model")
    
    try:
        topics, probs = model.transform(documents)
        logger.info(f"Documents transformed successfully")
        return topics, probs
    except IndexError as e:
        # Fallback: disable probability mapping when topics mismatch
        logger.warning(
            f"Transform probability mapping failed ({e}); "
            "retrying without probabilities"
        )
        prev_calc = getattr(model, "calculate_probabilities", True)
        try:
            model.calculate_probabilities = False
            topics, probs = model.transform(documents)
            return topics, probs
        finally:
            model.calculate_probabilities = prev_calc


@task(name="update-topic-representations", retries=1)
def update_topic_representations_task(
    model: BERTopic,
    documents: List[str],
    topics: np.ndarray
) -> BERTopic:
    """
    Update topic representations with new documents (logic moved from BERTopicOnlineWrapper).
    
    Re-calculates:
    - Bag-of-words for topics
    - Re-ranks keywords with c-TF-IDF
    - Updates topic labels
    
    Args:
        model: BERTopic model
        documents: Document texts
        topics: Topic assignments
        
    Returns:
        Updated BERTopic model
    """
    logger = get_run_logger()
    logger.info(f"Updating topic representations with {len(documents)} documents")
    
    # Recreate vectorizer model with same config
    config = load_config()
    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=config.model.min_df,
        max_df=config.model.max_df,
        ngram_range=tuple(config.model.ngram_range)
    )
    
    model.update_topics(documents, topics=topics, vectorizer_model=vectorizer_model)
    
    logger.info("Topic representations updated successfully")
    return model


@task(name="archive-model-file", retries=1)
def archive_model_file_task(current_path: str, previous_path: str) -> str:
    """
    Archive current model as previous version (logic moved from BERTopicOnlineWrapper).
    
    Args:
        current_path: Path to current model
        previous_path: Path to save as previous model
        
    Returns:
        Previous model path
    """
    logger = get_run_logger()
    logger.info(f"Archiving model: {current_path} → {previous_path}")
    
    current_path_obj = Path(current_path)
    previous_path_obj = Path(previous_path)
    
    if current_path_obj.exists():
        previous_path_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path_obj, previous_path_obj)
        logger.info(f"Model archived successfully")
    else:
        logger.warning(f"No current model to archive at {current_path}")
    
    return str(previous_path_obj)


@task(name="extract-topic-metadata", retries=1)
def extract_topic_metadata_task(
    model: BERTopic,
    batch_id: str,
    window_start: str,
    window_end: str,
    config: Any = None
) -> List[Dict[str, Any]]:
    """
    Extract topic metadata from model.
    
    Args:
        model: BERTopic model
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        config: Configuration object
        
    Returns:
        List of topic metadata dictionaries
    """
    logger = get_run_logger()
    logger.info("Extracting topic metadata")
    
    if config is None:
        config = load_config()
    
    topic_info = model.get_topic_info()
    topics_metadata = []
    
    for _, row in topic_info.iterrows():
        topic_id = int(row['Topic'])
        
        # Skip outlier topic
        if topic_id == -1:
            continue
        
        # Get top words for this topic
        topic_words = model.get_topic(topic_id)
        if topic_words:
            top_words = [word for word, _ in topic_words[:config.model.top_n_words]]
            # Generate custom label from configurable top words for label
            custom_label = ", ".join([word for word, _ in topic_words[:config.model.top_words_for_label]])
        else:
            top_words = []
            custom_label = f"Topic {topic_id}"
        gpt_label = None
        gpt_summary = None

        # Optional: Use Ollama to generate better label/summary
        if getattr(config, "ollama", None) and config.ollama.enabled:
            examples = []
            try:
                if hasattr(model, "get_representative_docs"):
                    examples = model.get_representative_docs(topic_id) or []
            except Exception:
                examples = []

            result = generate_topic_label(
                base_url=config.ollama.base_url,
                model=config.ollama.model,
                top_words=top_words,
                examples=examples,
                prompt_template=config.ollama.prompt_template,
                temperature=config.ollama.temperature,
                max_tokens=config.ollama.max_tokens,
                timeout_seconds=config.ollama.timeout_seconds,
                examples_limit=config.ollama.examples_limit
            )
            gpt_label = result.get("label")
            gpt_summary = result.get("summary")

            # Prefer GPT label when available
            if gpt_label:
                custom_label = gpt_label
        
        topic_metadata = {
            'topic_id': topic_id,
            'custom_label': custom_label,
            'top_words': top_words,
            'size': int(row['Count']),
            'created_at': datetime.now().isoformat(),
            'batch_id': batch_id,
            'window_start': window_start,
            'window_end': window_end,
            'count': int(row['Count']),
            'gpt_label': gpt_label,
            'gpt_summary': gpt_summary
        }
        topics_metadata.append(topic_metadata)
    
    logger.info(f"Extracted metadata for {len(topics_metadata)} topics")
    return topics_metadata


@task(name="save-topic-metadata", retries=1)
def save_topic_metadata_task(topics_metadata: List[Dict[str, Any]]) -> int:
    """
    Save topic metadata to storage.
    
    Args:
        topics_metadata: List of topic metadata dictionaries
        
    Returns:
        Number of topics saved
    """
    logger = get_run_logger()
    logger.info(f"Saving metadata for {len(topics_metadata)} topics")
    
    config = load_config()
    storage = StorageManager(config)
    storage.save_topics_metadata(topics_metadata)
    
    logger.info(f"Topic metadata saved successfully")
    return len(topics_metadata)


# Original task wrappers (now calling granular tasks)
@task(name="train_seed_model", retries=1)
def train_seed_model_task(
    documents: List[str],
    batch_id: str,
    window_start: str,
    window_end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train initial seed BERTopic model (orchestrates granular tasks).
    
    Args:
        documents: List of document texts
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    logger.info(f"Training seed model (orchestrating granular tasks)")
    
    config = load_config()
    
    # Step 1: Initialize model (task)
    logger.info("Step 1: Initializing BERTopic model")
    model = initialize_bertopic_model_task(config)
    
    # Step 2: Fit model (task)
    logger.info("Step 2: Fitting model on documents")
    topics, probs, model = fit_seed_model_task(model, documents)
    
    # Step 3: Save model (task)
    logger.info("Step 3: Saving model to disk")
    save_bertopic_model_task(model, config.storage.current_model_path)
    
    # Step 4: Extract metadata (task)
    logger.info("Step 4: Extracting topic metadata")
    topics_metadata = extract_topic_metadata_task(
        model, batch_id, window_start, window_end, config
    )
    
    # Step 5: Save metadata (task)
    logger.info("Step 5: Saving topic metadata")
    save_topic_metadata_task(topics_metadata)
    
    logger.info(f"Seed model trained successfully with {len(set(topics))} topics")
    return topics, probs


@task(name="update_model_online", retries=1)
def update_model_online_task(
    documents: List[str],
    batch_id: str,
    window_start: str,
    window_end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update model with new batch (online learning) - orchestrates granular tasks.
    
    Args:
        documents: New documents
        batch_id: Batch identifier
        window_start: Window start
        window_end: Window end
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    logger.info(f"Updating model online (orchestrating granular tasks)")
    
    config = load_config()
    
    # Step 1: Archive current model (task)
    logger.info("Step 1: Archiving current model as previous")
    archive_model_file_task(
        current_path=config.storage.current_model_path,
        previous_path=config.storage.previous_model_path
    )
    
    # Step 2: Load existing model (task)
    logger.info("Step 2: Loading current model")
    model = load_bertopic_model_task(config.storage.current_model_path)
    
    # Step 3: Transform documents (task)
    logger.info("Step 3: Transforming documents with existing model")
    topics, probs = transform_documents_task(model, documents)
    
    # Ensure probs is always available for downstream tasks
    if probs is None:
        probs = np.array([0.0] * len(documents))
    
    # Step 4: Update topic representations (task)
    logger.info("Step 4: Updating topic representations")
    model = update_topic_representations_task(model, documents, topics)
    
    # Step 5: Save updated model (task)
    logger.info("Step 5: Saving updated model")
    save_bertopic_model_task(model, config.storage.current_model_path)
    
    # Step 6: Extract metadata (task)
    logger.info("Step 6: Extracting topic metadata")
    topics_metadata = extract_topic_metadata_task(
        model, batch_id, window_start, window_end, config
    )
    
    # Step 7: Save metadata (task)
    logger.info("Step 7: Saving topic metadata")
    save_topic_metadata_task(topics_metadata)
    
    logger.info(f"Model updated successfully with {len(set(topics))} topics")
    return topics, probs


@task(name="merge-models", retries=1)
def merge_models_task(
    base_model: BERTopic,
    batch_model: BERTopic,
    min_similarity: Optional[float] = None,
    batch_id: str = None
) -> Tuple[BERTopic, Dict[str, Any]]:
    """
    Merge batch-trained model with base model using BERTopic.merge_models.
    
    This implements the "batch retrain + merge_models" pattern:
    1. Train fresh model on new batch (batch_model)
    2. Merge with base model (already includes HITL merges/splits)
    3. Save merged as new base
    
    Args:
        base_model: Base BERTopic model (includes HITL changes)
        batch_model: Newly trained model on batch data
        min_similarity: Minimum similarity threshold for merging topics (0.0-1.0)
                       Higher = stricter (only very similar topics merge)
        batch_id: Batch identifier for logging
        
    Returns:
        Tuple of (merged_model, merge_info_dict)
    """
    logger = get_run_logger()
    
    # Load config for min_similarity if not provided
    if min_similarity is None:
        config = load_config()
        min_similarity = config.model.min_similarity
    
    logger.info(f"Merging batch model with base model (min_similarity={min_similarity})")
    logger.info(f"Base model topics: {len(base_model.get_topics())}")
    logger.info(f"Batch model topics: {len(batch_model.get_topics())}")
    
    try:
        # Merge models using BERTopic's merge_models
        merged_model = BERTopic.merge_models(
            models=[base_model, batch_model],
            min_similarity=min_similarity
        )
        
        merged_topics = len(merged_model.get_topics())
        logger.info(f"Merged model topics: {merged_topics}")
        
        # Create merge info for logging
        base_model_topics = len(base_model.get_topics())
        batch_model_topics = len(batch_model.get_topics())
        
        merge_info = {
            'merge_strategy': 'batch_retrain_merge_models',
            'min_similarity': min_similarity,
            'base_model_topics': base_model_topics,
            'batch_model_topics': batch_model_topics,
            'merged_model_topics': merged_topics,
            'batch_id': batch_id,
            'merged_at': datetime.now().isoformat()
        }
        
        logger.info(f"Merge completed successfully. Merge info: {merge_info}")
        return merged_model, merge_info
    
    except Exception as e:
        logger.error(f"Error merging models: {e}", exc_info=True)
        raise


@task(name="train_batch_and_merge_models", retries=1)
def train_batch_and_merge_models_task(
    documents: List[str],
    batch_id: str,
    window_start: str,
    window_end: str,
    min_similarity: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train fresh model on batch and merge with base model (recommended approach).
    
    This implements the "batch retrain + merge_models" pattern:
    1. Initialize fresh BERTopic model
    2. Train on new batch documents
    3. Load base model (includes all HITL merges/splits)
    4. Merge batch model with base using merge_models
    5. Save merged as new base
    6. Archive previous base
    
    Args:
        documents: New batch documents
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        min_similarity: Similarity threshold for merging topics
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    logger.info(f"Training batch model and merging with base (batch retrain + merge_models)")
    
    config = load_config()
    model_base_dir = str(Path(config.storage.current_model_path).parent.parent)
    version_manager = ModelVersionManager(base_dir=model_base_dir)
    
    # Use config min_similarity if not provided
    if min_similarity is None:
        min_similarity = config.model.min_similarity
    
    # Step 1: Initialize and train fresh model on batch
    logger.info("Step 1: Training fresh model on batch documents")
    batch_model = initialize_bertopic_model_task(config)
    topics, probs, batch_model = fit_seed_model_task(batch_model, documents)
    logger.info(f"Batch model trained: {len(set(topics))} topics")
    
    # Step 2: Load base model (contains HITL merges/splits)
    logger.info("Step 2: Loading base model")
    base_model_path = version_manager.get_current_model_path()
    
    if Path(base_model_path).exists():
        # Base model exists → merge with it
        base_model = load_bertopic_model_task(base_model_path)
        
        # Step 3: Merge models
        logger.info("Step 3: Merging batch model with base model")
        merged_model, merge_info = merge_models_task(
            base_model=base_model,
            batch_model=batch_model,
            min_similarity=min_similarity,
            batch_id=batch_id
        )
        
        # Step 4: Archive previous base(copy previous model in to previous and archive folder with metadata)
        logger.info("Step 4: Archiving current model as previous")
        archived_path, timestamp = version_manager.archive_current_as_previous()
        logger.info(f"Archived to: {archived_path} (timestamp: {timestamp})")
        
        # Step 5: Save merged model as new base
        logger.info("Step 5: Saving merged model as new base")
        model_to_save = merged_model
        merge_info['archived_previous_at'] = timestamp
    else:
        # No base model exists → use batch model as base
        logger.info("No base model found → using batch model as base")
        model_to_save = batch_model
        merge_info = {
            'merge_strategy': 'first_run_seed',
            'batch_id': batch_id,
            'saved_at': datetime.now().isoformat()
        }
    
    # Save model
    logger.info("Saving model to disk")
    save_bertopic_model_task(model_to_save, base_model_path)

    # Note: Cumulative corpus is maintained by model_training_flow
    # Each batch appends its documents to the corpus file

    # Save merge metadata
    version_manager.save_model_metadata(base_model_path, merge_info)

    # Extract and save topic metadata
    logger.info("Extracting and saving topic metadata")
    topics_metadata = extract_topic_metadata_task(
        model_to_save, batch_id, window_start, window_end, config
    )
    save_topic_metadata_task(topics_metadata)

    # Get final topic assignments from the merged/saved model
    logger.info("Getting final topic assignments from merged model")
    final_topics, final_probs = transform_documents_task(model_to_save, documents)

    return final_topics, final_probs


@task(name="archive_model")
def archive_model_task() -> None:
    """
    Archive current model as previous version (simplified wrapper).
    
    NOTE: This is a simplified wrapper for backward compatibility.
    Use archive_model_file_task for more control.
    """
    logger = get_run_logger()
    config = load_config()
    
    archive_model_file_task(
        current_path=config.storage.current_model_path,
        previous_path=config.storage.previous_model_path
    )
