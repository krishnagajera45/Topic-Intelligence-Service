"""Prefect tasks for NMF (Non-negative Matrix Factorization) model training and evaluation.

NMF is the third baseline alongside LDA for benchmarking BERTopic.
It operates on TF-IDF features — a different paradigm from both:
  - BERTopic  : contextual transformer embeddings + HDBSCAN clustering
  - LDA       : bag-of-words + Dirichlet generative model
  - NMF (this): TF-IDF matrix factorization (deterministic optimisation)
"""
from prefect import task, get_run_logger
from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import time
import json

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
for resource, path in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(path, quiet=True)


# ── Preprocessing ──────────────────────────────────────────────────────────────

@task(name="preprocess-for-nmf", retries=1)
def preprocess_documents_for_nmf_task(
    documents: List[str],
    max_features: int = 5000,
) -> Tuple[np.ndarray, TfidfVectorizer, List[List[str]], Dictionary]:
    """
    Preprocess documents for NMF using TF-IDF vectorisation.

    Args:
        documents: Raw document texts.
        max_features: Vocabulary size cap for TF-IDF.

    Returns:
        (tfidf_matrix, vectorizer, tokenized_docs, gensim_dictionary)
        - tfidf_matrix     : (n_docs, vocab) sparse matrix used to train NMF
        - vectorizer       : fitted TfidfVectorizer (needed to inspect feature names)
        - tokenized_docs   : list-of-token-lists for coherence scoring via Gensim
        - gensim_dictionary: Gensim Dictionary built from tokenized_docs
    """
    logger = get_run_logger()
    logger.info(f"Preprocessing {len(documents)} documents for NMF (TF-IDF)")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenise for Gensim coherence scoring
    tokenized_docs = []
    for doc in documents:
        tokens = simple_preprocess(doc, deacc=True, min_len=3)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) >= 3]
        if tokens:
            tokenized_docs.append(tokens)

    # Build Gensim dictionary (coherence only).
    # keep_n=10000 matches LDA and BERTopic preprocessing so all three models
    # use the same vocabulary budget for coherence scoring.
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)

    # TF-IDF matrix used for NMF training.
    # ngram_range=(1,1) — unigrams only.
    # Bigrams would appear as NMF topic words but the Gensim dictionary only
    # holds unigrams, so bigrams get silently filtered out during coherence
    # scoring and diversity computation, making those metrics unfairly deflated.
    # Keeping unigrams ensures every topic word can be matched in the reference
    # corpus, giving a like-for-like comparison with LDA and BERTopic.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 1),
        stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    logger.info(
        f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features"
    )
    logger.info(f"Gensim dictionary: {len(dictionary)} tokens")
    return tfidf_matrix, vectorizer, tokenized_docs, dictionary


# ── Model training ─────────────────────────────────────────────────────────────

@task(name="train-nmf-model", retries=1)
def train_nmf_model_task(
    tfidf_matrix: np.ndarray,
    num_topics: int,
    random_state: int = 42,
    max_iter: int = 400,
    alpha_W: float = 0.1,
    alpha_H: float = 0.1,
) -> NMF:
    """
    Train an NMF model on the TF-IDF matrix.

    Args:
        tfidf_matrix : Document-term TF-IDF matrix.
        num_topics   : Number of latent topics (components).
        random_state : Reproducibility seed.
        max_iter     : Max solver iterations.
        alpha_W      : L1 regularisation weight for document-topic matrix W.
        alpha_H      : L1 regularisation weight for topic-word matrix H.

    Returns:
        Fitted sklearn NMF model.
    """
    logger = get_run_logger()
    # Hard cap: NMF H-matrix collapses when num_topics ≈ n_features.
    # Keep topics ≤ min(n_samples, n_features) // 3 for well-conditioned factors.
    n_samples, n_features = tfidf_matrix.shape
    safe_max = max(2, min(n_samples, n_features) // 3)
    if num_topics > safe_max:
        logger.warning(f"Clamping num_topics {num_topics} → {safe_max} (vocab={n_features})")
        num_topics = safe_max

    logger.info(f"Training NMF model: {num_topics} topics, max_iter={max_iter}")

    start = time.time()
    # Use nndsvda init (SVD-based, deterministic) — more stable than 'random'.
    # Regularisation (alpha_W / alpha_H) is deliberately set to 0 because even
    # small L1/L2 penalties drive the H matrix to all-zeros on the small,
    # sparse TF-IDF matrices produced by short-text batches, which causes
    # model.transform() to raise "Array passed to NMF is full of zeros".
    model = NMF(
        n_components=num_topics,
        random_state=random_state,
        max_iter=max_iter,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        init='nndsvda',
        solver='cd',
        tol=1e-4,
    )
    model.fit(tfidf_matrix)
    elapsed = time.time() - start
    logger.info(f"NMF training completed in {elapsed:.2f}s  (reconstruction error: {model.reconstruction_err_:.4f})")
    return model


# ── Metrics ────────────────────────────────────────────────────────────────────

def _get_nmf_topic_words(model: NMF, feature_names: List[str], top_n: int = 10) -> List[List[str]]:
    """Return top-N words for each NMF topic (H matrix)."""
    topics = []
    for topic_idx, component in enumerate(model.components_):
        top_indices = np.argsort(component)[::-1][:top_n]
        topics.append([feature_names[i] for i in top_indices])
    return topics


@task(name="calculate-nmf-coherence", retries=1)
def calculate_nmf_coherence_task(
    model: NMF,
    feature_names: List[str],
    tokenized_docs: List[List[str]],
    dictionary: Dictionary,
    top_n: int = 10,
    coherence_type: str = 'c_v',
) -> float:
    """
    Calculate topic coherence for NMF using Gensim's CoherenceModel.

    The H matrix (topics × words) encodes the NMF word weights; we extract
    the top-N words per topic and feed them to Gensim's C_v coherence scorer
    which evaluates semantic similarity via NPMI over the original corpus.

    Args:
        model          : Trained NMF model.
        feature_names  : Vocabulary (from TfidfVectorizer.get_feature_names_out()).
        tokenized_docs : Tokenised documents for reference corpus.
        dictionary     : Gensim dictionary.
        top_n          : Words per topic.
        coherence_type : Gensim coherence metric ('c_v', 'u_mass', etc.).

    Returns:
        Coherence score (float).
    """
    logger = get_run_logger()
    logger.info(f"Calculating NMF coherence ({coherence_type}), top_n={top_n}")

    topics_words = _get_nmf_topic_words(model, feature_names, top_n)

    # Filter topic word lists to only include words known to the dictionary
    valid_topics = []
    for words in topics_words:
        filtered = [w for w in words if w in dictionary.token2id]
        if len(filtered) >= 2:
            valid_topics.append(filtered)

    if not valid_topics:
        logger.warning("No valid topics for coherence — returning 0.0")
        return 0.0

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=coherence_type,
    )
    score = coherence_model.get_coherence()
    logger.info(f"NMF Coherence ({coherence_type}): {score:.4f}")
    return float(score)


@task(name="calculate-nmf-diversity", retries=1)
def calculate_nmf_diversity_task(model: NMF, feature_names: List[str], top_n: int = 10) -> float:
    """
    Calculate topic diversity — proportion of unique words across all NMF topics.

    Formula: |unique words across all topics| / (top_n × num_topics)

    Args:
        model        : Trained NMF model.
        feature_names: Vocabulary list.
        top_n        : Words per topic.

    Returns:
        Diversity score in [0, 1].
    """
    logger = get_run_logger()
    all_words = []
    for words in _get_nmf_topic_words(model, feature_names, top_n):
        all_words.extend(words)

    unique = len(set(all_words))
    total = len(all_words)
    diversity = unique / total if total > 0 else 0.0
    logger.info(f"NMF Diversity: {diversity:.4f}  ({unique}/{total} unique)")
    return diversity


@task(name="calculate-nmf-silhouette", retries=1)
def calculate_nmf_silhouette_task(
    tfidf_matrix: np.ndarray,
    model: NMF,
    sample_size: int = 1000,
) -> float:
    """
    Calculate silhouette score for NMF topic assignments.

    Each document is assigned to its dominant topic (argmax of W row).
    Silhouette is then computed on the TF-IDF vectors in the original
    feature space (cosine distance, capped sample for efficiency).

    Args:
        tfidf_matrix: Document-term TF-IDF matrix.
        model       : Trained NMF model.
        sample_size : Max docs sampled for silhouette (speed / memory).

    Returns:
        Silhouette score in [-1, 1].
    """
    logger = get_run_logger()
    logger.info("Calculating NMF silhouette score")

    try:
        try:
            W = model.transform(tfidf_matrix)
        except ValueError as e:
            logger.warning(f"NMF transform failed ({e}) — returning 0.0")
            return 0.0

        # Filter out zero-norm document rows.
        # When a document's TF-IDF row is all-zero (out-of-vocabulary), NMF
        # transform returns a zero W row and argmax assigns it to topic 0,
        # collapsing all labels to a single value → silhouette undefined.
        from scipy.sparse import issparse
        if issparse(tfidf_matrix):
            row_norms = np.array(tfidf_matrix.power(2).sum(axis=1)).ravel()
        else:
            row_norms = np.linalg.norm(tfidf_matrix, axis=1)
        nonzero_mask = row_norms > 0

        W_filtered = W[nonzero_mask]
        X_filtered = tfidf_matrix[nonzero_mask]

        if W_filtered.shape[0] < 10:
            logger.warning("Too few non-zero document rows for silhouette — returning 0.0")
            return 0.0

        labels = np.argmax(W_filtered, axis=1)
        unique_topics = np.unique(labels)
        if len(unique_topics) < 2:
            logger.warning(f"Only {len(unique_topics)} unique topic label(s) after filtering — returning 0.0")
            return 0.0

        n = X_filtered.shape[0]
        actual_sample = min(sample_size, n)
        idx = np.random.default_rng(42).choice(n, size=actual_sample, replace=False)
        X_sample = X_filtered[idx]
        y_sample = labels[idx]

        # Ensure at least 2 classes in the sample
        if len(np.unique(y_sample)) < 2:
            logger.warning("Sample has fewer than 2 classes — expanding to full filtered set")
            X_sample = X_filtered
            y_sample = labels

        score = float(silhouette_score(X_sample, y_sample, metric='cosine'))
        logger.info(f"NMF Silhouette: {score:.4f}")
        return score

    except Exception as exc:
        logger.warning(f"Silhouette calculation failed: {exc}")
        return 0.0


# ── Metadata extraction ────────────────────────────────────────────────────────

@task(name="extract-nmf-metadata", retries=1)
def extract_nmf_metadata_task(
    model: NMF,
    tfidf_matrix: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Extract per-topic metadata from the NMF model.

    Args:
        model        : Trained NMF model.
        tfidf_matrix : Document-term matrix (for doc-count per topic).
        feature_names: Vocabulary list.
        top_n        : Words per topic.

    Returns:
        {num_topics, num_documents, topics: [{topic_id, top_words, word_weights, document_count}]}
    """
    logger = get_run_logger()

    # Try to get per-document topic assignments; fall back gracefully if
    # the transform fails (e.g. degenerate H matrix on tiny vocabularies).
    doc_assignments = None
    try:
        W = model.transform(tfidf_matrix)          # (n_docs, n_topics)
        doc_assignments = np.argmax(W, axis=1)
    except Exception as exc:
        logger.warning(f"model.transform failed ({exc}) — doc counts will be 0")

    topics_info = []
    for topic_id, component in enumerate(model.components_):
        top_indices = np.argsort(component)[::-1][:top_n]
        words = [feature_names[i] for i in top_indices]
        weights = [float(component[i]) for i in top_indices]
        doc_count = (
            int(np.sum(doc_assignments == topic_id))
            if doc_assignments is not None else 0
        )
        topics_info.append({
            'topic_id': topic_id,
            'top_words': words,
            'word_weights': weights,
            'document_count': doc_count,
        })

    logger.info(f"Extracted NMF metadata: {model.n_components} topics")
    return {
        'num_topics': model.n_components,
        'num_documents': tfidf_matrix.shape[0],
        'topics': topics_info,
    }


# ── Persistence ────────────────────────────────────────────────────────────────

@task(name="save-nmf-metrics", retries=1)
def save_nmf_metrics_task(
    metrics: Dict[str, Any],
    output_path: str = "outputs/metrics/nmf_metrics.json",
) -> str:
    """
    Persist NMF metrics to disk. Appends per-batch records for temporal analysis.

    Stored structure:
        {"batches": [{batch_id, coherence_c_v, diversity, silhouette_score, ...}],
         "latest": {...full metrics dict...}}

    Args:
        metrics    : Metrics dict (must include 'batch_id').
        output_path: File path.

    Returns:
        Absolute path of saved file.
    """
    logger = get_run_logger()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"batches": [], "latest": {}}
    if Path(output_path).exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            data.setdefault("batches", [])
        except Exception:
            data = {"batches": [], "latest": {}}

    batch_id = metrics.get("batch_id")
    batch_record = {
        "batch_id": batch_id,
        "coherence_c_v": metrics.get("coherence_c_v", 0.0),
        "diversity": metrics.get("diversity", 0.0),
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "num_topics": metrics.get("num_topics", 0),
        "timestamp": metrics.get("timestamp", ""),
        "training_time_seconds": metrics.get("training_time_seconds") or metrics.get("total_time_seconds", 0.0),
    }

    existing_ids = [b.get("batch_id") for b in data["batches"]]
    if batch_id in existing_ids:
        data["batches"][existing_ids.index(batch_id)] = batch_record
    else:
        data["batches"].append(batch_record)

    data["latest"] = {k: v for k, v in metrics.items() if k != "batches"}
    data["status"] = metrics.get("status", "success")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"NMF metrics saved → {output_path}  ({len(data['batches'])} batches total)")
    return output_path
