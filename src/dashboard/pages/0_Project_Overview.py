"""
📖 Project Overview — About the system, motivation, architecture, and data.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.components.theme import inject_custom_css, page_header, render_footer
from src.utils.config import load_config as _load_cfg
_cfg = _load_cfg()
_ds_name = _cfg.active_dataset  # e.g. "twitter" or "ag_news"
_ds_title = _ds_name.replace("_", " ").title()

st.set_page_config(page_title="Project Overview", page_icon="📖", layout="wide")
inject_custom_css()

page_header(
    "Project Overview",
    "Online BERTopic with Human-in-the-Loop for Customer Support Insights",
    "📖",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_intro, tab_arch, tab_data, tab_method, tab_eval, tab_compare, tab_team = st.tabs([
    "🎯 Introduction",
    "🏗️ Architecture",
    "📊 Data",
    "🔬 Methodology",
    "📈 Evaluation",
    "⚖️ Three-Model Comparison",
    "👥 About",
])

# ── INTRODUCTION ──────────────────────────────────────────────────────────────
with tab_intro:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ## Motivation

        Customer support channels on social media generate **millions of
        messages daily**.  Manually categorizing these messages is
        impractical — yet understanding the *topics* customers talk about is
        critical for improving products, reducing churn, and routing tickets
        to the right teams.

        Traditional topic models (LDA, NMF) struggle with:
        - Short, noisy social-media text
        - Evolving vocabulary over time
        - Lack of interpretable topic labels

        **BERTopic** solves these problems by combining contextual language
        models (Sentence-BERT) with density-based clustering, producing
        coherent and interpretable topics without the bag-of-words
        limitations.

        ## Objective

        This project implements an **end-to-end online topic modeling
        pipeline** that:

        1. **Ingests** customer support tweets in configurable time-window
           batches
        2. **Discovers topics** using BERTopic (Sentence-BERT → UMAP →
           HDBSCAN → c-TF-IDF)
        3. **Detects drift** between consecutive models (prevalence shifts,
           centroid drift, keyword divergence)
        4. **Enables human oversight** — domain experts can merge or relabel
           topics through an interactive UI (Human-in-the-Loop)
        5. **Tracks experiments** via MLflow & Prefect orchestration
        6. **Serves results** through a FastAPI + Streamlit stack
        """)
    with c2:
        st.markdown("""
        ## 🔑 Key Benefits

        <div class="info-card">
            <h3>🚀 Real-Time Insights</h3>
            <p>New batches are processed incrementally — no need to retrain from scratch.</p>
        </div>
        <div class="info-card">
            <h3>🧑‍🔬 Human-in-the-Loop</h3>
            <p>Experts refine topics via merge & relabel with full audit trail.</p>
        </div>
        <div class="info-card">
            <h3>📉 Drift Detection</h3>
            <p>Automatic alerts when topics shift — centroid, prevalence, and JS divergence.</p>
        </div>
        <div class="info-card">
            <h3>📊 Model Versioning</h3>
            <p>Every model version is archived — compare, rollback, and reproduce.</p>
        </div>
        """, unsafe_allow_html=True)

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
with tab_arch:
    st.markdown("## System Architecture")
    st.markdown("""
    The system follows a **layered architecture** with clear separation of
    concerns.  Each layer communicates through well-defined interfaces.
    """)

    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     PRESENTATION LAYER                          │
    │                                                                  │
    │   Streamlit Dashboard  ◄──── REST API ────►  FastAPI Server     │
    │   (Port 8501)                                (Port 8000)         │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                  ORCHESTRATION / BUSINESS LOGIC                  │
    │                                                                  │
    │   Prefect Flows          ETL Tasks          Model Tasks         │
    │   ├─ data_ingestion      ├─ data_tasks      ├─ train_seed      │
    │   ├─ model_training      ├─ drift_tasks     ├─ batch_merge     │
    │   ├─ drift_detection                        ├─ save / load     │
    │   └─ complete_pipeline                      └─ metadata        │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                       CORE ML COMPONENTS                        │
    │                                                                  │
    │   Sentence-BERT ─► UMAP ─► HDBSCAN ─► c-TF-IDF ─► BERTopic   │
    │   (Embeddings)    (Dim Red) (Cluster) (Repr.)    (Wrapper)      │
    └──────────────────────────────────┬───────────────────────────────┘
                                       │
    ┌──────────────────────────────────┴───────────────────────────────┐
    │                        STORAGE LAYER                            │
    │                                                                  │
    │   Models (.pkl)    Metadata (JSON)    Tabular (CSV / Parquet)   │
    │   ├─ current/      ├─ topics.json     ├─ assignments.csv       │
    │   ├─ previous/     ├─ state.json      ├─ alerts.csv            │
    │   └─ archive/ts/   └─ model_meta      └─ audit_log.csv        │
    │                                                                  │
    │   MLflow (mlruns/)  ─  experiment tracking & metrics            │
    └──────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Component Descriptions")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="info-card">
            <h3>📥 Data Ingestion Flow</h3>
            <p>Reads raw CSV, filters by configurable time-window batches (e.g., hourly, daily),
            cleans text (URL/mention removal, emoji stripping, phone/version masking), normalizes
            unicode, and saves processed data as Parquet. Filters inbound (customer) tweets only.</p>
        </div>
        <div class="info-card">
            <h3>🤖 Model Training Flow</h3>
            <p><strong>Seed mode:</strong> fit_transform on first batch to establish base model.<br/>
            <strong>Online/incremental mode:</strong> train fresh model on new batch data only, then
            merge with cumulative base model via <code>merge_models()</code>. This preserves all
            historical topics and HITL edits while incorporating new discoveries.</p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="info-card">
            <h3>📉 Drift Detection Flow</h3>
            <p>Compares current vs. previous model after each batch using:<br/>
            • <strong>Prevalence change</strong> (TVD of topic distributions)<br/>
            • <strong>Centroid shift</strong> (cosine distance in embedding space)<br/>
            • <strong>JS divergence</strong> (Jensen-Shannon on keyword distributions)<br/>
            • <strong>New / disappeared topics</strong> (excluding outlier topic -1)<br/>
            Alerts are stored in CSV with severity levels and JSON metrics for analysis.</p>
        </div>
        <div class="info-card">
            <h3>🧑‍🔬 HITL Module</h3>
            <p>Experts merge similar topics or relabel them directly in BERTopic model.
            Every action triggers model re-save and creates:
            • Archived version (timestamped .pkl)<br/>
            • Audit log entry (CSV with old/new topics, user note, timestamp)<br/>
            Supports full version history and rollback capability.</p>
        </div>
        """, unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
with tab_data:
    _dataset_descriptions = {
        "twitter": (
            "## Dataset — Twitter Customer Support (TwCS)",
            """The [TwCS dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
    contains **~3 million tweets** exchanged between customers and support
    agents across major brands on Twitter/X.""",
        ),
        "ag_news": (
            "## Dataset — AG News",
            """The [AG News dataset](https://huggingface.co/datasets/ag_news)
    contains **~127 k news articles** across 4 categories:
    World, Sports, Business, and Sci/Tech.""",
        ),
    }
    _h, _d = _dataset_descriptions.get(_ds_name, (f"## Dataset — {_ds_title}", ""))
    st.markdown(_h)
    st.markdown(_d)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📨</div>
            <div class="metric-value">~3M</div>
            <div class="metric-label">Total Tweets</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🏢</div>
            <div class="metric-value">108</div>
            <div class="metric-label">Brands Covered</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📅</div>
            <div class="metric-value">2017</div>
            <div class="metric-label">Collection Year</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Key Columns")

    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | `tweet_id` | Unique identifier for each tweet |
    | `author_id` | Anonymized user who sent the tweet |
    | `inbound` | `True` = customer message, `False` = agent reply |
    | `created_at` | Timestamp (used for time-windowed batching) |
    | `text` | Raw tweet body |
    | `response_tweet_id` | ID of the reply (links conversations) |
    | `in_response_to_tweet_id` | Parent tweet ID |
    """)

    st.markdown("### Preprocessing Pipeline")
    st.markdown("""
    1. **Filter** inbound tweets only (customer messages where `inbound = True`)
    2. **Sort** by `created_at` timestamp for chronological time-window batching
    3. **Clean text** using comprehensive pipeline:
       - Remove URLs, @mentions, hashtag symbols
       - Strip emojis and special characters
       - Mask phone numbers (XXX-XXX-XXXX) and version strings (v1.2.3)
       - Normalize unicode characters (NFD decomposition)
       - Remove repeated punctuation and extra whitespace
    4. **Drop** empty or near-empty documents (< 3 characters after cleaning)
    5. **Save** as partitioned Parquet for fast I/O in batch processing
    6. **State tracking** via `processing_state.json` for resumable processing
    """)

# ── METHODOLOGY ───────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("## Technical Methodology")

    st.markdown("### BERTopic Pipeline")
    st.markdown("""
    BERTopic combines four modular steps:

    | Step | Component | Purpose |
    |------|-----------|---------|
    | 1 | **Sentence-BERT** (`all-MiniLM-L6-v2`) | Encode each document into a 384-dim dense vector |
    | 2 | **UMAP** (5 components, cosine metric) | Non-linear dimensionality reduction preserving local structure |
    | 3 | **HDBSCAN** (density-based clustering) | Discover clusters of arbitrary shape — outliers mapped to topic -1 |
    | 4 | **c-TF-IDF** (class-based TF-IDF) | Extract representative keywords per cluster → human-readable topics |
    """)

    st.markdown("### Online / Incremental Learning Strategy")
    st.markdown("""
    Rather than re-processing the entire corpus each time:

    1. **New batch arrives** → train a *fresh* BERTopic model on new data only
    2. **Merge** the new batch model with the existing *cumulative* base model using
       `BERTopic.merge_models()` with `min_similarity` threshold
    3. The merged model **accumulates**:
       - All historical topics from previous batches
       - New topics discovered in the current batch
       - All HITL edits (topic merges and custom labels)
    4. Previous model is archived with timestamp for:
       - Drift comparison and alerting
       - Version history and rollback capability
    5. Topic -1 (outliers) are consistently excluded from metrics and counts

    This approach preserves human expertise while enabling continuous learning.
    """)

    st.markdown("### Drift Detection Metrics")
    st.markdown("""
    After each batch we compare **current** vs **previous** model:

    | Metric | Formula / Idea | Threshold (Alert Trigger) |
    |--------|---------------|---------------------------|
    | **Prevalence Change** | Total Variation Distance between topic distributions | 0.25 (High: >0.30, Med: >0.15, Low: >0.05) |
    | **Centroid Shift** | 1 − cosine_similarity(centroid_curr, centroid_prev) | 0.55 (High: >0.40, Med: >0.25, Low: >0.10) |
    | **JS Divergence** | Jensen-Shannon divergence on keyword weight distributions | 0.40 (High: >0.50, Med: >0.30, Low: >0.10) |
    | **New Topics** | Topics in current but not in previous (excluding outlier -1) | >10 new topics |
    | **Disappeared Topics** | Topics in previous but not in current | >6 disappeared topics |

    Alerts are generated at **high / medium / low** severity levels based on configurable thresholds.
    All thresholds are defined in `config/drift_thresholds.yaml` and can be tuned based on your data.
    """)

    st.markdown("### Human-in-the-Loop Workflow")
    st.markdown("""
    ```
    Expert reviews topics in Dashboard
        │
        ├── Merge similar topics ─── BERTopic.merge_topics(docs, topics_to_merge)
        │                                └─ Model is re-saved, previous version archived
        │
        └── Relabel topics ────────── BERTopic.set_topic_labels({id: label})
                                          └─ Metadata + model updated
    ```
    Every action is logged to `hitl_audit_log.csv` with timestamp, old/new
    topics, and optional user note.
    """)

# ── EVALUATION ────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## Evaluation Framework")
    st.markdown("""
    Topic model quality is assessed using both **intrinsic** and
    **extrinsic** measures.
    """)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("""
        <div class="info-card">
            <h3>Intrinsic Metrics</h3>
            <p>
            <strong>Topic Coherence (C_v)</strong> — measures semantic
            similarity among top keywords using sliding window and
            normalized pointwise mutual information.<br/><br/>
            <strong>Topic Diversity</strong> — fraction of unique words
            across all topic representations (higher = less redundancy).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class="info-card">
            <h3>Extrinsic / Operational Metrics</h3>
            <p>
            <strong>Silhouette Score</strong> — cluster separation in
            embedding space (−1 to 1, higher is better).<br/><br/>
            <strong>Outlier Ratio</strong> — fraction of documents assigned
            to topic −1 (lower indicates better coverage).<br/><br/>
            <strong>Drift Scores</strong> — prevalence TVD, centroid shift,
            JS divergence tracked per batch.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ### Planned Comparative Analysis

    | Model | Embedding | Clustering | Representation |
    |-------|-----------|------------|----------------|
    | **BERTopic** (ours) | Sentence-BERT | HDBSCAN | c-TF-IDF |
    | **LDA (Gensim)** | BoW | Dirichlet prior | Word distributions |
    | **NMF** | TF-IDF | Non-negative factorization | Weight vectors |

    The *Model Benchmarking* page allows side-by-side comparison of
    coherence, diversity, and silhouette across these approaches.
    """)

# ── ABOUT ─────────────────────────────────────────────────────────────────────
with tab_team:
    st.markdown("## About This Project")
    st.markdown("""
    <div class="info-card">
        <h3>🎓 Academic Context</h3>
        <p>
        <strong>Course:</strong> COMP-EE 798 — Final Year Project<br/>
        <strong>Title:</strong> Online BERTopic with Human-in-the-Loop for
        Customer Support Insights<br/>
        <strong>Year:</strong> 2025 / 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Tech Stack")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        **ML / NLP**
        - BERTopic
        - Sentence-Transformers
        - UMAP / HDBSCAN
        - scikit-learn
        """)
    with t2:
        st.markdown("""
        **Backend**
        - FastAPI
        - Prefect (orchestration)
        - MLflow (tracking)
        - Ollama (LLM labels)
        """)
    with t3:
        st.markdown("""
        **Frontend**
        - Streamlit
        - Plotly
        - Custom CSS theme
        """)
    with t4:
        st.markdown("""
        **Infrastructure**
        - Docker Compose
        - YAML config
        - CSV / Parquet / JSON
        - Git version control
        """)

    st.markdown("""
    ### Repository

    🔗 [github.com/krishnagajera45/Online-BERTopic-with-Human-in-the-Loop-for-Customer-Support-Insights](
    https://github.com/krishnagajera45/Online-BERTopic-with-Human-in-the-Loop-for-Customer-Support-Insights)
    """)

# ── THREE-MODEL COMPARISON ────────────────────────────────────────────────────
with tab_compare:
    st.markdown("## ⚖️ BERTopic vs LDA vs NMF — Complete Implementation Comparison")
    st.caption("""
    This section explains **exactly how** each model is implemented in our codebase, based on:  
    • `src/etl/tasks/model_tasks.py` (BERTopic)  
    • `src/etl/tasks/lda_tasks.py` (LDA)  
    • `src/etl/tasks/nmf_tasks.py` (NMF)
    """)
    
    st.markdown("""
    ### 🎯 Why Three Models?
    
    We compare BERTopic against **two distinct baseline paradigms** to rigorously validate its advantages:
    
    | Model | Paradigm | Why Include? |
    |-------|----------|--------------|
    | **BERTopic** (Primary) | Neural embeddings + density clustering | State-of-art for short, noisy text with contextual understanding |
    | **LDA** (Baseline 1) | Probabilistic generative (Dirichlet) | Classical gold standard — widely used benchmark in NLP research |
    | **NMF** (Baseline 2) | Algebraic matrix factorization | Often outperforms LDA on short text; tests if BERTopic beats TF-IDF methods |
    
    **Research Question:** Does BERTopic's neural approach justify the computational cost vs simpler TF-IDF/BoW methods?
    """)

    st.divider()

    # ── Paradigm Overview ─────────────────────────────────────────────────────
    st.markdown("### 🧠 Core Paradigms — Fundamentally Different Approaches")
    
    par1, par2, par3 = st.columns(3)
    with par1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — Neural Clustering</h3>
            <p>
            <strong>Paradigm:</strong> Representation learning + density clustering<br/><br/>
            <strong>Core idea:</strong><br/>
            1. Pre-trained transformer (BERT) maps text → semantic embeddings<br/>
            2. UMAP reduces dimensions while preserving local structure<br/>
            3. HDBSCAN finds dense regions (clusters) of any shape<br/>
            4. c-TF-IDF extracts representative words per cluster<br/><br/>
            <strong>Philosophy:</strong> Let the data's geometric structure dictate topics — no generative assumptions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with par2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA — Generative Bayesian</h3>
            <p>
            <strong>Paradigm:</strong> Probabilistic generative model<br/><br/>
            <strong>Core idea:</strong><br/>
            1. Assume: corpus generated from K hidden topics<br/>
            2. Each document = mixture of topics (Dirichlet prior)<br/>
            3. Each topic = distribution over words (Dirichlet prior)<br/>
            4. Variational Bayes infers latent structure<br/><br/>
            <strong>Philosophy:</strong> Model the generation process — invert it to find topics. Established in 2003 (Blei et al.).
            </p>
        </div>
        """, unsafe_allow_html=True)
    with par3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF — Matrix Factorization</h3>
            <p>
            <strong>Paradigm:</strong> Algebraic optimization<br/><br/>
            <strong>Core idea:</strong><br/>
            1. Document-term matrix V (TF-IDF weights)<br/>
            2. Factorize: V ≈ W × H<br/>
            &nbsp;&nbsp;• W = document-topic weights<br/>
            &nbsp;&nbsp;• H = topic-word weights<br/>
            3. Constrain W, H ≥ 0 (interpretable)<br/>
            4. Minimize reconstruction error via coordinate descent<br/><br/>
            <strong>Philosophy:</strong> Parts-based representation — documents are weighted sums of topics. Deterministic (unlike LDA's Bayesian sampling).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    st.markdown("### Step 1 — Text Preprocessing")
    st.caption("What happens to your text *before* it enters the model?")
    
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic — Minimal</h3>
            <p>
            <strong>Code:</strong> <code>initialize_bertopic_model_task()</code><br/><br/>
            <strong>Steps:</strong><br/>
            1. Remove URLs, @mentions, emojis<br/>
            2. Normalize unicode (NFD decomposition)<br/>
            3. Pass <em>full sentences</em> to Sentence-BERT<br/><br/>
            <strong>Why minimal?</strong><br/>
            BERT is pre-trained on billions of natural sentences — it understands context, grammar, negation natively. Stopwords like "not" carry critical semantic weight ("not happy" ≠ "happy").
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA — Heavy NLP</h3>
            <p>
            <strong>Code:</strong> <code>preprocess_documents_for_lda_task()</code><br/><br/>
            <strong>Steps:</strong><br/>
            1. <code>simple_preprocess(deacc=True, min_len=3)</code><br/>
            2. Remove NLTK stopwords<br/>
            3. Lemmatize (WordNetLemmatizer)<br/>
            4. Gensim Dictionary:<br/>
            &nbsp;&nbsp;• no_below=5, no_above=0.5<br/>
            &nbsp;&nbsp;• keep_n=10000<br/>
            5. Convert to Bag-of-Words<br/><br/>
            <strong>Why heavy?</strong> BoW only sees word counts — stopwords and inflections ("runs", "running") would dominate without filtering.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF — TF-IDF</h3>
            <p>
            <strong>Code:</strong> <code>preprocess_documents_for_nmf_task()</code><br/><br/>
            <strong>Dual pipeline:</strong><br/>
            <strong>A) For NMF training:</strong><br/>
            <code>TfidfVectorizer(max_features=5000,<br/>
            min_df=5, max_df=0.85,<br/>
            ngram_range=(1,1))</code><br/><br/>
            <strong>B) For coherence:</strong><br/>
            Same lemmatization as LDA → Gensim Dictionary (keep_n=10000)<br/><br/>
            <strong>Why TF-IDF?</strong> Downweights frequent words while preserving matrix structure. NMF needs non-negative inputs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 2: Document Representation ───────────────────────────────────────
    st.markdown("### Step 2 — Document Representation")
    st.caption("How is each document mathematically encoded?")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic<br/>384-dim Dense</h3>
            <p>
            <strong>Model:</strong> <code>all-MiniLM-L6-v2</code><br/><br/>
            <strong>Example:</strong> "App keeps crashing!"<br/>
            → <code>[0.23, -0.45, ..., 0.89]</code><br/><br/>
            <strong>Advantages:</strong><br/>
            • Semantic: "crash" ≈ "freeze"<br/>
            • Context: "not happy" ≠ "happy"<br/>
            • Robust on short text<br/>
            • Similar sentences → nearby points
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA<br/>BoW Sparse</h3>
            <p>
            <strong>Output:</strong> ~10k dims, mostly zeros<br/><br/>
            <strong>Example:</strong> "App keeps crashing!"<br/>
            After preprocessing:<br/>
            <code>['app', 'keep', 'crash']</code><br/>
            BoW: <code>{app:1, keep:1, crash:1, …rest:0}</code><br/><br/>
            <strong>Limitations:</strong><br/>
            • Order lost: "not happy" = "happy not"<br/>
            • No semantics: "crash" ≠ "freeze"<br/>
            • Very sparse
            </p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF<br/>TF-IDF Sparse</h3>
            <p>
            <strong>Output:</strong> 5000 dims, sparse<br/><br/>
            <strong>Example:</strong> "App keeps crashing!"<br/>
            TF-IDF: <code>{app:0.38, keep:0.29, crash:0.51, …rest:0}</code><br/><br/>
            <strong>vs BoW:</strong><br/>
            • TF-IDF downweights frequent words<br/>
            • Emphasizes distinctive terms<br/>
            • Still no semantics ("crash" ≠ "freeze")<br/>
            • Unigrams only (no bigrams)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 3: Topic Discovery ────────────────────────────────────────────────
    st.markdown("### Step 3 — Topic Discovery Alggorithm")
    st.caption("How are topics identified from document representations?")
    
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic<br/>UMAP → HDBSCAN</h3>
            <p>
            <strong>UMAP:</strong><br/>
            <code>n_neighbors=15, n_components=5,<br/>
            min_dist=0.0, metric='cosine'</code><br/>
            Reduces 384→5 dims<br/><br/>
            <strong>HDBSCAN:</strong><br/>
            <code>min_cluster_size=15,<br/>
            min_samples=5,<br/>
            metric='euclidean'</code><br/><br/>
            • Density-based<br/>
            • K auto-detected<br/>
            • Outliers → Topic -1<br/>
            • Hard assignment
            </p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA<br/>Dirichlet + VB</h3>
            <p>
            <strong>Config:</strong><br/>
            <code>LdaModel(corpus,<br/>
            num_topics=N, alpha='auto', eta='auto',<br/>
            passes=10, iterations=200)</code><br/><br/>
            <strong>Mechanics:</strong><br/>
            • Document = mixture of K topics<br/>
            • Topic = distribution over words<br/>
            • <strong>K fixed upfront</strong> (matched to BERTopic)<br/>
            • Soft assignment (probability per topic)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF<br/>V ≈ W × H</h3>
            <p>
            <strong>Config:</strong><br/>
            <code>NMF(n_components=N,<br/>
            init='nndsvda', solver='cd',<br/>
            max_iter=400,<br/>
            alpha_W=0.0, alpha_H=0.0)</code><br/><br/>
            <strong>Mechanics:</strong><br/>
            • Factorize TF-IDF matrix<br/>
            • Minimize ||V − W×H||²<br/>
            • W, H ≥ 0 (interpretable)<br/>
            • <strong>K fixed</strong> (matched to BERTopic)<br/>
            • Hard assignment (argmax W row)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 4: Topic Representation ──────────────────────────────────────────
    st.markdown("### Step 4 — Topic Representation & Labels")
    st.caption("How are topics described and labeled?")
    
    rep1, rep2, rep3 = st.columns(3)
    with rep1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic<br/>c-TF-IDF + LLM</h3>
            <p>
            <strong>CountVectorizer:</strong><br/>
            <code>min_df=5, max_df=0.95,<br/>
            ngram_range=(1,2)</code><br/>
            Includes bigrams!<br/><br/>
            <strong>c-TF-IDF:</strong><br/>
            Scores words by distinctiveness per cluster vs all others<br/><br/>
            <strong>LLM Labels:</strong><br/>
            Ollama (DeepSeek-R1:1.5b) reads keywords → generates human-readable topic name
            </p>
        </div>
        """, unsafe_allow_html=True)
    with rep2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA<br/>Top-N Words</h3>
            <p>
            <strong>Extraction:</strong><br/>
            Highest-probability words from Dirichlet distribution<br/><br/>
            <strong>Example:</strong><br/>
            Topic 3: <code>['flight', 'cancel', 'refund', 'book', 'ticket']</code><br/><br/>
            <strong>Limitation:</strong><br/>
            No automatic labels — experts must infer meaning. No bigrams.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with rep3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF<br/>H-Matrix Top-N</h3>
            <p>
            <strong>Extraction:</strong><br/>
            Highest-weight words from H matrix (topic-word weights)<br/><br/>
            <strong>Example:</strong><br/>
            Topic 3: <code>['flight', 'delay', 'airport', 'hour', 'gate']</code><br/><br/>
            <strong>Limitation:</strong><br/>
            No automatic labels. Unigrams only (ngram_range=(1,1)) to match Gensim coherence dictionary.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Step 5: Online Learning ───────────────────────────────────────────────
    st.markdown("### Step 5 — Online/Incremental Learning Strategy")
    st.caption("How does each model handle new batches over time?")
    
    ol1, ol2, ol3 = st.columns(3)
    with ol1:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #6C5CE7;">
            <h3>🤖 BERTopic<br/>merge_models()</h3>
            <p>
            <strong>Strategy:</strong><br/>
            1. Train fresh model on new batch only<br/>
            2. <code>base.merge_models([new_batch],<br/>
            &nbsp;&nbsp;min_similarity=...)</code><br/>
            3. Archive previous version<br/><br/>
            <strong>Preserved:</strong><br/>
            • All historical topics<br/>
            • HITL edits (merged topics, labels)<br/>
            • New discoveries<br/><br/>
            <strong>Advantage:</strong> Only new data processed per run
            </p>
        </div>
        """, unsafe_allow_html=True)
    with ol2:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #00B894;">
            <h3>📊 LDA<br/>Full Retrain</h3>
            <p>
            <strong></strong><br/>
            No native online learning for full model structure<br/><br/>
            <strong>Strategy:</strong><br/>
            Retrain on <em>entire</em> cumulative corpus from scratch:<br/>
            <code>LdaModel(full_cumulative_corpus,<br/>
            num_topics=N, passes=10, ...)</code><br/><br/>
            <strong>Drawback:</strong> Training time grows with each batch — all historical docs reprocessed
            </p>
        </div>
        """, unsafe_allow_html=True)
    with ol3:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #E17055;">
            <h3>🔢 NMF<br/>Full Retrain</h3>
            <p>
            <strong>Strategy:</strong><br/>
            Retrain on <em>entire</em> cumulative corpus:<br/>
            <code>NMF(n_components=N).fit(cumulative_tfidf)</code><br/><br/>
            <strong>Note:</strong> Faster than LDA (deterministic coordinate descent vs probabilistic sampling)<br/><br/>
            <strong>Drawback:</strong> Still O(corpus size) — no incremental accumulation
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Master Comparison Table ──────────────────────────────────────────────
    st.markdown("### 📊 Master Comparison Table")
    st.markdown("""
| **Parameter** | **BERTopic** | **LDA** | **NMF** |
|---------------|--------------|---------|---------|
| **Library** | `bertopic.BERTopic` | `gensim.models.LdaModel` | `sklearn.decomposition.NMF` |
| **Paradigm** | Neural embeddings + clustering | Probabilistic generative | Matrix factorization |
| **Embedding** | Sentence-BERT (384 dims, dense) | Bag-of-Words (~10k dims, sparse) | TF-IDF (5k dims, sparse) |
| **Preprocessing** | Minimal: URL/mention/emoji removal | Heavy: tokenize→stopwords→lemmatize→BoW | TF-IDF + lemmatization for coherence |
| **Vocabulary filter** | min_df=5, max_df=0.95 (CountVectorizer) | no_below=5, no_above=0.5, keep_n=10000 | min_df=5, max_df=0.85, max_features=5000 |
| **Dim reduction** | UMAP: n_neighbors=15, n_components=5, metric=cosine | None | None (TF-IDF already dimensioned) |
| **Clustering** | HDBSCAN: min_cluster_size=15, min_samples=5 | Dirichlet (Variational Bayes) | Coordinate descent (||V−W×H||²) |
| **Ngrams** | Unigrams + bigrams (1, 2) | Unigrams only | Unigrams only (for coherence compatibility) |
| **Topic count (K)** | Auto-detected by HDBSCAN | Fixed (matched to BERTopic) | Fixed (matched to BERTopic) |
| **Assignment type** | Hard (1 topic per doc) | Soft (probability per topic) | Hard (argmax W row) |
| **Outlier handling** | Topic -1 excluded from metrics | None (all docs assigned) | Zero-norm TF-IDF rows filtered |
| **Topic labels** | c-TF-IDF keywords + Ollama LLM labels | Top-N weighted words only | H-matrix top-N words only |
| **Training passes** | Single forward pass (fit_transform) | passes=10, iterations=200, chunksize=100 | max_iter=400, init='nndsvda', solver='cd' |
| **Online strategy** | merge_models() — new batch only | Full corpus retrain | Full corpus retrain (faster than LDA) |
| **HITL support** | ✅ merge_topics(), set_topic_labels(), archive | ❌ None | ❌ None |
| **Regularization** | None (HDBSCAN density-based) | alpha='auto', eta='auto' (Dirichlet prior) | alpha_W=0.0, alpha_H=0.0 (disabled for short text) |
| **Coherence metric** | Gensim C_v | Gensim C_v | Gensim C_v (same formula across all 3) |
| **Diversity metric** | unique_words / (top_n × K) | unique_words / (top_n × K) | unique_words / (top_n × K) |
| **Silhouette metric** | UMAP 5-dim space (cosine) | TF-IDF space (cosine) | TF-IDF space (cosine) |
    """)

    st.divider()

    # ── Key Takeaways ─────────────────────────────────────────────────────────
    st.markdown("### 🎯 Key Takeaways")
    
    take1, take2 = st.columns(2)
    with take1:
        st.markdown("""
        <div class="info-card">
            <h3>✅ Strengths by Model</h3>
            <p>
            <strong>BERTopic:</strong><br/>
            • Best for short, noisy text (tweets)<br/>
            • Semantic understanding → better coherence<br/>
            • Auto-detects topic count<br/>
            • True online learning via merge_models()<br/>
            • HITL support for human refinement<br/><br/>
            <strong>LDA:</strong><br/>
            • Established benchmark (Blei 2003)<br/>
            • Probabilistic reasoning<br/>
            • Soft topic assignments<br/><br/>
            <strong>NMF:</strong><br/>
            • Fast, deterministic<br/>
            • Often beats LDA on short text<br/>
            • Interpretable parts-based decomposition
            </p>
        </div>
        """, unsafe_allow_html=True)
    with take2:
        st.markdown("""
        <div class="info-card">
            <h3>⚠️ Limitations by Model</h3>
            <p>
            <strong>BERTopic:</strong><br/>
            • Computationally expensive (Sentence-BERT)<br/>
            • Requires pre-trained embeddings<br/>
            • Hard assignments (no mixed-topic docs)<br/><br/>
            <strong>LDA:</strong><br/>
            • BoW loses word order & semantics<br/>
            • K must be fixed upfront<br/>
            • Poor on short text (sparse BoW)<br/>
            • No online learning<br/><br/>
            <strong>NMF:</strong><br/>
            • No semantics (TF-IDF limitations)<br/>
            • K must be fixed upfront<br/>
            • No online learning
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    ### 📈 Evaluation Results
    
    See the **Model Benchmarking** page for live temporal comparison of **Coherence (C_v)**, **Diversity**, and **Silhouette Score** across all three models on your active dataset.
    
    **Note on metric comparisons:**
    - **Coherence (C_v)**: Directly comparable — all three use Gensim's C_v with same parameters
    - **Diversity**: NOT directly comparable — BERTopic's c-TF-IDF penalizes overlap more than LDA's Dirichlet, NMF is intermediate
    - **Silhouette**: NOT directly comparable — computed in different feature spaces (UMAP vs TF-IDF)
    """)

render_footer()
