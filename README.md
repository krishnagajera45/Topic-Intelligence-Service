# Batch-Recurring Topic Modeling Platform (BERTopic + Drift + HITL)

Production-style topic intelligence platform for evolving text streams.

This project implements an end-to-end system that:
- ingests time-windowed text batches,
- trains/updates topics with BERTopic via **train-and-merge**,
- benchmarks against LDA and NMF,
- detects drift with multiple statistical signals,
- serves APIs via FastAPI,
- exposes analyst workflows (HITL relabel/merge/rollback) in Streamlit,
- tracks orchestration/experiments with Prefect and MLflow.

The implementation follows the architecture and orchestration strategy described in the report artifacts:
- `report/figures/Fig1.drawio.pdf`
- `report/figures/Fig2.drawio (3).pdf`
- `report/figures/fig1-layered-architecture.mmd`
- `report/figures/fig2-orchestration-flow.mmd`
- `report/figures/fig2-orchestration-merging.mmd`

## Why This Project Exists

Classic topic modeling workflows are static. Real systems are not.

This platform is designed for continuously arriving corpora (social media, support tickets, news feeds) where topics evolve over time and operations need:
- repeatable scheduled updates,
- model versioning and rollback,
- observable drift signals,
- human governance and auditability,
- side-by-side baseline comparisons for scientific rigor.

## Core Capabilities

- **Operational BERTopic**: sentence embeddings + UMAP + HDBSCAN + c-TF-IDF for semantic topic discovery.
- **Temporal train-and-merge**: each new batch trains a fresh BERTopic model and merges into the deployed model using `min_similarity`.
- **Multi-signal drift detection**:
  - prevalence shift (TVD),
  - centroid shift (cosine distance in embedding space),
  - keyword divergence (Jensen-Shannon),
  - topic birth/death events.
- **Human-in-the-loop controls**:
  - relabel topics,
  - merge topics,
  - inspect audit history,
  - rollback to archived model versions.
- **Comparative baselines**: LDA + NMF metrics computed on aligned cumulative corpus.
- **MLOps observability**: Prefect for orchestration, MLflow for run metrics/artifacts.

## Repository Layout

```text
.
├── src/
│   ├── api/                 # FastAPI app + endpoints
│   ├── dashboard/           # Streamlit multipage UI
│   ├── etl/                 # Prefect flows/tasks (ingest, model, drift)
│   ├── data/                # preprocessing and dataset loading helpers
│   └── utils/               # config, storage, logging, versioning, helpers
├── config/                  # config.yaml, drift thresholds, model config
├── docker/                  # Dockerfiles for backend/frontend
├── scripts/                 # Prefect startup/deploy/manual pipeline scripts
├── data/                    # raw/processed/state inputs and runtime files
├── models/                  # current/previous model artifacts per dataset
├── outputs/                 # topics, assignments, alerts, metrics, audit
├── report/                  # paper + diagrams
└── docker-compose.yml       # local container orchestration
```

## System Architecture (At a Glance)

1. **Trigger/windowing** (manual or cron) defines batch boundaries.
2. **Data pipeline** ingests, cleans, deduplicates, persists batch artifacts.
3. **Modeling** executes BERTopic train-and-merge (or seed on first run).
4. **Baselines** (optional) compute LDA/NMF metrics on cumulative corpus.
5. **Drift monitoring** computes alerts against previous deployed model.
6. **Persistence/handoff** publishes artifacts consumed by API + dashboard.
7. **HITL governance** lets analysts relabel/merge/rollback with audit logs.

See `docs/ARCHITECTURE.md` for detailed layer and dataflow documentation.

## Quick Start (Local Python)

### 1) Prerequisites

- Python 3.11 recommended
- `pip` (or `uv`)
- Optional but recommended:
  - [Prefect 3](https://docs.prefect.io/)
  - [MLflow](https://mlflow.org/)
  - [Ollama](https://ollama.com/) for LLM-assisted topic labels

### 2) Create environment and install dependencies

Using `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### 3) Configure dataset/profile

Edit `config/config.yaml`:
- set `active_dataset` (`twitter` or `ag_news`),
- verify the corresponding dataset profile paths/columns.

### 4) Start services

FastAPI:

```bash
python -m src.api.main
```

Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

Complete stack helper:

```bash
./run_full_system.sh
```

## Quick Start (Docker Compose)

The repository now includes runnable container builds for backend/frontend and optional ops services.

```bash
docker compose up --build
```

This starts:
- API on `http://localhost:8000`
- Dashboard on `http://localhost:8501`

To include MLflow + Prefect services:

```bash
docker compose --profile ops up --build
```

Detailed Docker workflow is documented in `docs/DOCKER.md`.

## Configuration Model

Primary config file: `config/config.yaml`

Important sections:
- `active_dataset` + `datasets`
- `model` (BERTopic/UMAP/HDBSCAN/vectorizer parameters)
- `storage` paths
- `scheduler` windowing/cron
- `lda` and `nmf` baseline toggles
- `ollama` labeling config

Environment overrides are supported for deployment scenarios:
- `API_HOST`, `API_PORT`
- `DASHBOARD_HOST`, `DASHBOARD_PORT`
- `API_BASE_URL` (or `DASHBOARD_API_BASE_URL`)
- `ACTIVE_DATASET`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

## Data and Runtime Artifacts

Expected inputs:
- `data/raw/twcs_cleaned.csv` for Twitter profile
- `data/raw/ag_news_cleaned.csv` for AG News profile

Generated outputs (dataset-scoped):
- `models/{dataset}/current/bertopic_model.pkl`
- `models/{dataset}/previous/bertopic_model.pkl`
- `outputs/{dataset}/topics/topics_metadata.json`
- `outputs/{dataset}/assignments/doc_assignments.csv`
- `outputs/{dataset}/alerts/drift_alerts.csv`
- `outputs/{dataset}/audit/hitl_audit_log.csv`
- `outputs/{dataset}/metrics/*.json`
- `data/state/{dataset}_processing_state.json`

## Operational Commands

Makefile helpers:

```bash
make install
make run-api
make run-dashboard
make run-pipeline
make run-all
make stop
```

Prefect helper scripts:

```bash
./scripts/start_prefect.sh
./scripts/deploy_flows.sh
./scripts/run_pipeline_manual.sh
./scripts/stop_prefect.sh
```

## API Surface

Base path: `/api/v1`

Main endpoint groups:
- `/topics` (current topics, topic detail, topic examples)
- `/trends` (topic counts over windows)
- `/alerts` (drift alerts)
- `/infer` (single-text topic inference)
- `/hitl` (merge/split/relabel/audit/version-history/rollback)
- `/pipeline/status` (latest processing state)
- `/batch-stats` (batch and cumulative stats)
- `/bertopic`, `/lda`, `/nmf` (evaluation metrics and comparisons)

Interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Developer Workflow

- Start from `docs/GETTING_STARTED.md`.
- Read architecture notes in `docs/ARCHITECTURE.md`.
- Use `docs/DOCKER.md` for containerized development.
- Follow contribution standards in `CONTRIBUTING.md`.

## Known Boundaries

- Topic split endpoint exists but full semantic split automation is still limited.
- End-to-end reproducibility depends on availability/quality of external datasets.
- LLM-assisted labels require a reachable local Ollama endpoint and model.

## Contribution

Contributions are welcome. Please read `CONTRIBUTING.md` before opening a PR.
