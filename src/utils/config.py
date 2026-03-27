"""Configuration management for Topic Modeling system."""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    """Per-dataset configuration profile."""
    raw_csv_path: str
    id_column: str
    text_column: str
    timestamp_column: str
    id_prefix: str = "doc"
    default_start: str = "2017-10-01"
    clean_mode: str = "general"   # "twitter" | "general"
    min_tokens: int = 5
    sample_csv_path: str = ""


@dataclass
class DataConfig:
    """Data-related configuration."""
    processed_parquet_dir: str


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 15
    min_samples: int = 5
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_metric: str = "euclidean"
    min_df: int = 5
    max_df: float = 0.95
    ngram_range: list = field(default_factory=lambda: [1, 2])
    top_n_words: int = 10
    min_similarity: float = 0.7
    top_words_for_label: int = 3


@dataclass
class StorageConfig:
    """Storage paths configuration."""
    topics_metadata_path: str
    doc_assignments_path: str
    alerts_path: str
    audit_log_path: str
    current_model_path: str
    previous_model_path: str
    state_file: str
    metrics_dir: str = "outputs/metrics"


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "twcs_topic_modeling"


@dataclass
class OllamaConfig:
    """Ollama (local LLM) configuration."""
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "phi3:mini"
    temperature: float = 0.2
    max_tokens: int = 128
    timeout_seconds: int = 20
    examples_limit: int = 3
    prompt_template: str = (
        "You are labeling support-ticket topics.\n"
        "Given keywords and example texts, return JSON with keys: label, summary.\n"
        "Label: 3-6 words. Summary: one sentence.\n\n"
        "Keywords: {keywords}\n"
        "Examples:\n{examples}\n"
    )


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["http://localhost:8501"])


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    api_base_url: str = "http://localhost:8000"


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    batch_size: int = 5000
    window_minutes: int = 30
    schedule_cron: str = "*/30 * * * *"


@dataclass
class PrefectConfig:
    """Prefect orchestration configuration."""
    api_url: str = "http://127.0.0.1:4200/api"
    logging_level: str = "INFO"
    work_queue: str = "default"
    work_pool: str = "default-agent-pool"
    flow_run_name_template: str = "pipeline-{date}"
    task_retry_delay_seconds: int = 10
    max_retries: int = 2
    storage_path: str = "data/prefect"


@dataclass
class LDAConfig:
    """LDA model configuration for comparison."""
    enabled: bool = True
    num_topics: str = "auto"  # 'auto' or integer
    passes: int = 10
    iterations: int = 200
    top_n_words: int = 10


@dataclass
class NMFConfig:
    """NMF model configuration for comparison."""
    enabled: bool = True
    num_topics: str = "auto"
    max_iter: int = 400
    alpha_W: float = 0.1
    alpha_H: float = 0.1
    max_features: int = 5000
    top_n_words: int = 10


@dataclass
class Config:
    """Main configuration object."""
    active_dataset: str
    dataset: DatasetProfile
    data: DataConfig
    model: ModelConfig
    storage: StorageConfig
    mlflow: MLflowConfig
    ollama: OllamaConfig
    api: APIConfig
    dashboard: DashboardConfig
    scheduler: SchedulerConfig
    prefect: PrefectConfig
    lda: LDAConfig = None  # Optional LDA config
    nmf: NMFConfig = None  # Optional NMF config


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Resolve active dataset
    active = config_dict['active_dataset']
    datasets = config_dict.get('datasets', {})
    if active not in datasets:
        raise ValueError(
            f"active_dataset '{active}' not found in datasets. "
            f"Available: {list(datasets.keys())}"
        )
    ds_profile = DatasetProfile(**datasets[active])

    # Interpolate {dataset} placeholders in storage / mlflow / data paths
    def _interpolate(d: dict, key: str = "dataset", value: str = active) -> dict:
        out = {}
        for k, v in d.items():
            out[k] = v.replace(f"{{{key}}}", value) if isinstance(v, str) else v
        return out

    storage_dict = _interpolate(config_dict['storage'])
    mlflow_dict = _interpolate(config_dict['mlflow'])
    data_dict = _interpolate(config_dict['data'])

    return Config(
        active_dataset=active,
        dataset=ds_profile,
        data=DataConfig(**data_dict),
        model=ModelConfig(**config_dict['model']),
        storage=StorageConfig(**storage_dict),
        mlflow=MLflowConfig(**mlflow_dict),
        ollama=OllamaConfig(**config_dict.get('ollama', {})),
        api=APIConfig(**config_dict['api']),
        dashboard=DashboardConfig(**config_dict['dashboard']),
        scheduler=SchedulerConfig(**config_dict['scheduler']),
        prefect=PrefectConfig(**config_dict.get('prefect', {})),
        lda=LDAConfig(**config_dict.get('lda', {})) if 'lda' in config_dict else None,
        nmf=NMFConfig(**config_dict.get('nmf', {})) if 'nmf' in config_dict else None
    )


def load_drift_thresholds(config_path: str = "config/drift_thresholds.yaml") -> Dict[str, Any]:
    """Load drift detection thresholds."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

