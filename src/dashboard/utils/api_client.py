"""API client for Streamlit dashboard to communicate with FastAPI backend."""
import requests
from typing import List, Dict, Any, Optional
from src.utils import load_config

config = load_config()


class APIClient:
    """Client for interacting with the FastAPI backend."""
    
    def __init__(self, base_url: str = None):
        """Initialize API client."""
        self.base_url = base_url or config.dashboard.api_base_url
    
    # ── Connectivity ──────────────────────────────────────────────
    def health_check(self) -> bool:
        """Return True if the API is reachable."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── Topics ────────────────────────────────────────────────────
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get current topics."""
        response = requests.get(f"{self.base_url}/api/v1/topics/current")
        response.raise_for_status()
        return response.json()
    
    def get_topic_details(self, topic_id: int) -> Dict[str, Any]:
        """Get details for a specific topic."""
        response = requests.get(f"{self.base_url}/api/v1/topics/{topic_id}")
        response.raise_for_status()
        return response.json()
    
    def get_topic_examples(self, topic_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get example documents for a topic."""
        response = requests.get(f"{self.base_url}/api/v1/topics/{topic_id}/examples?limit={limit}")
        response.raise_for_status()
        return response.json()
    
    # ── Trends ────────────────────────────────────────────────────
    def get_trends(self, topic_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get topic trends."""
        params = {}
        if topic_id is not None:
            params['topic_id'] = topic_id
        response = requests.get(f"{self.base_url}/api/v1/trends", params=params)
        response.raise_for_status()
        return response.json()
    
    # ── Alerts ────────────────────────────────────────────────────
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get drift alerts."""
        response = requests.get(f"{self.base_url}/api/v1/alerts?limit={limit}")
        response.raise_for_status()
        return response.json()
    
    # ── Inference ─────────────────────────────────────────────────
    def infer_topic(self, text: str) -> Dict[str, Any]:
        """Predict topic for text."""
        response = requests.post(
            f"{self.base_url}/api/v1/infer",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    # ── HITL ──────────────────────────────────────────────────────
    def merge_topics(self, topic_ids: List[int], new_label: str = None, note: str = None) -> Dict[str, Any]:
        """Merge topics."""
        response = requests.post(
            f"{self.base_url}/api/v1/hitl/merge",
            json={"topic_ids": topic_ids, "new_label": new_label, "note": note}
        )
        response.raise_for_status()
        return response.json()
    
    def relabel_topic(self, topic_id: int, new_label: str, note: str = None) -> Dict[str, Any]:
        """Relabel a topic."""
        response = requests.post(
            f"{self.base_url}/api/v1/hitl/relabel",
            json={"topic_id": topic_id, "new_label": new_label, "note": note}
        )
        response.raise_for_status()
        return response.json()
    
    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get HITL audit log."""
        response = requests.get(f"{self.base_url}/api/v1/hitl/audit?limit={limit}")
        response.raise_for_status()
        return response.json()
    
    def get_version_history(self) -> Dict[str, Any]:
        """Get model version history."""
        response = requests.get(f"{self.base_url}/api/v1/hitl/version-history")
        response.raise_for_status()
        return response.json()
    
    def rollback_model(self, version_timestamp: str) -> Dict[str, Any]:
        """Rollback model to a previous version."""
        response = requests.post(f"{self.base_url}/api/v1/hitl/rollback/{version_timestamp}")
        response.raise_for_status()
        return response.json()

    # ── Pipeline ──────────────────────────────────────────────────
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        response = requests.get(f"{self.base_url}/api/v1/pipeline/status")
        response.raise_for_status()
        return response.json()

    # ── Batch Stats ───────────────────────────────────────────────
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get per-batch + cumulative aggregate statistics."""
        response = requests.get(f"{self.base_url}/api/v1/batch-stats")
        response.raise_for_status()
        return response.json()

    # ── LDA Metrics ───────────────────────────────────────────────
    def get_lda_metrics(self) -> Dict[str, Any]:
        """Get LDA model metrics."""
        response = requests.get(f"{self.base_url}/api/v1/lda/")
        response.raise_for_status()
        return response.json()
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get BERTopic vs LDA comparison."""
        response = requests.get(f"{self.base_url}/api/v1/lda/comparison")
        response.raise_for_status()
        return response.json()
    
    # ── BERTopic Metrics ──────────────────────────────────────────
    def get_bertopic_metrics(self) -> Dict[str, Any]:
        """Get BERTopic model evaluation metrics."""
        response = requests.get(f"{self.base_url}/api/v1/bertopic/")
        response.raise_for_status()
        return response.json()
    
    def get_bertopic_metrics_history(self) -> Dict[str, Any]:
        """Get BERTopic metrics history for temporal charts."""
        response = requests.get(f"{self.base_url}/api/v1/bertopic/history")
        response.raise_for_status()
        return response.json()
    
    def get_lda_metrics_history(self) -> Dict[str, Any]:
        """Get LDA metrics history for temporal charts."""
        response = requests.get(f"{self.base_url}/api/v1/lda/history")
        response.raise_for_status()
        return response.json()

    # ── NMF Metrics ───────────────────────────────────────────────
    def get_nmf_metrics(self) -> Dict[str, Any]:
        """Get NMF model evaluation metrics."""
        response = requests.get(f"{self.base_url}/api/v1/nmf/")
        response.raise_for_status()
        return response.json()

    def get_nmf_metrics_history(self) -> Dict[str, Any]:
        """Get NMF metrics history for temporal charts."""
        response = requests.get(f"{self.base_url}/api/v1/nmf/history")
        response.raise_for_status()
        return response.json()

    def get_three_way_comparison(self) -> Dict[str, Any]:
        """Get BERTopic vs LDA vs NMF three-way comparison."""
        response = requests.get(f"{self.base_url}/api/v1/nmf/comparison3")
        response.raise_for_status()
        return response.json()

    # ── Internal helper ───────────────────────────────────────────
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Generic request method for flexibility."""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

