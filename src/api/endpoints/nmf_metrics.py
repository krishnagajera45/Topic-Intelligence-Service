"""API endpoints for NMF model metrics and three-way model comparison."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

from src.utils import setup_logger

logger = setup_logger(__name__, "logs/api.log")

router = APIRouter()


# ── Internal loaders ──────────────────────────────────────────────────────────

def _load_nmf_metrics() -> Dict[str, Any]:
    """Load NMF metrics from disk."""
    p = Path("outputs/metrics/nmf_metrics.json")
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_lda_metrics() -> Dict[str, Any]:
    p = Path("outputs/metrics/lda_metrics.json")
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_bertopic_metrics() -> Dict[str, Any]:
    p = Path("outputs/metrics/bertopic_metrics.json")
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ── Single-model endpoints ────────────────────────────────────────────────────

@router.get("/", response_model=Dict[str, Any])
async def get_nmf_metrics():
    """
    Get the latest NMF evaluation metrics.

    Returns coherence_c_v, diversity, silhouette_score, num_topics,
    training_time_seconds, and topics list.
    """
    try:
        data = _load_nmf_metrics()

        if not data:
            return {
                "status": "not_available",
                "message": "NMF metrics not yet computed. Enable NMF in config and run the pipeline.",
                "coherence_c_v": 0.0,
                "diversity": 0.0,
                "silhouette_score": 0.0,
                "num_topics": 0,
                "training_time_seconds": 0.0,
            }

        metrics = data.get("latest", data)
        # Fallback: pull from latest batch entry if root lacks values
        if not metrics.get("coherence_c_v") and data.get("batches"):
            metrics = {**metrics, **data["batches"][-1]}

        if "topics" in data.get("latest", {}):
            metrics["topics"] = data["latest"]["topics"]
        elif "topics" in data:
            metrics["topics"] = data["topics"]

        logger.info(f"NMF metrics retrieved: {metrics.get('num_topics', 0)} topics")
        return metrics

    except Exception as exc:
        logger.error(f"Error retrieving NMF metrics: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve NMF metrics: {exc}")


@router.get("/history", response_model=Dict[str, Any])
async def get_nmf_metrics_history():
    """Get NMF per-batch metrics history for temporal analysis charts."""
    try:
        data = _load_nmf_metrics()
        batches = data.get("batches", [])
        # Legacy single-run format
        if not batches and data.get("batch_id"):
            batches = [{
                "batch_id": data.get("batch_id"),
                "coherence_c_v": data.get("coherence_c_v"),
                "diversity": data.get("diversity"),
                "silhouette_score": data.get("silhouette_score"),
                "num_topics": data.get("num_topics"),
                "timestamp": data.get("timestamp"),
                "training_time_seconds": data.get("training_time_seconds") or data.get("total_time_seconds"),
            }]
        return {"batches": batches, "status": "ok" if batches else "not_available"}

    except Exception as exc:
        logger.error(f"Error retrieving NMF history: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/topics", response_model=Dict[str, Any])
async def get_nmf_topics():
    """Get NMF topic details with top-words and weights."""
    try:
        data = _load_nmf_metrics()
        if not data:
            raise HTTPException(
                status_code=404,
                detail="NMF metrics not found. Enable NMF in config and run the pipeline.",
            )
        topics = data.get("latest", data).get("topics", data.get("topics", []))
        return {
            "num_topics": len(topics),
            "topics": topics,
            "batch_id": data.get("latest", data).get("batch_id"),
            "timestamp": data.get("latest", data).get("timestamp"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error retrieving NMF topics: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Three-way comparison endpoint ─────────────────────────────────────────────

@router.get("/comparison3", response_model=Dict[str, Any])
async def get_three_way_comparison():
    """
    Return a side-by-side comparison of all three models:
    BERTopic (primary system), LDA (baseline 1), NMF (baseline 2).

    All three share the same evaluation metrics (coherence C_v, diversity,
    silhouette score) computed on the same cumulative corpus after each batch.
    """
    try:
        lda_raw = _load_lda_metrics()
        lda = lda_raw.get("latest", lda_raw) if lda_raw else {}

        nmf_raw = _load_nmf_metrics()
        nmf = nmf_raw.get("latest", nmf_raw) if nmf_raw else {}

        bt_raw = _load_bertopic_metrics()
        bt = bt_raw.get("latest", bt_raw) if bt_raw else {}

        # Count total BERTopic documents from topics_metadata
        total_docs = 0
        tp = Path("outputs/topics/topics_metadata.json")
        if tp.exists():
            try:
                with open(tp) as f:
                    td = json.load(f)
                total_docs = sum(t.get("count", 0) for t in td.get("topics", []))
            except Exception:
                pass

        def _extract(d: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "coherence_c_v": d.get("coherence_c_v", 0.0),
                "diversity": d.get("diversity", 0.0),
                "silhouette_score": d.get("silhouette_score", 0.0),
                "num_topics": d.get("num_topics", 0),
                "training_time_seconds": d.get("training_time_seconds", 0.0),
                "status": d.get("status", "not_available"),
            }

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "bertopic": {**_extract(bt), "total_documents": total_docs},
            "lda":      _extract(lda),
            "nmf":      _extract(nmf),
        }

        logger.info(
            f"Three-way comparison: BERTopic={comparison['bertopic']['num_topics']} "
            f"LDA={comparison['lda']['num_topics']} "
            f"NMF={comparison['nmf']['num_topics']} topics"
        )
        return comparison

    except Exception as exc:
        logger.error(f"Error generating three-way comparison: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {exc}")
