"""Batch statistics API endpoint — per-batch and cumulative aggregates."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from src.utils import StorageManager, setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("")
async def get_batch_stats() -> Dict[str, Any]:
    """
    Return per-batch aggregate stats plus cumulative totals.

    Response shape::

        {
          "cumulative": {
            "total_docs": 539,
            "total_batches": 3,
            "last_run": "2026-02-07T19:38:37",
          },
          "batches": [
            {
              "batch_id": "batch_20171001_0000_to_20171001_0005",
              "docs": 259,
              "topics": 37,
              "timestamp": "2017-10-01 00:00:00",
              "window_start": "2017-10-01 00:00:00",
              "window_end": "2017-10-01 00:05:00",
            },
            ...
          ]
        }
    """
    try:
        storage = StorageManager()
        assignments = storage.load_doc_assignments()

        if assignments.empty:
            return {
                "cumulative": {
                    "total_docs": 0,
                    "total_batches": 0,
                    "last_run": None,
                },
                "batches": [],
            }

        # Load topics metadata to get accurate topic counts per batch (excluding outlier -1)
        topics_list = storage.load_topics_metadata()  # Returns list of topic dicts
        
        # Count topics per batch from topics metadata (exclude outlier topic -1)
        topic_counts_by_batch = {}
        for topic in topics_list:
            topic_id = topic.get("topic_id")
            batch_id = topic.get("batch_id")
            # Exclude outlier topic -1
            if batch_id and topic_id != -1:
                topic_counts_by_batch[batch_id] = topic_counts_by_batch.get(batch_id, 0) + 1

        # Per-batch doc counts from assignments
        batch_agg = (
            assignments.groupby("batch_id")
            .agg(
                docs=("doc_id", "count"),
            )
            .reset_index()
        )
        
        # Add topic counts from metadata (not from assignments)
        batch_agg["topics"] = batch_agg["batch_id"].map(topic_counts_by_batch).fillna(0).astype(int)

        # Timestamp per batch (first occurrence)
        if "timestamp" in assignments.columns:
            ts = assignments.groupby("batch_id")["timestamp"].first().reset_index()
            batch_agg = batch_agg.merge(ts, on="batch_id", how="left")
        else:
            batch_agg["timestamp"] = None

        # Derive window_start / window_end from batch_id pattern
        # e.g. batch_20171001_0000_to_20171001_0005
        def parse_window(bid: str):
            try:
                parts = bid.split("_to_")
                prefix = parts[0].replace("batch_", "")  # "20171001_0000"
                suffix = parts[1] if len(parts) > 1 else prefix  # "20171001_0005"
                
                # prefix format: YYYYMMDD_HHMM
                if len(prefix) >= 13:
                    date_part = prefix[:8]  # "20171001"
                    time_part = prefix[9:13]  # "0000"
                    ws = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:00"
                else:
                    ws = None
                    
                if len(suffix) >= 13:
                    date_part = suffix[:8]  # "20171001"
                    time_part = suffix[9:13]  # "0005"
                    we = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:00"
                else:
                    we = None
                    
                return ws, we
            except Exception:
                return None, None

        batch_agg[["window_start", "window_end"]] = batch_agg["batch_id"].apply(
            lambda b: parse_window(b)
        ).apply(lambda x: list(x)).tolist()

        # Sort batches chronologically
        batch_agg = batch_agg.sort_values("timestamp").reset_index(drop=True)

        # Cumulative stats (total_topics removed per user request - causes statistics mismatch)
        state = storage.load_processing_state()
        cumulative = {
            "total_docs": int(assignments.shape[0]),
            "total_batches": int(batch_agg.shape[0]),
            "last_run": state.get("last_run_timestamp"),
        }

        # Build list
        batches_list = []
        for _, row in batch_agg.iterrows():
            batches_list.append({
                "batch_id": row["batch_id"],
                "docs": int(row["docs"]),
                "topics": int(row["topics"]),
                "timestamp": row.get("timestamp"),
                "window_start": row.get("window_start"),
                "window_end": row.get("window_end"),
            })

        return {"cumulative": cumulative, "batches": batches_list}

    except Exception as e:
        logger.error(f"Error computing batch stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
