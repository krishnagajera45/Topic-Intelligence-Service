"""Trends API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from src.api.models.responses import TrendResponse
from src.utils import StorageManager
from src.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("", response_model=List[TrendResponse])
async def get_trends(
    topic_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get topic counts across time windows."""
    try:
        storage = StorageManager()
        logger.debug(f"Fetching trends: topic_id={topic_id}, start={start_date}, end={end_date}")
        assignments = storage.load_doc_assignments(topic_id=topic_id)
        
        if len(assignments) == 0:
            logger.info("No assignments found for trend calculation")
            return []
        
        # Group by batch_id and topic_id
        trends = assignments.groupby(['batch_id', 'topic_id']).size().reset_index(name='count')
        
        # Add timestamp if available
        if 'timestamp' in assignments.columns:
            batch_timestamps = assignments.groupby('batch_id')['timestamp'].first()
            trends = trends.merge(
                batch_timestamps.reset_index(),
                on='batch_id',
                how='left'
            )
        
        logger.info(f"Retrieved {len(trends)} trend data points")
        return trends.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

