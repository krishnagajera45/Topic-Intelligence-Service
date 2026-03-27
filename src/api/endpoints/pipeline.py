"""Pipeline status API endpoint."""
from fastapi import APIRouter, HTTPException
from src.api.models.responses import PipelineStatusResponse
from src.utils import StorageManager
from src.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """Get ETL pipeline status."""
    try:
        storage = StorageManager()
        state = storage.load_processing_state()
        
        return PipelineStatusResponse(
            last_run=state.get('last_run_timestamp'),
            last_batch_id=state.get('last_batch_id'),
            documents_processed=state.get('documents_processed'),
            status=state.get('status', 'unknown'),
            next_scheduled_run=None  # Could calculate from cron schedule
        )
    
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

