"""Drift alerts API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List
from src.api.models.responses import AlertResponse
from src.utils import StorageManager
from src.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("", response_model=List[AlertResponse])
async def get_alerts(limit: int = 50):
    """Get recent drift alerts."""
    try:
        storage = StorageManager()
        logger.debug(f"Fetching alerts with limit={limit}")
        alerts_df = storage.load_drift_alerts(limit=limit)
        
        if len(alerts_df) == 0:
            logger.info("No alerts found")
            return []
        
        logger.info(f"Retrieved {len(alerts_df)} alerts")
        return alerts_df.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=AlertResponse)
async def get_latest_alert():
    """Get the most recent drift alert."""
    try:
        storage = StorageManager()
        logger.debug("Fetching latest alert")
        alerts_df = storage.load_drift_alerts(limit=1)
        
        if len(alerts_df) == 0:
            logger.info("No alerts found")
            raise HTTPException(status_code=404, detail="No alerts found")
        
        alert = alerts_df.iloc[0].to_dict()
        logger.info(f"Latest alert: {alert.get('alert_type', 'N/A')} - {alert.get('message', 'N/A')[:50]}")
        return alert
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest alert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

