"""Data utility functions for loading and processing data.

NOTE: New code should use ``src.data.dataset_loader.load_dataset`` instead.
This module is kept for backward compatibility.
"""
import pandas as pd
from typing import Optional
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__, "logs/data_utils.log")


def load_twcs_data(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Backward-compatible wrapper — delegates to the unified loader."""
    from src.data.dataset_loader import load_dataset as _load
    return _load(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        timestamp_column="created_at",
    )


def load_processed_data(parquet_path: str) -> pd.DataFrame:
    """
    Load processed data from Parquet file.
    
    Args:
        parquet_path: Path to processed Parquet file
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading processed data from {parquet_path}")
    return pd.read_parquet(parquet_path)
