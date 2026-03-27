"""
Unified dataset loader.

Reads the active dataset profile from config and returns a standardised
DataFrame regardless of which dataset is selected.  Every row has at
minimum:  ``text``, ``created_at`` (tz-aware UTC), and the original ID column.
"""
import pandas as pd
from typing import Optional
from src.utils.logging_config import setup_logger
from src.utils.config import load_config

logger = setup_logger(__name__, "logs/dataset_loader.log")


def load_dataset(
    csv_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    config=None,
) -> pd.DataFrame:
    """
    Load any dataset CSV into a standardised DataFrame.

    The function reads the active dataset profile from config unless
    explicit overrides are given.

    Args:
        csv_path: Override CSV path (defaults to config profile's raw_csv_path)
        start_date: Start date filter (inclusive)
        end_date:   End date filter (inclusive)
        timestamp_column: Override timestamp column name
        config: Pre-loaded Config object (loads fresh if None)

    Returns:
        DataFrame with at least ``text`` and ``created_at`` columns.
    """
    if config is None:
        config = load_config()

    ds = config.dataset
    csv_path = csv_path or ds.raw_csv_path
    ts_col = timestamp_column or ds.timestamp_column

    logger.info(f"Loading dataset '{config.active_dataset}' from {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=[ts_col])
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Ensure timezone-aware UTC
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
        logger.info("Applied UTC timezone to timestamps")

    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df[ts_col] >= start_dt]
        logger.info(f"Filtered from {start_date}: {len(df)} rows remaining")

    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True)
        df = df[df[ts_col] <= end_dt]
        logger.info(f"Filtered until {end_date}: {len(df)} rows remaining")

    return df
