"""Prefect flow for data ingestion and ETL (with granular task-level tracking)."""
from typing import Optional
from prefect import flow, get_run_logger
from datetime import datetime
from src.etl.tasks.data_tasks import (
    load_data_window_task,
    clean_text_column_task,
    add_document_ids_task,
    save_to_parquet_task,
    validate_data_task
)
from src.utils import load_config


@flow(name="data-ingestion-flow", log_prints=True)
def data_ingestion_flow(
    csv_path: str,
    output_parquet: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_docs: int = 10
):
    """
    Prefect flow for data ingestion and preprocessing with granular task tracking.
    
    This flow orchestrates granular data processing tasks:
    1. Load data window (task)
    2. Clean text (task)
    3. Add document IDs (task)
    4. Validate data (task)
    5. Save to Parquet (task)
    
    Args:
        csv_path: Path to input CSV
        output_parquet: Path to output Parquet
        start_date: Start date for filtering
        end_date: End date for filtering
        min_docs: Minimum documents required per batch (for validation) 
        
    Returns:
        Processed DataFrame
    """
    logger = get_run_logger()
    config = load_config()
    ds = config.dataset  # active dataset profile
    
    logger.info(f"Starting data ingestion flow (dataset={config.active_dataset})")
    logger.info(f"Input: {csv_path}")
    logger.info(f"Output: {output_parquet}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Step 1: Load data window (task)
    logger.info("Step 1: Loading data window")
    df = load_data_window_task(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        timestamp_column=ds.timestamp_column,
    )
    
    # Step 2: Clean text (task)
    logger.info("Step 2: Cleaning text")
    df = clean_text_column_task(
        df=df,
        text_column=ds.text_column,
        min_tokens=ds.min_tokens,
        clean_mode=ds.clean_mode,
    )
    
    # Step 3: Add document IDs (task)
    logger.info("Step 3: Adding document IDs")
    df = add_document_ids_task(
        df,
        id_column=ds.id_column,
        id_prefix=ds.id_prefix,
    )
    
    # Step 4: Validate data (task)
    logger.info("Step 4: Validating data")
    validate_data_task(df, min_docs=min_docs)
    
    # Step 5: Save to Parquet (task)
    logger.info("Step 5: Saving to Parquet")
    output_path = save_to_parquet_task(df, output_parquet)
    
    logger.info(f"Data ingestion flow completed: {len(df)} documents")
    logger.info(f"Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    config = load_config()
    df = data_ingestion_flow(
        csv_path=config.dataset.raw_csv_path,
        output_parquet="data/processed/test.parquet"
    )
    print(f"Processed {len(df)} documents")
