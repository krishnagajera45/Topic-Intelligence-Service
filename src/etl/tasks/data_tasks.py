"""Prefect tasks for data processing (with extracted ETL logic)."""
from prefect import task, get_run_logger
from pathlib import Path
import pandas as pd
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import clean_for_dataset
from src.utils import load_config


@task(name="load-data-window", retries=2, retry_delay_seconds=10)
def load_data_window_task(
    csv_path: str,
    start_date: str = None,
    end_date: str = None,
    timestamp_column: str = None,
) -> pd.DataFrame:
    """
    Load data window from preprocessed CSV.
    
    Args:
        csv_path: Path to CSV file
        start_date: Start date for filtering
        end_date: End date for filtering
        timestamp_column: Name of the timestamp column (reads from config if None)
        
    Returns:
        DataFrame with loaded data
    """
    logger = get_run_logger()
    logger.info(f"Loading data window from {csv_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    df = load_dataset(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        timestamp_column=timestamp_column,
    )
    
    logger.info(f"Loaded {len(df)} rows")
    return df


@task(name="clean-text-column", retries=2)
def clean_text_column_task(
    df: pd.DataFrame,
    text_column: str = 'text',
    min_tokens: int = 5,
    clean_mode: str = 'general',
) -> pd.DataFrame:
    """
    Clean text column using the appropriate preset for the active dataset.
    
    Args:
        df: DataFrame with text data
        text_column: Name of text column
        min_tokens: Minimum tokens required per document
        clean_mode: Cleaning preset ('twitter' or 'general')
        
    Returns:
        DataFrame with cleaned text column
    """
    logger = get_run_logger()
    logger.info(f"Cleaning text in column '{text_column}' (mode={clean_mode})")
    
    # Create cleaned text column using the dataset-aware cleaner
    df['text_cleaned'] = df[text_column].fillna('').apply(
        lambda t: clean_for_dataset(t, mode=clean_mode)
    )
    
    # Remove empty texts
    original_len = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    logger.info(f"Removed {original_len - len(df)} empty texts")
    
    # Drop ultra-short documents (token count threshold)
    token_counts = df['text_cleaned'].str.split().str.len()
    original_len = len(df)
    df = df[token_counts >= min_tokens]
    logger.info(f"Removed {original_len - len(df)} short texts (< {min_tokens} tokens)")
    
    # Deduplicate on cleaned text
    original_len = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'])
    logger.info(f"Removed {original_len - len(df)} duplicate cleaned texts")
    
    return df


@task(name="add-document-ids", retries=1)
def add_document_ids_task(
    df: pd.DataFrame,
    id_column: str = None,
    id_prefix: str = None,
) -> pd.DataFrame:
    """
    Add unique document IDs to DataFrame.
    
    Uses the active dataset profile from config unless explicit overrides
    are given.
    
    Args:
        df: DataFrame to add IDs to
        id_column: Column to use as source ID (reads from config if None)
        id_prefix: Prefix for doc_id values (reads from config if None)
        
    Returns:
        DataFrame with doc_id column
    """
    logger = get_run_logger()
    logger.info("Adding document IDs")
    
    if id_column is None or id_prefix is None:
        config = load_config()
        id_column = id_column or config.dataset.id_column
        id_prefix = id_prefix or config.dataset.id_prefix
    
    if id_column in df.columns:
        df['doc_id'] = id_prefix + '_' + df[id_column].astype(str)
    else:
        # Fallback: use DataFrame index
        df['doc_id'] = id_prefix + '_' + df.index.astype(str)
        logger.warning(f"Column '{id_column}' not found; used DataFrame index for doc_id")
    
    sample = df['doc_id'].iloc[0] if len(df) > 0 else '<empty>'
    logger.info(f"Added {len(df)} document IDs (sample: {sample})")
    return df


@task(name="save-to-parquet", retries=1)
def save_to_parquet_task(df: pd.DataFrame, output_path: str) -> str:
    """
    Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        output_path: Path to output Parquet file
        
    Returns:
        Output path
    """
    logger = get_run_logger()
    logger.info(f"Saving {len(df)} rows to {output_path}")
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved to {output_path}")
    return output_path


@task(name="validate-data", retries=1)
def validate_data_task(df: pd.DataFrame, min_docs: int = 10) -> bool:
    """
    Validate that data meets minimum requirements.
    Valiadtion checks:
        - Minimum number of documents (min_docs)
        - Required columns: doc_id, text_cleaned, created_at
    Args:
        df: DataFrame to validate
        min_docs: Minimum number of documents required
        
    Returns:
        True if valid, raises error otherwise
    """
    logger = get_run_logger()
    
    if len(df) < min_docs:
        raise ValueError(f"Insufficient data: {len(df)} < {min_docs} documents")
    
    config = load_config()
    ts_col = config.dataset.timestamp_column
    required_cols = ['doc_id', 'text_cleaned', ts_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Data validation passed: {len(df)} documents")
    return True
