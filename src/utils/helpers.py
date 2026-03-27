"""Helper utilities for the TwCS Topic Modeling system."""
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any
import html
import unicodedata


def generate_batch_id(start_date: str, end_date: str) -> str:
    """
    Generate a unique batch ID from date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Batch ID string
    """
    start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    return f"batch_{start}_to_{end}"
import re


def generate_alert_id() -> str:
    """Generate a unique alert ID."""
    return f"alert_{uuid.uuid4().hex[:8]}"


def generate_batch_id(window_start: str, window_end: str = None) -> str:
    """
    Generate a batch ID from window dates.
    
    Args:
        window_start: Start date string
        window_end: End date string (optional)
        
    Returns:
        Batch ID string
    """
    if window_end:
        return f"batch_{window_start}_to_{window_end}"
    return f"batch_{window_start}"


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, mentions, and extra whitespace.
    
    NOTE: This is kept for backward compatibility. New code should use
    ``src.data.preprocessing.clean_for_dataset(text, mode)`` instead.
    
    Defaults to the *twitter* preset (removes @mentions, agent signatures, etc.).
    """
    from src.data.preprocessing import clean_text_twitter
    return clean_text_twitter(text)


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str)
    except:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.strptime(timestamp_str, "%Y-%m-%d")


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat()


def normalize_weights(word_weight_list: List[tuple]) -> List[float]:
    """
    Normalize word weights to probability distribution.
    
    Args:
        word_weight_list: List of (word, weight) tuples
        
    Returns:
        List of normalized weights
    """
    weights = [w for _, w in word_weight_list]
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return ((current - previous) / previous) * 100


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

