"""
Helper utility functions for the Streamlit app
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
import re

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with decimals"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def calculate_age(birth_date) -> int:
    """Calculate age from birth date"""
    try:
        if isinstance(birth_date, str):
            birth = datetime.strptime(birth_date, "%Y-%m-%d")
        else:
            birth = birth_date
        return (datetime.now() - birth).days // 365
    except Exception as e:
        logger.error(f"Error calculating age: {e}")
        return 0


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers"""
    return numerator / denominator if denominator != 0 else default


def calculate_percentile(value: float, data: List[float]) -> float:
    """Calculate percentile of value in data"""
    try:
        return (sum(1 for x in data if x <= value) / len(data)) * 100
    except:
        return 0


def get_risk_level(risk_score: float) -> str:
    """Get risk level label from score"""
    if risk_score < 0.30:
        return "🟢 LOW"
    elif risk_score < 0.60:
        return "🟡 MODERATE"
    else:
        return "🔴 HIGH"


def get_risk_color(risk_score: float) -> str:
    """Get color based on risk score"""
    if risk_score < 0.30:
        return "#28a745"  # Green
    elif risk_score < 0.60:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def format_currency(value: float) -> str:
    """Format as currency"""
    return f"${value:,.2f}"


def format_date(date_obj, fmt: str = "%Y-%m-%d") -> str:
    """Format date object"""
    try:
        if isinstance(date_obj, str):
            return date_obj
        return date_obj.strftime(fmt)
    except:
        return str(date_obj)


def get_days_ago(date_obj) -> int:
    """Calculate days since date"""
    try:
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        return (datetime.now() - date_obj).days
    except:
        return 0


def truncate_text(text: str, length: int = 50) -> str:
    """Truncate text to length"""
    return text[:length] + "..." if len(text) > length else text


def parse_csv_data(csv_content: str) -> pd.DataFrame:
    """Parse CSV content"""
    try:
        from io import StringIO
        return pd.read_csv(StringIO(csv_content))
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        return pd.DataFrame()


def create_download_link(data: str, filename: str, format_type: str = "csv"):
    """Create download link for data"""
    import base64
    b64 = base64.b64encode(data.encode()).decode()
    
    if format_type == "csv":
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    else:
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download File</a>'
    
    return href
