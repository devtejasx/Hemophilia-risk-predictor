"""
Common helper functions and utilities.
Reusable functions across the application.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{value:.{decimals}f}"


def format_risk_score(score: float) -> str:
    """Format risk score as percentage string."""
    return f"{score:.1%}"


def get_timestamp() -> str:
    """Get current timestamp formatted for display."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_string() -> str:
    """Get current date as formatted string."""
    return datetime.now().strftime("%B %d, %Y")


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def validate_email(email: str) -> bool:
    """Basic email validation."""
    return "@" in email and "." in email.split("@")[-1]


def validate_age(age: int, min_age: int = 0, max_age: int = 120) -> bool:
    """Validate age is within reasonable bounds."""
    try:
        age_int = int(age)
        return min_age <= age_int <= max_age
    except (ValueError, TypeError):
        return False


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def scale_value(value: float, old_min: float, old_max: float, 
                new_min: float, new_max: float) -> float:
    """Scale a value from one range to another."""
    if old_max == old_min:
        return new_min
    old_range = old_max - old_min
    new_range = new_max - new_min
    scaled = ((value - old_min) / old_range) * new_range + new_min
    return clamp(scaled, new_min, new_max)


def average(values: List[float]) -> float:
    """Calculate average of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_ordinal(n: int) -> str:
    """Convert number to ordinal (1st, 2nd, 3rd, etc.)."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix


def format_list_readable(items: List[str], conjunction: str = "and") -> str:
    """Format a list as readable string: 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"


def dict_to_table_data(data_dict: Dict) -> List[List]:
    """Convert dictionary to table data format for display."""
    return [[key, str(value)] for key, value in data_dict.items()]


def log_action(action: str, details: str = "", user: str = "system") -> None:
    """Log an action with timestamp (can be extended for file logging)."""
    timestamp = get_timestamp()
    message = f"[{timestamp}] {action}"
    if details:
        message += f" - {details}"
    # In production, this would write to a log file
    print(message)


def debounce_submit(key: str, cooldown_seconds: float = 1.0) -> bool:
    """Check if enough time has passed since last submit (prevents double-submission)."""
    if "last_submit" not in st.session_state:
        st.session_state.last_submit = {}
    
    current_time = datetime.now().timestamp()
    last_time = st.session_state.last_submit.get(key, 0)
    
    if current_time - last_time >= cooldown_seconds:
        st.session_state.last_submit[key] = current_time
        return True
    return False


def highlight_text(text: str, highlight_words: List[str]) -> str:
    """Highlight specific words in text (returns markdown)."""
    result = text
    for word in highlight_words:
        result = result.replace(word, f"**{word}**")
    return result


def parse_json_safe(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string, return default on error."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries, later ones override earlier ones."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def unique_list(items: List) -> List:
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
