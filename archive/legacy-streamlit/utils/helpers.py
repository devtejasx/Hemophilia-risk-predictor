"""
Utility helper functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with decimals"""
    return f"{value:.{decimals}f}"


def calculate_age(birth_date: str) -> int:
    """Calculate age from birth date"""
    try:
        birth = datetime.strptime(birth_date, "%Y-%m-%d")
        return (datetime.now() - birth).days // 365
    except Exception as e:
        logger.error(f"Error calculating age: {e}")
        return 0


def validate_email(email: str) -> bool:
    """Validate email format"""
    import re
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
    """Get risk level from score"""
    if risk_score >= 75:
        return "🔴 Critical"
    elif risk_score >= 50:
        return "🟠 High"
    elif risk_score >= 25:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def get_severity_color(severity: str) -> str:
    """Get color code for severity"""
    colors = {
        "Mild": "#00ff88",
        "Moderate": "#ffa500",
        "Severe": "#ff1744",
        "Critical": "#cc0000",
    }
    return colors.get(severity, "#b0b5c0")


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to max length"""
    return text[:max_length] + "..." if len(text) > max_length else text


def format_large_number(num: int) -> str:
    """Format large numbers (e.g., 1000 -> 1K)"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def get_date_range_label(days: int) -> str:
    """Get label for date range"""
    if days == 7:
        return "Last 7 Days"
    elif days == 30:
        return "Last 30 Days"
    elif days == 90:
        return "Last 90 Days"
    elif days == 365:
        return "Last Year"
    else:
        return f"Last {days} Days"


def calculate_trend(current: float, previous: float) -> Tuple[str, str]:
    """Calculate trend between two values"""
    if previous == 0:
        return "→", "No change"
    
    percent_change = ((current - previous) / previous) * 100
    
    if percent_change > 0:
        return "↑", f"+{percent_change:.1f}%"
    elif percent_change < 0:
        return "↓", f"{percent_change:.1f}%"
    else:
        return "→", "No change"


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string"""
    return df.to_csv(index=False).encode('utf-8')


def format_metric_with_unit(value: float, unit: str) -> str:
    """Format metric with unit"""
    return f"{format_number(value, 1)} {unit}"


def get_risk_factors_summary(importance_dict: Dict[str, float]) -> List[Tuple[str, float]]:
    """Get top risk factors sorted by importance"""
    sorted_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_factors[:5]  # Top 5


def classify_mutation(mutation_type: str) -> str:
    """Classify mutation severity"""
    mild_mutations = ["F5", "F8", "FV"]
    moderate_mutations = ["FIX", "FII", "FVII"]
    severe_mutations = ["FX", "FXI", "FXII"]
    
    if mutation_type in mild_mutations:
        return "Mild"
    elif mutation_type in moderate_mutations:
        return "Moderate"
    elif mutation_type in severe_mutations:
        return "Severe"
    else:
        return "Unknown"


def calculate_patient_risk_score(**kwargs) -> Dict[str, Any]:
    """
    Calculate patient risk score based on clinical parameters
    
    Args:
        age, severity, mutation, blood_type, dose, exposure, treatment_adherence
    
    Returns:
        Dict with risk_score, classification, and factors
    """
    try:
        age = kwargs.get('age', 0)
        severity = kwargs.get('severity', 'Moderate')
        mutation = kwargs.get('mutation', '')
        blood_type = kwargs.get('blood_type', 'O')
        dose = kwargs.get('dose', 50)
        exposure = kwargs.get('exposure', 0)
        treatment_adherence = kwargs.get('treatment_adherence', 80)
        
        # Base score
        score = 20
        
        # Age factor
        if age < 10:
            score += 15
        elif age < 20:
            score += 10
        elif age > 65:
            score += 5
        
        # Severity factor
        if severity == "Severe":
            score += 35
        elif severity == "Moderate":
            score += 20
        elif severity == "Mild":
            score += 5
        
        # Treatment adherence factor
        score += (100 - treatment_adherence) * 0.3
        
        # Exposure factor
        score += min(exposure * 2, 20)
        
        # Normalize to 0-100
        risk_score = min(max(score, 0), 100)
        
        return {
            "risk_score": risk_score,
            "classification": get_risk_level(risk_score),
            "factors": {
                "Age": age,
                "Severity": severity,
                "Mutation": mutation,
                "Dose (IU)": dose,
                "Exposure": exposure,
                "Adherence": treatment_adherence,
            }
        }
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return {
            "risk_score": 0,
            "classification": "Unknown",
            "factors": {}
        }
