"""
Colors and styling constants for the medical AI dashboard.
Centralized color management for consistency.
"""

# Primary colors
PRIMARY = "#3B82F6"
PRIMARY_DARK = "#1E40AF"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

# Background colors
LIGHT_BG = "#F9FAFB"
DARK_BG = "#111827"

# Card colors
LIGHT_CARD = "#FFFFFF"
DARK_CARD = "#1F2937"

# Text colors
LIGHT_TEXT = "#111827"
DARK_TEXT = "#F9FAFB"

# Border colors
LIGHT_BORDER = "#E5E7EB"
DARK_BORDER = "#374151"

# Risk colors
RISK_LOW = "#10B981"  # Green
RISK_MEDIUM = "#F59E0B"  # Yellow
RISK_HIGH = "#EF4444"  # Red


def get_risk_color(score: float) -> str:
    """Get color based on risk score (0-1)."""
    if score < 0.4:
        return RISK_LOW
    elif score < 0.7:
        return RISK_MEDIUM
    else:
        return RISK_HIGH


def get_risk_label(score: float) -> str:
    """Get risk label based on score (0-1)."""
    if score < 0.4:
        return "LOW RISK"
    elif score < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def get_risk_emoji(score: float) -> str:
    """Get emoji for risk level."""
    if score < 0.4:
        return "🟢"
    elif score < 0.7:
        return "🟡"
    else:
        return "🔴"
