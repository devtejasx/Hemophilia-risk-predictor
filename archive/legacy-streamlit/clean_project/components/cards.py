"""
Card components for displaying metrics and KPIs.
Reusable card widgets for dashboard layout.
"""

import streamlit as st
from typing import Optional, Any
from utils import format_number, format_percentage
from colors import THEMES, get_risk_color, get_risk_label
from utils import is_dark_mode


def render_kpi_card(
    title: str,
    value: Any,
    unit: str = "",
    icon: str = "",
    delta: Optional[float] = None,
    help_text: str = "",
) -> None:
    """Render a KPI (Key Performance Indicator) card.
    
    Args:
        title: Card title
        value: Main metric value to display
        unit: Unit suffix (e.g., "%", "mg/dL")
        icon: Emoji icon for the card
        delta: Change value to display (positive/negative)
        help_text: Hover text for tooltip
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    # Format value
    if isinstance(value, float):
        formatted_value = format_number(value)
    else:
        formatted_value = str(value)
    
    # Delta styling (green up, red down)
    delta_html = ""
    if delta is not None:
        delta_color = theme_colors["success"] if delta >= 0 else theme_colors["danger"]
        delta_symbol = "↑" if delta >= 0 else "↓"
        delta_html = f"""
        <div style="color: {delta_color}; font-size: 12px; margin-top: 5px;">
            {delta_symbol} {abs(delta):.1f}
        </div>
        """
    
    tooltip = f'title="{help_text}"' if help_text else ""
    
    card_html = f"""
    <style>
        .kpi-card {{
            background: linear-gradient(135deg, {theme_colors['card_bg']}, {theme_colors['card_border']});
            border: 1px solid {theme_colors['card_border']};
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .kpi-title {{
            font-size: 12px;
            font-weight: 600;
            color: {theme_colors['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .kpi-value {{
            font-size: 36px;
            font-weight: bold;
            color: {theme_colors['primary']};
            margin: 0;
        }}
        .kpi-icon {{
            font-size: 32px;
            margin-right: 10px;
        }}
        .kpi-content {{
            display: flex;
            align-items: center;
        }}
    </style>
    <div class="kpi-card" {tooltip}>
        <div class="kpi-title">{title}</div>
        <div class="kpi-content">
            {f'<div class="kpi-icon">{icon}</div>' if icon else ''}
            <div>
                <div class="kpi-value">{formatted_value}{unit}</div>
                {delta_html}
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: Any,
    format_type: str = "number",
    color: Optional[str] = None,
    size: str = "medium",
) -> None:
    """Render a simple metric card with label and value.
    
    Args:
        label: Metric label
        value: Metric value
        format_type: 'number', 'percentage', 'text'
        color: Color to use for styling
        size: 'small', 'medium', 'large'
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    # Format value
    if format_type == "percentage" and isinstance(value, (int, float)):
        formatted_value = format_percentage(value)
    elif format_type == "number" and isinstance(value, float):
        formatted_value = format_number(value)
    else:
        formatted_value = str(value)
    
    # Color selection
    if color is None:
        color = theme_colors["primary"]
    
    # Size settings
    size_settings = {
        "small": {"padding": "10px", "title_size": "12px", "value_size": "20px"},
        "medium": {"padding": "15px", "title_size": "13px", "value_size": "24px"},
        "large": {"padding": "20px", "title_size": "14px", "value_size": "32px"},
    }
    
    settings = size_settings.get(size, size_settings["medium"])
    
    card_html = f"""
    <style>
        .metric-card {{
            background: {theme_colors['card_bg']};
            border: 1px solid {color}33;
            border-left: 4px solid {color};
            border-radius: 6px;
            padding: {settings['padding']};
            margin: 8px 0;
        }}
        .metric-label {{
            font-size: {settings['title_size']};
            color: {theme_colors['text_secondary']};
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: {settings['value_size']};
            color: {color};
            font-weight: bold;
        }}
    </style>
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{formatted_value}</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_risk_card(risk_score: float, risk_label: Optional[str] = None) -> None:
    """Render a risk assessment card with color coding.
    
    Args:
        risk_score: Risk score from 0 to 1
        risk_label: Optional custom risk label
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    # Use provided label or auto-generate
    if risk_label is None:
        risk_label = get_risk_label(risk_score)
    
    color = get_risk_color(risk_score)
    percentage = format_percentage(risk_score)
    
    card_html = f"""
    <style>
        .risk-card {{
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 2px solid {color};
            border-radius: 12px;
            padding: 25px;
            text-align: center;
        }}
        .risk-label {{
            font-size: 14px;
            color: {theme_colors['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .risk-value {{
            font-size: 48px;
            font-weight: bold;
            color: {color};
            margin: 10px 0;
        }}
        .risk-status {{
            font-size: 18px;
            color: {color};
            font-weight: 600;
        }}
    </style>
    <div class="risk-card">
        <div class="risk-label">Risk Level</div>
        <div class="risk-value">{percentage}</div>
        <div class="risk-status">{risk_label}</div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_stat_card(
    title: str,
    stat1_label: str,
    stat1_value: Any,
    stat2_label: str = "",
    stat2_value: Any = None,
) -> None:
    """Render a card with multiple statistics.
    
    Args:
        title: Card title
        stat1_label: First statistic label
        stat1_value: First statistic value
        stat2_label: Second statistic label (optional)
        stat2_value: Second statistic value (optional)
    """
    
    theme = "dark" if is_dark_mode() else "light"
    theme_colors = THEMES[theme]
    
    stat2_html = ""
    if stat2_label and stat2_value is not None:
        stat2_html = f"""
        <div style="display: flex; justify-content: space-between; margin-top: 10px; padding-top: 10px; border-top: 1px solid {theme_colors['card_border']};">
            <div style="color: {theme_colors['text_secondary']}; font-size: 12px;">{stat2_label}</div>
            <div style="color: {theme_colors['primary']}; font-weight: bold;">{stat2_value}</div>
        </div>
        """
    
    card_html = f"""
    <style>
        .stat-card {{
            background: {theme_colors['card_bg']};
            border: 1px solid {theme_colors['card_border']};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .stat-title {{
            font-size: 12px;
            font-weight: 600;
            color: {theme_colors['text_secondary']};
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stat-label {{
            color: {theme_colors['text']};
            font-size: 13px;
        }}
        .stat-value {{
            color: {theme_colors['primary']};
            font-weight: bold;
            font-size: 16px;
        }}
    </style>
    <div class="stat-card">
        <div class="stat-title">{title}</div>
        <div class="stat-row">
            <div class="stat-label">{stat1_label}</div>
            <div class="stat-value">{stat1_value}</div>
        </div>
        {stat2_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
