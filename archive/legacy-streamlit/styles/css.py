"""
Styling module for consistent theming across the app
"""

import streamlit as st


def apply_theme(theme: str = "dark") -> None:
    """Apply CSS theme to the app"""
    
    if theme == "dark":
        dark_theme_css = """
        <style>
            :root {
                --primary: #00d4ff;
                --secondary: #0099ff;
                --success: #00ff88;
                --danger: #ff1744;
                --warning: #ffa500;
                --bg-primary: #0a0e27;
                --bg-secondary: #1a1f3a;
                --bg-tertiary: #2a2f4a;
                --text-primary: #ffffff;
                --text-secondary: #b0b5c0;
                --border-color: #2a3f5f;
            }
            
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .stMetric {
                background-color: var(--bg-secondary);
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid var(--primary);
            }
            
            .stButton>button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: var(--text-primary);
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3);
            }
            
            .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
                background-color: var(--bg-secondary);
                color: var(--text-primary);
                border-radius: 8px;
            }
            
            .stCard {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 0px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding: 0px 20px;
                background-color: var(--bg-secondary);
                border-radius: 8px 8px 0px 0px;
                color: var(--text-secondary);
                border: 1px solid var(--border-color);
                margin-right: 5px;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: var(--primary);
                color: var(--bg-primary);
                border: 1px solid var(--primary);
            }
            
            [data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
            }
            
            .stRadio [role="radiogroup"] {
                gap: 10px;
            }
            
            .stRadio [role="radio"] {
                padding: 10px;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .stRadio [role="radio"][aria-checked="true"] {
                background-color: var(--primary);
                color: var(--bg-primary);
            }
            
            .metric-card {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
                border-left: 4px solid var(--primary);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
            }
            
            .success-card {
                background-color: rgba(0, 255, 136, 0.1);
                border-left: 4px solid var(--success);
            }
            
            .danger-card {
                background-color: rgba(255, 23, 68, 0.1);
                border-left: 4px solid var(--danger);
            }
            
            .warning-card {
                background-color: rgba(255, 165, 0, 0.1);
                border-left: 4px solid var(--warning);
            }
            
            .info-card {
                background-color: rgba(0, 212, 255, 0.1);
                border-left: 4px solid var(--primary);
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--bg-secondary);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--primary);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary);
            }
        </style>
        """
    else:  # light theme
        dark_theme_css = """
        <style>
            :root {
                --primary: #0066cc;
                --secondary: #0099ff;
                --success: #00aa00;
                --danger: #cc0000;
                --warning: #ff9900;
                --bg-primary: #ffffff;
                --bg-secondary: #f5f5f5;
                --bg-tertiary: #eeeeee;
                --text-primary: #333333;
                --text-secondary: #666666;
                --border-color: #cccccc;
            }
            
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
        </style>
        """
    
    st.markdown(dark_theme_css, unsafe_allow_html=True)


def get_metric_color(value: float, threshold_high: float = 70, threshold_low: float = 30) -> str:
    """Get color based on metric value"""
    if value >= threshold_high:
        return "🔴"  # Red - High risk
    elif value >= threshold_low:
        return "🟡"  # Yellow - Medium
    else:
        return "🟢"  # Green - Low risk


def create_metric_card(label: str, value: str, delta: str = "", color: str = "info") -> str:
    """Create a metric card HTML"""
    bg_class = f"{color}-card"
    return f"""
    <div class="{bg_class}" style='border-radius: 10px; padding: 20px; margin: 10px 0;'>
        <h3 style='margin: 0; color: var(--text-secondary);'>{label}</h3>
        <h2 style='margin: 10px 0 0 0; color: var(--text-primary);'>{value}</h2>
        {f'<p style="margin: 5px 0 0 0; color: var(--text-secondary);">{delta}</p>' if delta else ''}
    </div>
    """
