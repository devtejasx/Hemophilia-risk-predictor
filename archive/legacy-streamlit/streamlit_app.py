"""
Modern Streamlit Medical AI Dashboard
Hemophilia Clinical Decision Support System with professional UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="Medical AI - Hemophilia Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# INITIALIZE THEME STATE
# ============================================================================

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ============================================================================
# DEFINE THEME & COLORS
# ============================================================================

LIGHT_THEME = {
    "primary": "#4F46E5",
    "primary_light": "#6366F1",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#F9FAFB",
    "card": "#FFFFFF",
    "text": "#111827",
    "text_secondary": "#6B7280",
    "border": "#E5E7EB",
    "shadow": "rgba(0, 0, 0, 0.05)",
    "shadow_hover": "rgba(0, 0, 0, 0.08)",
}

DARK_THEME = {
    "primary": "#6366F1",
    "primary_light": "#818CF8",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#0F172A",
    "card": "#1E293B",
    "text": "#F1F5F9",
    "text_secondary": "#CBD5E1",
    "border": "#334155",
    "shadow": "rgba(0, 0, 0, 0.3)",
    "shadow_hover": "rgba(0, 0, 0, 0.5)",
}

# Select theme based on session state
COLORS = DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME

def get_theme_css():
    """Generate CSS based on current theme"""
    return f"""
<style>
    :root {{
        --primary: {COLORS['primary']};
        --primary-light: {COLORS['primary_light']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --danger: {COLORS['danger']};
        --background: {COLORS['background']};
        --card: {COLORS['card']};
        --text: {COLORS['text']};
        --text-secondary: {COLORS['text_secondary']};
        --border: {COLORS['border']};
        --shadow: {COLORS['shadow']};
        --shadow-hover: {COLORS['shadow_hover']};
        
        /* Animation timing functions for smooth motion */
        --ease-out-smooth: cubic-bezier(0.34, 1.56, 0.64, 1);
        --ease-in-out-quad: cubic-bezier(0.45, 0, 0.55, 1);
        --ease-out-cubic: cubic-bezier(0.215, 0.61, 0.355, 1);
    }}

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{
        background-color: var(--background) !important;
        color: var(--text) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
            'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica',
            'Arial', sans-serif !important;
        transition: background-color 0.4s var(--ease-in-out-quad), color 0.4s var(--ease-in-out-quad) !important;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: var(--card) !important;
        border-right: 1px solid var(--border) !important;
        transition: all 0.4s var(--ease-in-out-quad) !important;
        will-change: background-color, border-color;
        transform: translateZ(0);
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--card) !important;
        transition: background-color 0.4s var(--ease-in-out-quad) !important;
    }}

    /* Main Content */
    [data-testid="stAppViewContainer"] {{
        background-color: var(--background) !important;
        transition: all 0.4s var(--ease-in-out-quad) !important;
        will-change: background-color;
        transform: translateZ(0);
    }}

    /* Cards & Containers */
    .card {{
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px var(--shadow) !important;
        transition: all 0.35s var(--ease-in-out-quad) !important;
        animation: slideInUp 0.6s var(--ease-out-cubic) backwards !important;
        will-change: transform, box-shadow, border-color;
        transform: translateZ(0);
        backface-visibility: hidden;
    }}

    .card:hover {{
        box-shadow: 0 8px 24px var(--shadow-hover) !important;
        transform: translateY(-4px) translateZ(0) !important;
        border-color: var(--primary) !important;
    }}

    .card:active {{
        transform: translateY(-2px) translateZ(0) !important;
    }}

    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, var(--card) 0%, rgba(79, 70, 229, 0.05) 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        text-align: center !important;
        transition: all 0.4s var(--ease-out-cubic) !important;
        animation: slideInUp 0.7s var(--ease-out-cubic) backwards !important;
        will-change: transform, box-shadow;
        transform: translateZ(0);
        backface-visibility: hidden;
        position: relative;
        overflow: hidden;
    }}

    .metric-card::before {{
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent) !important;
        transition: left 0.5s ease !important;
    }}

    .metric-card:hover {{
        box-shadow: 0 12px 32px var(--shadow-hover) !important;
        transform: translateY(-6px) scale(1.02) translateZ(0) !important;
    }}

    .metric-card:hover::before {{
        left: 100% !important;
    }}

    .metric-value {{
        font-size: 32px !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        margin: 8px 0 !important;
        animation: countUp 0.8s var(--ease-out-smooth) backwards !important;
        transition: color 0.3s ease !important;
    }}

    .metric-card:hover .metric-value {{
        color: var(--primary-light) !important;
    }}

    .metric-label {{
        font-size: 14px !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: color 0.3s ease !important;
        animation: fadeIn 0.8s ease backwards !important;
    }}

    .metric-card:hover .metric-label {{
        color: var(--text) !important;
    }}

    /* Risk Score Styling */
    .risk-low {{
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--success) !important;
        animation: slideInLeft 0.6s var(--ease-out-cubic) backwards !important;
        transition: all 0.3s ease !important;
        will-change: transform, background-color;
        transform: translateZ(0);
    }}

    .risk-medium {{
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid var(--warning) !important;
        animation: slideInLeft 0.6s var(--ease-out-cubic) backwards !important;
        transition: all 0.3s ease !important;
        will-change: transform, background-color;
        transform: translateZ(0);
    }}

    .risk-high {{
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid var(--danger) !important;
        animation: slideInLeft 0.6s var(--ease-out-cubic) backwards !important;
        transition: all 0.3s ease !important;
        will-change: transform, background-color;
        transform: translateZ(0);
    }}

    .risk-low:hover, .risk-medium:hover, .risk-high:hover {{
        transform: translateX(4px) translateZ(0) !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        will-change: transform, box-shadow, background-color;
        transform: translateZ(0);
        backface-visibility: hidden;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2) !important;
    }}

    .stButton > button::before {{
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        border-radius: 50% !important;
        background: rgba(255, 255, 255, 0.4) !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.5s var(--ease-out-cubic), height 0.5s var(--ease-out-cubic) !important;
        pointer-events: none !important;
    }}

    .stButton > button:hover {{
        background-color: var(--primary-light) !important;
        transform: translateY(-2px) translateZ(0) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) translateZ(0) !important;
    }}

    .stButton > button:active::before {{
        width: 300px !important;
        height: 300px !important;
    }}

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {{
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        color: var(--text) !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        will-change: border-color, box-shadow, background-color;
        transform: translateZ(0);
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15), 0 0 0 1px var(--primary) !important;
        transform: scale(1.01) translateZ(0) !important;
        background-color: var(--card) !important;
    }}

    .stTextInput > div > div > input:hover:not(:focus),
    .stNumberInput > div > div > input:hover:not(:focus),
    .stSelectbox > div > div > select:hover:not(:focus),
    .stTextArea > div > div > textarea:hover:not(:focus) {{
        border-color: var(--text-secondary) !important;
    }}

    /* Labels */
    label {{
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
    }}

    /* Helper Text */
    .helper-text {{
        font-size: 13px !important;
        color: var(--text-secondary) !important;
        margin-top: 4px !important;
        animation: fadeIn 0.5s ease-out !important;
    }}

    /* Progress Bar */
    .stProgress > div > div > div {{
        background-color: var(--primary) !important;
        transition: width 0.5s ease-out !important;
    }}

    /* Alerts */
    .stAlert {{
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        padding: 16px !important;
        animation: slideDown 0.5s ease-out !important;
    }}

    .stSuccess {{
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-color: var(--success) !important;
    }}

    .stError {{
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-color: var(--danger) !important;
    }}

    .stWarning {{
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-color: var(--warning) !important;
    }}

    .stInfo {{
        background-color: rgba(79, 70, 229, 0.1) !important;
        border-color: var(--primary) !important;
    }}

    /* Chat Messages */
    .chat-message {{
        padding: 12px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        word-wrap: break-word !important;
        animation: slideInUp 0.4s var(--ease-out-cubic) backwards !important;
        will-change: transform, opacity;
        transform: translateZ(0);
        backface-visibility: hidden;
        transition: all 0.2s ease !important;
    }}

    .user-message {{
        background-color: var(--primary) !important;
        color: white !important;
        margin-left: 40px !important;
        border-radius: 18px 18px 4px 18px !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
    }}

    .user-message:hover {{
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
    }}

    .ai-message {{
        background-color: var(--background) !important;
        color: var(--text) !important;
        margin-right: 40px !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px 18px 18px 4px !important;
        box-shadow: 0 1px 4px var(--shadow) !important;
    }}

    .ai-message:hover {{
        box-shadow: 0 2px 8px var(--shadow-hover) !important;
    }}

    /* Typing Animation */
    .typing-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }}

    .typing-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--text-secondary);
        animation: typing 1.4s infinite;
    }}

    .typing-dot:nth-child(2) {{
        animation-delay: 0.2s;
    }}

    .typing-dot:nth-child(3) {{
        animation-delay: 0.4s;
    }}

    /* Skeleton Loader */
    .skeleton {{
        background: linear-gradient(90deg, var(--border) 25%, var(--background) 50%, var(--border) 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
    }}

    @keyframes loading {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    @keyframes fadeIn {{
        from {{
            opacity: 0;
        }}
        to {{
            opacity: 1;
        }}
    }}

    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateY(12px) translateZ(0);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) translateZ(0);
        }}
    }}

    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(16px) translateZ(0);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) translateZ(0);
        }}
    }}

    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-24px) translateZ(0);
        }}
        to {{
            opacity: 1;
            transform: translateX(0) translateZ(0);
        }}
    }}

    @keyframes slideDown {{
        from {{
            opacity: 0;
            transform: translateY(-12px) translateZ(0);
        }}
        to {{
            opacity: 1;
            transform: translateY(0) translateZ(0);
        }}
    }}

    @keyframes countUp {{
        from {{
            opacity: 0;
            transform: scale(0.85) translateZ(0);
        }}
        to {{
            opacity: 1;
            transform: scale(1) translateZ(0);
        }}
    }}

    @keyframes typing {{
        0%, 60%, 100% {{
            opacity: 0.6;
            transform: translateY(0);
        }}
        30% {{
            opacity: 1;
            transform: translateY(-8px);
        }}
    }}

    @keyframes spin {{
        from {{ 
            transform: rotate(0deg) translateZ(0); 
        }}
        to {{ 
            transform: rotate(360deg) translateZ(0); 
        }}
    }}

    @keyframes shimmer {{
        0% {{
            background-position: -1000px 0;
        }}
        100% {{
            background-position: 1000px 0;
        }}
    }}

    @keyframes float {{
        0%, 100% {{
            transform: translateY(0px) translateZ(0);
        }}
        50% {{
            transform: translateY(-8px) translateZ(0);
        }}
    }}

    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.7;
        }}
    }}

    .spinner {{
        animation: spin 1s linear infinite;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {{
        color: var(--text-secondary) !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.3s var(--ease-in-out-quad) !important;
        will-change: color, background-color;
        transform: translateZ(0);
        position: relative;
    }}

    .stTabs [data-baseweb="tab-list"] button:hover {{
        color: var(--text) !important;
        background-color: rgba(79, 70, 229, 0.05) !important;
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: var(--primary) !important;
        border-bottom: 3px solid var(--primary) !important;
        animation: slideDown 0.35s var(--ease-out-cubic) !important;
    }}

    /* Divider */
    hr {{
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 20px 0 !important;
        animation: fadeIn 0.6s ease-out !important;
        transition: border-color 0.3s ease !important;
    }}

    /* Headings */
    h1 {{
        color: var(--text) !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        animation: slideInUp 0.7s var(--ease-out-cubic) backwards !important;
        transition: color 0.3s ease !important;
    }}

    h2 {{
        color: var(--text) !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        animation: slideInUp 0.65s var(--ease-out-cubic) backwards !important;
        transition: color 0.3s ease !important;
    }}

    h3 {{
        color: var(--text) !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-top: 16px !important;
        margin-bottom: 12px !important;
        animation: slideInUp 0.6s var(--ease-out-cubic) backwards !important;
        transition: color 0.3s ease !important;
    }}

    p {{
        color: var(--text-secondary) !important;
        line-height: 1.6 !important;
        transition: color 0.3s ease !important;
    }}

    /* Data Table */
    .stDataFrame {{
        border-radius: 8px !important;
        overflow: hidden !important;
        animation: slideInUp 0.7s var(--ease-out-cubic) backwards !important;
        will-change: opacity;
        transform: translateZ(0);
    }}

    .stDataFrame:hover {{
        box-shadow: 0 4px 16px var(--shadow) !important;
    }}

    /* Chart Container */
    .plotly-container {{
        animation: slideInUp 0.8s var(--ease-out-cubic) backwards !important;
        will-change: opacity;
        transform: translateZ(0);
        transition: all 0.3s ease !important;
    }}

    .plotly-container:hover {{
        transform: translateY(-2px) translateZ(0) !important;
    }}

    /* Progress Indicators */
    .stProgress {{
        will-change: width;
        transform: translateZ(0);
    }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
        transition: width 0.6s var(--ease-out-cubic) !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
        border-radius: 4px !important;
    }}

    /* Select/Dropdown Styling */
    .stSelectbox {{
        will-change: border-color, background-color;
        transform: translateZ(0);
    }}

    .stSelectbox > div > div > select:hover {{
        border-color: var(--primary) !important;
    }}

    /* Radio & Checkbox */
    .stRadio, .stCheckbox {{
        will-change: opacity;
        transform: translateZ(0);
    }}

    .stRadio > div, .stCheckbox > div {{
        transition: all 0.2s ease !important;
    }}

    .stRadio > div:hover, .stCheckbox > div:hover {{
        opacity: 0.8 !important;
    }}

    /* Slider Styling */
    .stSlider {{
        will-change: opacity;
        transform: translateZ(0);
    }}

    .stSlider > div > div > div > input {{
        transition: all 0.2s ease !important;
    }}

    /* General Transition for Theme */
    * {{
        transition-property: background-color, color, border-color, box-shadow !important;
        transition-duration: 0.3s !important;
        transition-timing-function: var(--ease-in-out-quad) !important;
    }}

    /* Disable animations on reduced motion */
    @media (prefers-reduced-motion: reduce) {{
        * {{
            animation: none !important;
            transition: none !important;
        }}
    }}
</style>
"""

# ============================================================================
# UTILITY FUNCTIONS (DEFINE FIRST BEFORE USAGE)
# ============================================================================

def load_sample_patients():
    """Load sample patient data"""
    return [
        {
            "id": "PAT001",
            "name": "John Smith",
            "age": 45,
            "gender": "Male",
            "risk_score": 0.72,
            "risk_category": "High",
            "created_at": datetime.now() - timedelta(days=3),
        },
        {
            "id": "PAT002",
            "name": "Sarah Johnson",
            "age": 38,
            "gender": "Female",
            "risk_score": 0.45,
            "risk_category": "Medium",
            "created_at": datetime.now() - timedelta(days=1),
        },
        {
            "id": "PAT003",
            "name": "Michael Brown",
            "age": 52,
            "gender": "Male",
            "risk_score": 0.28,
            "risk_category": "Low",
            "created_at": datetime.now() - timedelta(days=5),
        },
        {
            "id": "PAT004",
            "name": "Emily Davis",
            "age": 41,
            "gender": "Female",
            "risk_score": 0.65,
            "risk_category": "High",
            "created_at": datetime.now() - timedelta(days=2),
        },
    ]

def get_risk_color(risk_score):
    """Get color based on risk score"""
    if risk_score < 0.4:
        return COLORS["success"]
    elif risk_score < 0.7:
        return COLORS["warning"]
    else:
        return COLORS["danger"]

def get_risk_category(risk_score):
    """Get risk category based on score"""
    if risk_score < 0.4:
        return "Low"
    elif risk_score < 0.7:
        return "Medium"
    else:
        return "High"

def render_metric_card(label, value, unit="", icon="📊"):
    """Render a metric card"""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<h1 style='font-size: 32px; margin: 0;'>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{value}{unit}</div>", unsafe_allow_html=True)

def render_risk_card(risk_score, patient_name="Patient"):
    """Render a risk score card"""
    risk_category = get_risk_category(risk_score)
    risk_color = get_risk_color(risk_score)
    risk_class = f"risk-{risk_category.lower()}"
    
    st.markdown(f"""
    <div class='card {risk_class}'>
        <h3 style='margin: 0 0 8px 0;'>Bleeding Risk Score</h3>
        <div style='
            font-size: 48px;
            font-weight: 700;
            color: {risk_color};
            margin: 16px 0;
        '>{risk_score:.0%}</div>
        <div style='
            font-size: 16px;
            font-weight: 600;
            color: {risk_color};
        '>Risk Level: {risk_category}</div>
        <p style='margin: 12px 0 0 0; color: var(--text-secondary);'>
            {patient_name} has a {risk_category.lower()} probability of experiencing a bleeding event
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "form_step" not in st.session_state:
        st.session_state.form_step = 0
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = {}
    if "patients_list" not in st.session_state:
        st.session_state.patients_list = load_sample_patients()

init_session_state()

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

def page_dashboard():
    """Dashboard page with KPIs and overview"""
    st.title("📊 Dashboard")
    st.markdown("<p style='color: var(--text-secondary); font-size: 16px;'>Welcome back! Here's your clinical overview</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Create a container for the card
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Patients</div>
            <div class='metric-value'>{len(st.session_state.patients_list)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = len([p for p in st.session_state.patients_list if get_risk_category(p["risk_score"]) == "High"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>High-Risk Patients</div>
            <div class='metric-value' style='color: {COLORS["danger"]};'>{high_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_risk = np.mean([p["risk_score"] for p in st.session_state.patients_list])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Risk Score</div>
            <div class='metric-value'>{avg_risk:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Today's Visits</div>
            <div class='metric-value'>3</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        
        # Risk distribution pie chart
        risk_categories = [get_risk_category(p["risk_score"]) for p in st.session_state.patients_list]
        risk_counts = pd.Series(risk_categories).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(
                colors=[COLORS["success"], COLORS["warning"], COLORS["danger"]],
                line=dict(color=COLORS["card"], width=2)
            ),
            hoverinfo="label+percent"
        )])
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], family="Arial, sans-serif"),
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Trends (Last 7 Days)")
        
        # Generate trend data
        dates = [datetime.now() - timedelta(days=i) for i in range(6, -1, -1)]
        trend_data = np.random.uniform(0.3, 0.7, 7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_data,
            mode='lines+markers',
            name='Average Risk',
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8, color=COLORS["primary"]),
            fill='tozeroy',
            fillcolor=f"rgba(79, 70, 229, 0.1)"
        ))
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], family="Arial, sans-serif"),
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='x unified',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Recent Patients Table
    st.markdown("### Recent Patients")
    
    df = pd.DataFrame(st.session_state.patients_list)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["risk_score"] = df["risk_score"].apply(lambda x: f"{x:.0%}")
    
    # Create a display-friendly dataframe
    display_df = df[["id", "name", "age", "risk_score", "risk_category", "created_at"]].rename(
        columns={
            "id": "ID",
            "name": "Patient Name",
            "age": "Age",
            "risk_score": "Risk Score",
            "risk_category": "Category",
            "created_at": "Created"
        }
    )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: ADD PATIENT (Multi-step Form)
# ============================================================================

def page_add_patient():
    """Add patient with multi-step form wizard"""
    st.title("➕ Add New Patient")
    st.markdown("<p style='color: var(--text-secondary); font-size: 16px;'>Enter patient information step by step</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Progress indicator
    step = st.session_state.form_step
    progress = (step + 1) / 4
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<div style='font-weight: 600; color: var(--primary);'>Step {step + 1}/4</div>", unsafe_allow_html=True)
    with col2:
        st.progress(progress)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # STEP 0: Basic Info
    if step == 0:
        st.markdown("### 👤 Basic Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<label>First Name *</label>", unsafe_allow_html=True)
            first_name = st.text_input("first_name", placeholder="Enter first name", label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Patient's first name</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<label>Last Name *</label>", unsafe_allow_html=True)
            last_name = st.text_input("last_name", placeholder="Enter last name", label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Patient's last name</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<label>Age *</label>", unsafe_allow_html=True)
            age = st.number_input("age", min_value=0, max_value=150, value=30, label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Patient's age in years</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<label>Gender *</label>", unsafe_allow_html=True)
            gender = st.selectbox("gender", ["Male", "Female", "Other"], label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Patient's gender</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Email</label>", unsafe_allow_html=True)
        email = st.text_input("email", placeholder="patient@example.com", label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Patient's contact email</p>", unsafe_allow_html=True)
        
        st.session_state.patient_data["first_name"] = first_name
        st.session_state.patient_data["last_name"] = last_name
        st.session_state.patient_data["age"] = age
        st.session_state.patient_data["gender"] = gender
        st.session_state.patient_data["email"] = email
    
    # STEP 1: Medical History
    elif step == 1:
        st.markdown("### 🏥 Medical History")
        
        st.markdown("<label>Hemophilia Type *</label>", unsafe_allow_html=True)
        hemo_type = st.selectbox("hemo_type", ["Type A", "Type B"], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Type of hemophilia diagnosis</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Severity Level *</label>", unsafe_allow_html=True)
        severity = st.selectbox("severity", ["Mild", "Moderate", "Severe"], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Severity classification</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<label>Baseline Factor Level (%) *</label>", unsafe_allow_html=True)
            factor_level = st.slider("factor_level", 0, 100, 50, label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Baseline clotting factor percentage</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<label>Previous Bleeding Episodes (Last Year) *</label>", unsafe_allow_html=True)
            episodes = st.number_input("episodes", min_value=0, max_value=50, value=0, label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Number of bleeding episodes</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Comorbidities</label>", unsafe_allow_html=True)
        comorbidities = st.multiselect("comorbidities", [
            "Hypertension",
            "Diabetes",
            "HIV",
            "Hepatitis C",
            "Joint Damage",
            "None"
        ], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Select all applicable conditions</p>", unsafe_allow_html=True)
        
        st.session_state.patient_data["hemo_type"] = hemo_type
        st.session_state.patient_data["severity"] = severity
        st.session_state.patient_data["factor_level"] = factor_level
        st.session_state.patient_data["episodes"] = episodes
        st.session_state.patient_data["comorbidities"] = comorbidities
    
    # STEP 2: Treatment
    elif step == 2:
        st.markdown("### 💊 Treatment Information")
        
        st.markdown("<label>Current Treatment Type *</label>", unsafe_allow_html=True)
        treatment_type = st.selectbox("treatment_type", [
            "Factor Replacement Prophylaxis",
            "Factor Replacement On-Demand",
            "Bypassing Agent",
            "Gene Therapy",
            "Novel Agent"
        ], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Patient's treatment regimen</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<label>Dosage (IU/kg) *</label>", unsafe_allow_html=True)
            dosage = st.number_input("dosage", min_value=0.0, value=50.0, label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Treatment dosage</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<label>Frequency (times per week) *</label>", unsafe_allow_html=True)
            frequency = st.slider("frequency", 1, 7, 3, label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Administration frequency</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Inhibitor Status *</label>", unsafe_allow_html=True)
        inhibitor = st.selectbox("inhibitor", [
            "Negative",
            "Positive (Low-titer)",
            "Positive (High-titer)"
        ], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Factor VIII/IX inhibitor status</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Adherence Level *</label>", unsafe_allow_html=True)
        adherence = st.selectbox("adherence", [
            "Excellent (>90%)",
            "Good (70-90%)",
            "Fair (50-70%)",
            "Poor (<50%)"
        ], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Treatment adherence</p>", unsafe_allow_html=True)
        
        st.session_state.patient_data["treatment_type"] = treatment_type
        st.session_state.patient_data["dosage"] = dosage
        st.session_state.patient_data["frequency"] = frequency
        st.session_state.patient_data["inhibitor"] = inhibitor
        st.session_state.patient_data["adherence"] = adherence
    
    # STEP 3: Lifestyle
    elif step == 3:
        st.markdown("### 🏃 Lifestyle & Additional Info")
        
        st.markdown("<label>Activity Level *</label>", unsafe_allow_html=True)
        activity = st.selectbox("activity", [
            "Sedentary",
            "Light",
            "Moderate",
            "Active",
            "Very Active"
        ], label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Patient's typical activity level</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<label>Smoking Status *</label>", unsafe_allow_html=True)
            smoking = st.selectbox("smoking", ["Never", "Former", "Current"], label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Smoking status</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<label>Alcohol Consumption *</label>", unsafe_allow_html=True)
            alcohol = st.selectbox("alcohol", ["None", "Moderate", "Frequent"], label_visibility="collapsed")
            st.markdown("<p class='helper-text'>Alcohol use frequency</p>", unsafe_allow_html=True)
        
        st.markdown("<label>Notable Medical Events / Surgeries</label>", unsafe_allow_html=True)
        medical_events = st.text_area("medical_events", placeholder="Enter any relevant surgeries or medical events", height=100, label_visibility="collapsed")
        st.markdown("<p class='helper-text'>Additional medical history (optional)</p>", unsafe_allow_html=True)
        
        st.session_state.patient_data["activity"] = activity
        st.session_state.patient_data["smoking"] = smoking
        st.session_state.patient_data["alcohol"] = alcohol
        st.session_state.patient_data["medical_events"] = medical_events
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if step > 0:
            if st.button("← Previous Step", use_container_width=True):
                st.session_state.form_step -= 1
                st.rerun()
    
    with col3:
        if step < 3:
            if st.button("Next Step →", use_container_width=True):
                # Simple validation
                if step == 0 and not st.session_state.patient_data.get("first_name"):
                    st.error("Please enter patient's first name")
                else:
                    st.session_state.form_step += 1
                    st.rerun()
        else:
            if st.button("✓ Submit Patient Data", use_container_width=True):
                # Create patient record
                new_patient = {
                    "id": f"PAT{len(st.session_state.patients_list) + 1:03d}",
                    "name": f"{st.session_state.patient_data.get('first_name', '')} {st.session_state.patient_data.get('last_name', '')}",
                    "age": st.session_state.patient_data.get("age", 0),
                    "gender": st.session_state.patient_data.get("gender", ""),
                    "risk_score": np.random.uniform(0.2, 0.8),
                    "risk_category": "Pending",
                    "created_at": datetime.now(),
                }
                
                st.session_state.patients_list.append(new_patient)
                
                # Reset form
                st.session_state.form_step = 0
                st.session_state.patient_data = {}
                
                st.success("✓ Patient added successfully!")
                st.balloons()

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

def page_predictions():
    """Predictions dashboard"""
    st.title("🔮 Predictions & Risk Assessment")
    st.markdown("<p style='color: var(--text-secondary); font-size: 16px;'>View risk scores and predictions for patients</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.session_state.patients_list:
        # Select patient
        patient_names = [p["name"] for p in st.session_state.patients_list]
        selected_patient_idx = st.selectbox("Select Patient", range(len(patient_names)), format_func=lambda x: patient_names[x])
        patient = st.session_state.patients_list[selected_patient_idx]
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("### Risk Assessment")
            render_risk_card(patient["risk_score"], patient["name"])
        
        with col2:
            st.markdown("### Key Metrics")
            st.markdown(f"""
            <div class='card'>
                <div style='margin-bottom: 16px;'>
                    <div class='helper-text'>Patient ID</div>
                    <div style='font-size: 16px; font-weight: 600;'>{patient['id']}</div>
                </div>
                <div style='margin-bottom: 16px;'>
                    <div class='helper-text'>Age</div>
                    <div style='font-size: 16px; font-weight: 600;'>{patient['age']} years</div>
                </div>
                <div>
                    <div class='helper-text'>Last Updated</div>
                    <div style='font-size: 16px; font-weight: 600;'>{patient['created_at'].strftime('%Y-%m-%d')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Feature importance / SHAP
        st.markdown("### Feature Importance")
        
        # Generate sample feature importance data
        features = ["Baseline Factor Level", "Treatment Adherence", "Previous Bleeding", "Activity Level", "Inhibitor Status"]
        importances = np.random.uniform(0.05, 0.4, len(features))
        importances = importances / importances.sum()
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importances,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
                    line=dict(color=COLORS["border"], width=1)
                ),
                text=[f"{imp:.0%}" for imp in importances],
                textposition='outside',
            )
        ])
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], family="Arial, sans-serif"),
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=False),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Clinical Recommendations
        st.markdown("### Clinical Recommendations")
        
        if patient["risk_score"] > 0.7:
            st.warning("⚠️ **High-Risk Patient** - Intensive monitoring and frequent follow-ups recommended")
            st.markdown("""
            **Recommended Actions:**
            - Increase treatment frequency
            - Schedule regular follow-up appointments
            - Consider switching to prophylactic therapy
            - Educate patient on bleeding prevention
            """)
        elif patient["risk_score"] > 0.4:
            st.info("ℹ️ **Medium-Risk Patient** - Regular monitoring recommended")
            st.markdown("""
            **Recommended Actions:**
            - Continue current treatment plan
            - Monitor for any changes in bleeding pattern
            - Annual comprehensive assessment
            - Reinforce lifestyle modifications
            """)
        else:
            st.success("✓ **Low-Risk Patient** - Continue current management")
            st.markdown("""
            **Recommended Actions:**
            - Maintain current treatment regimen
            - Standard follow-up schedule
            - Encourage healthy lifestyle
            - Review annually
            """)
    else:
        st.info("No patients yet. Add a patient first!")

# ============================================================================
# PAGE: CHATBOT
# ============================================================================

def page_chatbot():
    """ChatGPT-style chatbot interface"""
    st.title("💬 AI Clinical Assistant")
    st.markdown("<p style='color: var(--text-secondary); font-size: 16px;'>Ask questions about hemophilia management and patient care</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Chat container with custom styling
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message ai-message'>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([20, 1])
    
    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask about hemophilia management, treatment, or patient care...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", key="send_button")
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Simulate AI response (in production, call actual AI model)
        with st.spinner("Thinking..."):
            import time
            time.sleep(1)
            
            ai_response = """
            This is a sample response about hemophilia management. In a production system, 
            this would be replaced with actual AI model predictions and clinical insights.
            
            The response would provide evidence-based recommendations for:
            - Treatment optimization
            - Patient risk assessment
            - Clinical decision support
            - Management guidelines
            """
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        st.rerun()

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

def page_analytics():
    """Analytics dashboard"""
    st.title("📈 Analytics & Insights")
    st.markdown("<p style='color: var(--text-secondary); font-size: 16px;'>Comprehensive patient analytics and trends</p>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Advanced filters
    st.markdown("### Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_age = st.slider("Age Range (Min)", 0, 100, 0)
    
    with col2:
        max_age = st.slider("Age Range (Max)", 0, 100, 100)
    
    with col3:
        selected_risk = st.multiselect("Risk Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    
    # Filter patients
    filtered_patients = [
        p for p in st.session_state.patients_list
        if min_age <= p["age"] <= max_age and get_risk_category(p["risk_score"]) in selected_risk
    ]
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Analytics charts
    if filtered_patients:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Age Distribution")
            
            ages = [p["age"] for p in filtered_patients]
            
            fig = go.Figure(data=[go.Histogram(
                x=ages,
                nbinsx=10,
                marker=dict(
                    color=COLORS["primary"],
                    line=dict(color=COLORS["border"], width=1)
                ),
                hoverinfo="x+y"
            )])
            
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"], family="Arial, sans-serif"),
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Age",
                yaxis_title="Count",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Gender Distribution")
            
            genders = [p["gender"] for p in filtered_patients]
            gender_counts = pd.Series(genders).value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=gender_counts.index,
                y=gender_counts.values,
                marker=dict(
                    color=[COLORS["primary"], COLORS["primary_light"]],
                    line=dict(color=COLORS["border"], width=1)
                ),
                text=gender_counts.values,
                textposition='outside',
            )])
            
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["text"], family="Arial, sans-serif"),
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Gender",
                yaxis_title="Count",
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk vs Age Scatter
        st.markdown("### Risk Score vs Age")
        
        fig = go.Figure(data=[go.Scatter(
            x=[p["age"] for p in filtered_patients],
            y=[p["risk_score"] for p in filtered_patients],
            mode='markers',
            marker=dict(
                size=12,
                color=[get_risk_color(p["risk_score"]) for p in filtered_patients],
                line=dict(color=COLORS["border"], width=1),
                opacity=0.7
            ),
            text=[p["name"] for p in filtered_patients],
            hovertemplate="<b>%{text}</b><br>Age: %{x}<br>Risk: %{y:.0%}<extra></extra>"
        )])
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], family="Arial, sans-serif"),
            showlegend=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Age",
            yaxis_title="Risk Score",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Patient table with sorting
        st.markdown("### Patient Details")
        
        df = pd.DataFrame(filtered_patients)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
        df["risk_score"] = df["risk_score"].apply(lambda x: f"{x:.0%}")
        
        display_df = df[["id", "name", "age", "gender", "risk_score", "risk_category"]].rename(
            columns={
                "id": "ID",
                "name": "Name",
                "age": "Age",
                "gender": "Gender",
                "risk_score": "Risk",
                "risk_category": "Category"
            }
        )
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No patients match the selected filters")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Apply dynamic theme CSS
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom: 24px;'>🏥 Medical AI</h2>", unsafe_allow_html=True)
        
        # Theme toggle
        st.divider()
        st.markdown("**Settings**")
        theme_option = st.radio(
            "Theme",
            ["☀️ Light", "🌙 Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            label_visibility="collapsed",
            horizontal=True
        )
        st.session_state.theme = "dark" if "🌙" in theme_option else "light"
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "➕ Add Patient", "🔮 Predictions", "💬 Chatbot", "📈 Analytics"],
            label_visibility="collapsed"
        )
    
    # Route to pages
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "➕ Add Patient":
        page_add_patient()
    elif page == "🔮 Predictions":
        page_predictions()
    elif page == "💬 Chatbot":
        page_chatbot()
    elif page == "📈 Analytics":
        page_analytics()

if __name__ == "__main__":
    main()
