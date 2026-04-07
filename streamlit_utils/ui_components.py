"""
UI Components - Reusable Streamlit components and styling
"""

import streamlit as st
from contextlib import contextmanager
import plotly.graph_objects as go
from typing import Optional, List, Dict


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Medical AI System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://docs.streamlit.io",
            "Report a bug": "https://github.com",
            "About": "Medical AI Clinical Decision Support System v1.0"
        }
    )


def apply_custom_styling():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
        /* Main background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0d1428 0%, #1a1f3a 50%, #0a0e27 100%);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
            border-right: 1px solid #00d4ff;
        }
        
        /* Metric styling */
        [data-testid="metric-container"] {
            background-color: rgba(0, 212, 255, 0.05);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        /* Button styling */
        button {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: white !important;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        button:hover {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
            transform: translateY(-2px);
        }
        
        /* Input styling */
        input, select, textarea {
            background-color: #1a1f3a !important;
            color: #e0e6ff !important;
            border: 1px solid #00d4ff !important;
            border-radius: 6px !important;
        }
        
        /* Text styling */
        h1, h2, h3, h4, h5, h6 {
            color: #e0e6ff !important;
        }
        
        /* Success/Error boxes */
        .stSuccess {
            background-color: rgba(0, 200, 80, 0.1);
            color: #00c850;
        }
        
        .stError {
            background-color: rgba(255, 100, 100, 0.1);
            color: #ff6464;
        }
        
        .stWarning {
            background-color: rgba(255, 200, 0, 0.1);
            color: #ffc800;
        }
        
        /* Divider */
        hr {
            border-color: #00d4ff !important;
            opacity: 0.3;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# LOADING INDICATORS
# ============================================================================

@contextmanager
def loading_spinner(text: str = "Loading..."):
    """Context manager for loading spinner"""
    placeholder = st.empty()
    with placeholder.container():
        with st.spinner(text):
            yield


def show_loading_bar(progress: float, label: str = "Processing"):
    """Show progress bar"""
    st.progress(progress, text=label)


def show_status_indicator(status: str, message: str = ""):
    """Show status indicator"""
    status_config = {
        "success": ("✅", "green"),
        "error": ("❌", "red"),
        "warning": ("⚠️", "orange"),
        "info": ("ℹ️", "blue"),
        "loading": ("⏳", "blue"),
    }
    
    icon, color = status_config.get(status, ("❓", "gray"))
    st.markdown(f"<p style='color: {color}; font-size: 1.2em;'>{icon} {message}</p>", unsafe_allow_html=True)


# ============================================================================
# METRIC CARDS
# ============================================================================

def metric_card(title: str, value: str, delta: Optional[str] = None, icon: str = "📊"):
    """Create a styled metric card"""
    col = st.columns(1)[0]
    with col:
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 150, 180, 0.05) 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.3);
            text-align: center;
        '>
            <p style='font-size: 2em; margin: 0;'>{icon}</p>
            <p style='font-size: 0.9em; color: #888; margin: 0.5rem 0 0;'>{title}</p>
            <p style='font-size: 1.8em; font-weight: bold; color: #00d4ff; margin: 0.5rem 0;'>{value}</p>
            {f"<p style='font-size: 0.8em; color: #0f0;'>↑ {delta}</p>" if delta else ""}
        </div>
        """, unsafe_allow_html=True)


def risk_card(risk_score: float, risk_level: str = ""):
    """Create a risk score card"""
    if not risk_level:
        if risk_score > 0.75:
            risk_level = "🔴 CRITICAL"
            color = "#ff3333"
        elif risk_score > 0.6:
            risk_level = "🟠 HIGH"
            color = "#ff9900"
        elif risk_score > 0.4:
            risk_level = "🟡 MODERATE"
            color = "#ffcc00"
        else:
            risk_level = "🟢 LOW"
            color = "#00cc33"
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, rgba(255, 212, 0, 0.1) 0%, rgba(255, 100, 0, 0.05) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid {color};
        text-align: center;
    '>
        <p style='font-size: 2.5em; font-weight: bold; color: {color}; margin: 0;'>{risk_score:.0%}</p>
        <p style='font-size: 1.2em; color: {color}; margin: 0.5rem 0;'>{risk_level}</p>
        <p style='font-size: 0.9em; color: #aaa; margin: 0;'>Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA DISPLAY COMPONENTS
# ============================================================================

def display_patient_info(patient_data: Dict) -> None:
    """Display patient information in a formatted way"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Age", f"{patient_data.get('age', 'N/A')} years")
    with col2:
        st.metric("⚕️ Diagnosis", patient_data.get('diagnosis', 'N/A'))
    with col3:
        st.metric("📊 Severity", patient_data.get('severity', 'N/A'))
    with col4:
        st.metric("📅 Last Visit", patient_data.get('last_visit', 'N/A'))


def display_table_with_pagination(data, page_size: int = 10):
    """Display dataframe with pagination"""
    if len(data) == 0:
        st.warning("No data to display")
        return
    
    total_pages = (len(data) + page_size - 1) // page_size
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(data.iloc[start_idx:end_idx], use_container_width=True)
    
    st.caption(f"Showing {start_idx + 1} to {min(end_idx, len(data))} of {len(data)} records")


# ============================================================================
# CHART HELPERS
# ============================================================================

def empty_placeholder_chart(title: str = "No data available"):
    """Show empty state for charts"""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="#888")
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        plot_bgcolor="#0a0e27",
        paper_bgcolor="#0a0e27",
        height=400
    )
    return fig


def get_plotly_theme_config():
    """Get Plotly theme configuration"""
    return {
        "template": "plotly_dark",
        "font": {"family": "Arial, sans-serif", "color": "#e0e6ff"},
        "plot_bgcolor": "#0a0e27",
        "paper_bgcolor": "#1a1f3a",
        "xaxis": {"gridcolor": "#2a2f4a", "showgrid": True},
        "yaxis": {"gridcolor": "#2a2f4a", "showgrid": True},
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
    }


# ============================================================================
# FORM HELPERS
# ============================================================================

def form_section(title: str, icon: str = "📋"):
    """Create a section header for forms"""
    st.markdown(f"### {icon} {title}")
    st.divider()


def required_input(label: str, key: str, input_type: str = "text", **kwargs):
    """Create a required input field with validation"""
    st.markdown(f"**{label}** <span style='color: red;'>*</span>", unsafe_allow_html=True)
    
    if input_type == "text":
        return st.text_input(label, key=key, label_visibility="collapsed", **kwargs)
    elif input_type == "number":
        return st.number_input(label, key=key, label_visibility="collapsed", **kwargs)
    elif input_type == "select":
        return st.selectbox(label, key=key, label_visibility="collapsed", **kwargs)
    elif input_type == "multiselect":
        return st.multiselect(label, key=key, label_visibility="collapsed", **kwargs)
    elif input_type == "slider":
        return st.slider(label, key=key, label_visibility="collapsed", **kwargs)


# ============================================================================
# LAYOUT HELPERS
# ============================================================================

def create_header(title: str, subtitle: str = ""):
    """Create a page header"""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"_{subtitle}_")
    st.divider()


def create_two_column_layout():
    """Create a two-column layout"""
    return st.columns(2)


def create_three_column_layout():
    """Create a three-column layout"""
    return st.columns(3)


def create_tabs(tab_titles: List[str]):
    """Create tabs"""
    return st.tabs(tab_titles)


# ============================================================================
# MESSAGE DISPLAY
# ============================================================================

def show_success(message: str, icon: str = "✅"):
    """Show success message"""
    st.success(f"{icon} {message}")


def show_error(message: str, icon: str = "❌"):
    """Show error message"""
    st.error(f"{icon} {message}")


def show_warning(message: str, icon: str = "⚠️"):
    """Show warning message"""
    st.warning(f"{icon} {message}")


def show_info(message: str, icon: str = "ℹ️"):
    """Show info message"""
    st.info(f"{icon} {message}")
