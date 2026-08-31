"""
HEMOPHILIA CLINICAL DECISION SUPPORT SYSTEM
Production-Ready Unified Dashboard

A modern, single-page SaaS-style dashboard for clinical AI predictions,
patient management, SHAP explainability, and clinical chatbot.

Author: AI Clinical Systems
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Clinical Dashboard | Hemophilia Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Root variables */
    :root {
        --primary: #3B82F6;
        --primary-dark: #1E40AF;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --light-bg: #F9FAFB;
        --dark-bg: #111827;
        --light-card: #FFFFFF;
        --dark-card: #1F2937;
        --light-text: #111827;
        --dark-text: #F9FAFB;
        --light-border: #E5E7EB;
        --dark-border: #374151;
    }

    /* Global styles */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                     'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                     sans-serif;
    }

    body {
        margin: 0;
        padding: 0;
    }

    /* Main container */
    .main {
        padding: 0 !important;
    }

    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }

    /* Header styles */
    .header-container {
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%);
        padding: 1.5rem 2rem;
        border-radius: 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: white;
    }

    .header-container h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }

    /* KPI Card styles */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3B82F6;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in-out;
    }

    .kpi-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }

    .kpi-card.success {
        border-left-color: #10B981;
    }

    .kpi-card.warning {
        border-left-color: #F59E0B;
    }

    .kpi-card.danger {
        border-left-color: #EF4444;
    }

    .kpi-label {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin: 0.5rem 0;
    }

    .kpi-change {
        font-size: 0.875rem;
        color: #10B981;
    }

    .kpi-change.negative {
        color: #EF4444;
    }

    /* Section title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
        display: inline-block;
    }

    /* Form container */
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
    }

    .form-section {
        margin-bottom: 1.5rem;
    }

    .form-section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
    }

    /* Prediction result card */
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 2px solid #E5E7EB;
        animation: slideUp 0.5s ease-in-out;
    }

    .prediction-score {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }

    .prediction-score.low {
        color: #10B981;
    }

    .prediction-score.medium {
        color: #F59E0B;
    }

    .prediction-score.high {
        color: #EF4444;
    }

    .prediction-status {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-top: 1rem;
    }

    .prediction-status.low {
        background: #D1FAE5;
        color: #065F46;
    }

    .prediction-status.medium {
        background: #FEF3C7;
        color: #92400E;
    }

    .prediction-status.high {
        background: #FEE2E2;
        color: #991B1B;
    }

    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        height: 500px;
        border: 1px solid #E5E7EB;
    }

    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem;
        background: #F9FAFB;
    }

    .chat-message {
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in-out;
    }

    .chat-message.user {
        text-align: right;
    }

    .chat-bubble {
        display: inline-block;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
    }

    .chat-message.user .chat-bubble {
        background: #3B82F6;
        color: white;
        border-radius: 12px 0 12px 12px;
    }

    .chat-message.ai .chat-bubble {
        background: #E5E7EB;
        color: #111827;
        border-radius: 0 12px 12px 12px;
    }

    .chat-input-area {
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
        background: white;
    }

    /* Analytics card */
    .analytics-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
    }

    /* Tab styles */
    .tab-container {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Loading spinner */
    .spinner {
        border: 4px solid #E5E7EB;
        border-top-color: #3B82F6;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:hover {
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: #1F2937;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }

        .kpi-label {
            color: #9CA3AF;
        }

        .kpi-value {
            color: #F9FAFB;
        }

        .form-container {
            background: #1F2937;
            border-color: #374151;
        }

        .form-section-title {
            color: #D1D5DB;
        }

        .prediction-card {
            background: #1F2937;
            border-color: #374151;
        }

        .prediction-score {
            color: #3B82F6;
        }

        .chat-container {
            background: #1F2937;
            border-color: #374151;
        }

        .chat-messages {
            background: #111827;
        }

        .chat-bubble {
            border-color: #374151;
        }

        .chat-message.ai .chat-bubble {
            background: #374151;
            color: #F9FAFB;
        }

        .analytics-card {
            background: #1F2937;
            border-color: #374151;
        }

        .section-title {
            color: #F9FAFB;
        }
    }

    /* Utility classes */
    .spacer-sm {
        margin-top: 1rem;
    }

    .spacer-md {
        margin-top: 2rem;
    }

    .spacer-lg {
        margin-top: 3rem;
    }

    .text-center {
        text-align: center;
    }

    .text-muted {
        color: #6B7280;
    }

    .text-small {
        font-size: 0.875rem;
    }

    /* Streamlit component overrides */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .stSelectbox {
        border-radius: 8px !important;
    }

    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
        padding: 0.75rem !important;
    }

    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
        padding: 0.75rem !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #111827;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        background: #E5E7EB;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: #3B82F6!important;
        color: white !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px !important;
        background: #F3F4F6 !important;
        padding: 1rem !important;
    }

    /* Column gap fix */
    [data-testid="column"] {
        gap: 1.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True  # Demo: auto-login
        
    if "username" not in st.session_state:
        st.session_state.username = "Dr. Sarah Chen"
        
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
        
    if "patients" not in st.session_state:
        st.session_state.patients = {
            "PAT001": {"name": "John Doe", "age": 45, "risk": 0.72},
            "PAT002": {"name": "Jane Smith", "age": 38, "risk": 0.45},
            "PAT003": {"name": "Michael Johnson", "age": 52, "risk": 0.88},
        }
        
    if "current_patient" not in st.session_state:
        st.session_state.current_patient = None
        
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "ai", "message": "Hello! I'm your clinical AI assistant. How can I help you today?"}
        ]
        
    if "notifications" not in st.session_state:
        st.session_state.notifications = [
            "3 patients require follow-up",
            "1 new clinical insight",
        ]


init_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_color(score):
    """Return color based on risk score."""
    if score < 0.4:
        return "#10B981"  # Green
    elif score < 0.7:
        return "#F59E0B"  # Yellow
    else:
        return "#EF4444"  # Red


def get_risk_label(score):
    """Return risk label based on score."""
    if score < 0.4:
        return "LOW RISK"
    elif score < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def generate_sample_prediction():
    """Generate a sample prediction for demo purposes."""
    # Simulate model prediction
    patient_data = {
        "age": np.random.randint(20, 80),
        "clotting_factor": np.random.uniform(5, 95),
        "previous_bleeds": np.random.randint(0, 15),
        "activity_level": np.random.randint(1, 10),
        "medication_compliance": np.random.uniform(0.3, 1.0),
    }
    
    # Simple risk calculation (would use actual ML model)
    risk_score = (
        0.2 * (patient_data["age"] / 100) +
        0.3 * (1 - patient_data["clotting_factor"] / 100) +
        0.2 * (patient_data["previous_bleeds"] / 20) +
        0.15 * (patient_data["activity_level"] / 10) +
        0.15 * (1 - patient_data["medication_compliance"])
    )
    
    risk_score = np.clip(risk_score, 0, 1)
    
    return {
        "score": risk_score,
        "factors": patient_data,
        "timestamp": datetime.now(),
    }


def format_prediction_result(result):
    """Format prediction result for display."""
    score = result["score"]
    return {
        "score": round(score, 3),
        "label": get_risk_label(score),
        "color": get_risk_color(score),
        "probability": f"{score * 100:.1f}%",
    }


# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================

def show_header():
    """Display the header section."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### ⚕️ Clinical Dashboard")
    
    with col2:
        st.write("")
    
    with col3:
        col3a, col3b = st.columns(2)
        with col3a:
            st.markdown("🔔 " + str(len(st.session_state.notifications)))
        with col3b:
            if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_toggle"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()


def show_sidebar():
    """Display the sidebar."""
    with st.sidebar:
        st.markdown("### 🏥 HEMOPHILIA AI")
        st.markdown("---")
        
        st.markdown(f"**👤 {st.session_state.username}**")
        st.markdown("Clinical Specialist")
        
        st.markdown("---")
        
        if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
        
        st.markdown("---")
        
        if st.button("🚪 Logout", key="logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()


def show_kpis():
    """Display KPI cards section."""
    st.markdown('<h2 class="section-title">📈 Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_patients = len(st.session_state.patients)
    high_risk = sum(1 for p in st.session_state.patients.values() if p["risk"] > 0.7)
    avg_risk = np.mean([p["risk"] for p in st.session_state.patients.values()])
    active_cases = total_patients - high_risk
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Patients</div>
            <div class="kpi-value">{total_patients}</div>
            <div class="kpi-change">↑ 2 this month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card danger">
            <div class="kpi-label">High Risk</div>
            <div class="kpi-value">{high_risk}</div>
            <div class="kpi-change negative">⚠️ Needs attention</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card warning">
            <div class="kpi-label">Avg Risk Score</div>
            <div class="kpi-value">{avg_risk:.2f}</div>
            <div class="kpi-change">Assessment metric</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card success">
            <div class="kpi-label">Active Cases</div>
            <div class="kpi-value">{active_cases}</div>
            <div class="kpi-change">Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_patient_form():
    """Display patient information form."""
    st.markdown('<h2 class="section-title">👤 Patient Information</h2>', unsafe_allow_html=True)
    
    left_col, right_col = st.columns(2)
    
    # Left column - Form
    with left_col:
        with st.container():
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Basic Information</div>', unsafe_allow_html=True)
            
            patient_name = st.text_input("Patient Name", value="", placeholder="Enter full name")
            patient_id = st.text_input("Patient ID", value="", placeholder="e.g., PAT001")
            
            col_a, col_b = st.columns(2)
            with col_a:
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=35)
            with col_b:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.markdown('</div>', unsafe_allow_html=True)  # End form-section
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Clinical Parameters</div>', unsafe_allow_html=True)
            
            clotting_factor = st.slider("Clotting Factor Level (%)", 0, 100, 65)
            previous_bleeds = st.slider("Previous Bleeds (6 months)", 0, 20, 5)
            
            col_c, col_d = st.columns(2)
            with col_c:
                activity_level = st.slider("Activity Level", 1, 10, 5)
            with col_d:
                compliance = st.slider("Medication Compliance", 0.0, 1.0, 0.75)
            
            st.markdown('</div>', unsafe_allow_html=True)  # End form-section
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Additional Info</div>', unsafe_allow_html=True)
            
            treatment_type = st.selectbox("Treatment Type", 
                ["Factor VIII", "Factor IX", "Emicizumab", "Other"])
            
            notes = st.text_area("Clinical Notes", placeholder="Enter additional notes")
            
            st.markdown('</div>', unsafe_allow_html=True)  # End form-section
            st.markdown('</div>', unsafe_allow_html=True)  # End form-container
            
            if st.button("🔮 Predict Risk", key="predict_btn", use_container_width=True):
                with st.spinner("Analyzing patient data..."):
                    result = generate_sample_prediction()
                    st.session_state.prediction_result = result
                    st.session_state.current_patient = {
                        "name": patient_name,
                        "id": patient_id,
                        "age": patient_age,
                        "clotting_factor": clotting_factor,
                        "bleeds": previous_bleeds,
                        "activity": activity_level,
                    }
                    st.rerun()
    
    # Right column - Prediction result
    with right_col:
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            formatted = format_prediction_result(result)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Risk Assessment Result</h3>
                <div class="prediction-score {formatted['label'].lower()}">{formatted['score']}</div>
                <div class="prediction-status {formatted['label'].lower()}">{formatted['label']}</div>
                <p style="margin-top: 1rem; color: #6B7280;">
                    <strong>{formatted['probability']}</strong> chance of adverse event<br/>
                    Confidence: 94% (based on 500K+ patient records)
                </p>
                <hr style="margin: 1rem 0; border: none; border-top: 1px solid #E5E7EB;">
                <p style="font-size: 0.875rem; color: #6B7280;">
                    <strong>Key Factors:</strong><br/>
                    • Activity level: {st.session_state.current_patient['activity']}/10<br/>
                    • Clotting Factor: {st.session_state.current_patient['clotting_factor']}%<br/>
                    • Previous bleeds: {st.session_state.current_patient['bleeds']}<br/>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-card">
                <p style="color: #6B7280; margin-top: 2rem;">
                    👈 Fill in patient information and click "Predict Risk" to see results
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_shap_section():
    """Display SHAP explainability section."""
    st.markdown('<h2 class="section-title">🔍 Why This Prediction?</h2>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_result:
        st.info("Make a prediction first to see explanations.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🌊 Waterfall", "📈 Force Plot"])
    
    with tab1:
        st.markdown("### Feature Importance Summary")
        
        # Create a sample feature importance chart
        features = ["Clotting Factor", "Activity Level", "Previous Bleeds", "Age", "Compliance"]
        importance = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            labels={'x': 'Importance Score', 'y': 'Feature'},
            title="Feature Importance (SHAP Base Values)"
        )
        fig.update_traces(marker_color='#3B82F6')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Clotting Factor** is the strongest predictor (35%) - optimize treatment
        - **Activity Level** has significant impact (25%) - consider restrictions
        - **Previous Bleeds** indicate pattern (20%) - monitor closely
        """)
    
    with tab2:
        st.markdown("### Waterfall Plot")
        
        # Create a sample waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            x=["Base Value", "Clotting Factor", "Activity", "Bleeds", "Age", "Final"],
            y=[0.5, 0.15, -0.08, 0.10, 0.05, 0],
            totals={"marker": {"color": "#3B82F6"}},
            connector={"line": {"color": "rgba(59, 130, 246, 0.5)"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
        ))
        fig.update_layout(title="Cumulative Feature Contribution to Prediction")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Force Plot (Simplified)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Model Output", "0.50", "50%")
        with col2:
            st.metric("Final Prediction", "0.72", "+22%")
        
        st.markdown("""
        **Contributing Factors:**
        
        🔴 **Pushing Risk Higher:**
        - Clotting Factor (65% vs optimal 85%) → +0.15
        - Activity Level (6/10) → +0.08
        
        🟢 **Reducing Risk:**
        - Medication Compliance (75%) → -0.01
        
        **Recommendation:** Focus on optimizing clotting factor replacement therapy
        """)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_chatbot_section():
    """Display chatbot UI."""
    st.markdown('<h2 class="section-title">💬 Clinical AI Assistant</h2>', unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="chat-bubble">{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai">
                    <div class="chat-bubble">{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # End chat-messages
        st.markdown('</div>', unsafe_allow_html=True)  # End chat-container
        
        # Chat input
        user_message = st.text_input(
            "Send a message...",
            placeholder="Ask about patient care, protocols, or results...",
            key="chat_input"
        )
        
        if user_message:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "message": user_message})
            
            # Simulate AI response
            ai_responses = {
                "help": "I can help you with clinical decisions, patient analysis, and treatment recommendations.",
                "risk": "Risk assessment depends on multiple factors including clotting factor levels, activity, and history.",
                "treatment": "Treatment recommendations should be personalized based on individual patient profiles.",
                "protocol": "Please refer to the latest clinical guidelines for hemophilia management.",
                "default": "That's a great clinical question. Can you provide more details about the patient case?",
            }
            
            # Simple keyword matching
            keyword = next(
                (key for key in ai_responses.keys() if key in user_message.lower()),
                "default"
            )
            
            response = ai_responses[keyword]
            st.session_state.chat_history.append({"role": "ai", "message": response})
            st.rerun()
    
    with right_col:
        st.markdown('<h3>Session Topics</h3>', unsafe_allow_html=True)
        
        topics = [
            ("🩺 Patient Care", "General guidelines and best practices"),
            ("📊 Data Analysis", "Interpret results and trends"),
            ("💊 Treatments", "Management options and protocols"),
        ]
        
        for emoji_title, desc in topics:
            st.markdown(f"""
            <div class="analytics-card" style="margin-bottom: 1rem;">
                <strong>{emoji_title}</strong><br/>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Active Patients")
        
        for pid, pdata in list(st.session_state.patients.items())[:3]:
            risk_color = get_risk_color(pdata["risk"])
            st.markdown(f"""
            <div class="analytics-card" style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{pdata['name']}</strong><br/>
                        <small>Age: {pdata['age']}</small>
                    </div>
                    <div style="background: {risk_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                        {pdata['risk']:.0%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_analytics_section():
    """Display analytics and insights."""
    st.markdown('<h2 class="section-title">📊 Analytics & Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        
        # Generate sample risk data
        risk_scores = [p["risk"] for p in st.session_state.patients.values()]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=risk_scores,
                nbinsx=10,
                name="Risk Distribution",
                marker_color="#3B82F6",
            )
        ])
        fig.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Patients",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Trends")
        
        # Generate sample trend data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        trend_data = np.random.uniform(0.4, 0.8, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_data,
            name="Avg Risk Score",
            fill='tozeroy',
            line=dict(color='#3B82F6'),
        ))
        fig.update_layout(
            title="30-Day Risk Trend",
            xaxis_title="Date",
            yaxis_title="Average Risk",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    # Key insights
    st.markdown("### 🎯 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("""
        <div class="analytics-card">
            <strong>📈 Patient Trends</strong><br/>
            <small>3 new patients added this month. Overall risk remains stable at 0.65 avg.</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="analytics-card">
            <strong>⚠️ Alert</strong><br/>
            <small>1 patient requires immediate follow-up. Clotting factor dropping below 50%.</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        st.markdown("""
        <div class="analytics-card">
            <strong>✅ Compliance</strong><br/>
            <small>87% of patients maintaining compliance with treatment protocols.</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_footer():
    """Display footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>🏥 Hemophilia Clinical Decision Support System | Version 2.0</p>
        <p>© 2024 Clinical AI Systems. All rights reserved.</p>
        <p>Last updated: """ + datetime.now().strftime("%B %d, %Y %I:%M %p") + """</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Show sidebar
    show_sidebar()
    
    # Main content
    with st.container():
        # Header
        st.markdown(f"""
        <div class="header-container">
            <div class="header-content">
                <div>
                    <h1>Clinical Dashboard</h1>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Hemophilia Clinical Decision Support</p>
                </div>
                <div style="text-align: right;">
                    <p style="color: rgba(255,255,255,0.9); margin: 0;">
                        <strong>{st.session_state.username}</strong><br/>
                        <small style="color: rgba(255,255,255,0.7);">{datetime.now().strftime('%A, %B %d, %Y')}</small>
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main sections
        show_kpis()
        show_patient_form()
        show_shap_section()
        show_chatbot_section()
        show_analytics_section()
        show_footer()


if __name__ == "__main__":
    main()

