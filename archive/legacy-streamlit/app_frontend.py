"""
HEMOPHILIA CLINICAL DECISION SUPPORT SYSTEM
Streamlit Frontend with API Integration

A modern, single-page SaaS-style dashboard connected to FastAPI backend.

Author: AI Clinical Systems
Version: 2.0
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 10

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Clinical Dashboard | Hemophilia Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
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

    /* Analytics card */
    .analytics-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
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

    .spacer-sm {
        margin-top: 1rem;
    }

    .spacer-md {
        margin-top: 2rem;
    }

    .spacer-lg {
        margin-top: 3rem;
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
        st.session_state.logged_in = False
        
    if "token" not in st.session_state:
        st.session_state.token = None
        
    if "user" not in st.session_state:
        st.session_state.user = None
        
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
        
    if "patients" not in st.session_state:
        st.session_state.patients = []
        
    if "current_patient" not in st.session_state:
        st.session_state.current_patient = None
        
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


init_session_state()

# ============================================================================
# API CLIENT FUNCTIONS
# ============================================================================

def make_api_call(method: str, endpoint: str, data=None, headers=None):
    """Make API call with error handling."""
    try:
        url = urljoin(API_BASE_URL, endpoint)
        
        if headers is None:
            headers = {}
        
        if st.session_state.token:
            headers["Authorization"] = f"Bearer {st.session_state.token}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=API_TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=API_TIMEOUT)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
        logger.error(f"Connection error: {endpoint}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.json().get('detail', 'Unknown error')}")
        logger.error(f"HTTP error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Error: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_color(score):
    """Return color based on risk score."""
    if score < 0.4:
        return "#10B981"
    elif score < 0.7:
        return "#F59E0B"
    else:
        return "#EF4444"


def get_risk_label(score):
    """Return risk label based on score."""
    if score < 0.4:
        return "LOW RISK"
    elif score < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


# ============================================================================
# AUTHENTICATION PAGES
# ============================================================================

def show_login_page():
    """Show login page."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("## 🏥 Clinical Dashboard")
        st.markdown("### Login")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            
            col_login, col_signup = st.columns(2)
            
            with col_login:
                if st.form_submit_button("Login", use_container_width=True):
                    if email and password:
                        result = make_api_call("POST", "/api/auth/login", {
                            "email": email,
                            "password": password,
                        })
                        
                        if result:
                            st.session_state.token = result["access_token"]
                            st.session_state.user = result["user"]
                            st.session_state.logged_in = True
                            st.success("✅ Login successful!")
                            st.rerun()
                    else:
                        st.error("Please enter email and password")
            
            with col_signup:
                if st.form_submit_button("Register", use_container_width=True):
                    st.session_state.show_register = True
        
        if "show_register" in st.session_state and st.session_state.show_register:
            st.divider()
            
            with st.form("register_form"):
                st.markdown("### Create Account")
                
                username = st.text_input("Username", placeholder="your_username")
                reg_email = st.text_input("Email", placeholder="your@email.com", key="reg_email")
                full_name = st.text_input("Full Name", placeholder="Your Name")
                reg_password = st.text_input("Password", type="password", placeholder="min 8 characters", key="reg_password")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    if username and reg_email and full_name and reg_password:
                        result = make_api_call("POST", "/api/auth/register", {
                            "username": username,
                            "email": reg_email,
                            "full_name": full_name,
                            "password": reg_password,
                        })
                        
                        if result:
                            st.session_state.token = result["access_token"]
                            st.session_state.user = result["user"]
                            st.session_state.logged_in = True
                            st.success("✅ Account created!")
                            st.rerun()
                    else:
                        st.error("Please fill all fields")


# ============================================================================
# MAIN DASHBOARD COMPONENTS
# ============================================================================

def show_sidebar():
    """Display the sidebar."""
    with st.sidebar:
        st.markdown("### 🏥 HEMOPHILIA AI")
        st.markdown("---")
        
        if st.session_state.user:
            st.markdown(f"**👤 {st.session_state.user['full_name']}**")
            st.markdown(st.session_state.user['email'])
        
        st.markdown("---")
        
        if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
        
        st.markdown("---")
        
        if st.button("🚪 Logout", key="logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()


def show_header():
    """Display the header section."""
    st.markdown(f"""
    <div class="header-container">
        <div class="header-content">
            <div>
                <h1>Clinical Dashboard</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">Hemophilia Clinical Decision Support</p>
            </div>
            <div style="text-align: right;">
                <p style="color: rgba(255,255,255,0.9); margin: 0;">
                    <strong>{st.session_state.user['full_name'] if st.session_state.user else 'User'}</strong><br/>
                    <small style="color: rgba(255,255,255,0.7);">{datetime.now().strftime('%A, %B %d, %Y')}</small>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_kpis():
    """Display KPI cards section."""
    st.markdown('<h2 class="section-title">📈 Overview</h2>', unsafe_allow_html=True)
    
    # Fetch analytics
    analytics = make_api_call("GET", "/api/analytics")
    
    if analytics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Patients</div>
                <div class="kpi-value">{analytics['total_patients']}</div>
                <div class="kpi-change">↑ Active monitoring</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card danger">
                <div class="kpi-label">High Risk</div>
                <div class="kpi-value">{analytics['high_risk_count']}</div>
                <div class="kpi-change negative">⚠️ Needs attention</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card warning">
                <div class="kpi-label">Avg Risk Score</div>
                <div class="kpi-value">{analytics['avg_risk_score']:.2f}</div>
                <div class="kpi-change">Assessment metric</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-card success">
                <div class="kpi-label">Active Cases</div>
                <div class="kpi-value">{analytics['active_cases']}</div>
                <div class="kpi-change">Monitoring</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_patient_form():
    """Display patient information form."""
    st.markdown('<h2 class="section-title">👤 Patient Information</h2>', unsafe_allow_html=True)
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        with st.container():
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Basic Information</div>', unsafe_allow_html=True)
            
            patient_name = st.text_input("Patient Name", placeholder="Enter full name")
            
            col_a, col_b = st.columns(2)
            with col_a:
                patient_age = st.number_input("Age", min_value=0, max_value=120, value=35)
            with col_b:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Clinical Parameters</div>', unsafe_allow_html=True)
            
            clotting_factor = st.slider("Clotting Factor Level (%)", 0, 100, 65)
            previous_bleeds = st.slider("Previous Bleeds (6 months)", 0, 20, 5)
            
            col_c, col_d = st.columns(2)
            with col_c:
                activity_level = st.slider("Activity Level", 1, 10, 5)
            with col_d:
                compliance = st.slider("Medication Compliance", 0.0, 1.0, 0.75)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Treatment</div>', unsafe_allow_html=True)
            
            treatment_type = st.selectbox("Treatment Type", 
                ["Factor VIII", "Factor IX", "Emicizumab", "Other"])
            
            notes = st.text_area("Clinical Notes", placeholder="Enter additional notes")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("➕ Add Patient", use_container_width=True):
                if patient_name:
                    patient_data = {
                        "name": patient_name,
                        "age": patient_age,
                        "gender": gender,
                        "clotting_factor": clotting_factor,
                        "previous_bleeds": previous_bleeds,
                        "activity_level": activity_level,
                        "medication_compliance": compliance,
                        "treatment_type": treatment_type,
                        "notes": notes,
                    }
                    
                    result = make_api_call("POST", "/api/patients", patient_data)
                    if result:
                        st.success(f"✅ Patient {patient_name} added successfully!")
                        # Fetch patients again
                        patients_result = make_api_call("GET", "/api/patients")
                        if patients_result:
                            st.session_state.patients = patients_result
                        st.rerun()
                else:
                    st.error("Please enter patient name")
    
    with right_col:
        st.markdown('<h3>Recent Patients</h3>', unsafe_allow_html=True)
        
        # Fetch patients
        if not st.session_state.patients:
            patients_result = make_api_call("GET", "/api/patients")
            if patients_result:
                st.session_state.patients = patients_result
        
        if st.session_state.patients:
            for patient in st.session_state.patients[:5]:
                st.markdown(f"""
                <div class="analytics-card" style="margin-bottom: 1rem;">
                    <strong>{patient['name']}</strong><br/>
                    <small>Age: {patient['age']} | Treatment: {patient['treatment_type']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No patients added yet. Add one to get started!")
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)


def show_chatbot_section():
    """Display chatbot UI."""
    st.markdown('<h2 class="section-title">💬 Clinical AI Assistant</h2>', unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Fetch chat history
        chat_result = make_api_call("GET", "/api/chat-history")
        if chat_result and chat_result.get("messages"):
            for msg in chat_result["messages"]:
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
        else:
            st.markdown(f"""
            <div class="chat-message ai">
                <div class="chat-bubble">Hello! I'm your clinical AI assistant. How can I help you today?</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_message = st.text_input(
            "Send a message...",
            placeholder="Ask about patient care, protocols, or results...",
            key="chat_input"
        )
        
        if user_message:
            result = make_api_call("POST", "/api/chat", {
                "role": "user",
                "message": user_message,
            })
            if result:
                st.rerun()
    
    with right_col:
        st.markdown('<h3>Common Topics</h3>', unsafe_allow_html=True)
        
        topics = [
            ("🩺 Patient Care", "Guidelines and best practices"),
            ("📊 Data Analysis", "Interpret results and trends"),
            ("💊 Treatments", "Management options"),
        ]
        
        for emoji_title, desc in topics:
            st.markdown(f"""
            <div class="analytics-card" style="margin-bottom: 1rem;">
                <strong>{emoji_title}</strong><br/>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)


def show_footer():
    """Display footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>🏥 Hemophilia Clinical Decision Support System | Version 2.0<br/>
        © 2024 Clinical AI Systems. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    if not st.session_state.logged_in:
        show_login_page()
    else:
        # Show sidebar
        show_sidebar()
        
        # Main content
        with st.container():
            # Header
            show_header()
            
            # Main sections
            show_kpis()
            show_patient_form()
            show_chatbot_section()
            show_footer()


if __name__ == "__main__":
    main()
