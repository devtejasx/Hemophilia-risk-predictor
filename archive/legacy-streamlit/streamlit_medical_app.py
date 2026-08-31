"""
Medical AI System - Multi-Page Streamlit Application
Main entry point with page routing and global configuration
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import page modules
from streamlit_pages import patient_form, results_dashboard, patient_history, ai_chatbot, doctor_dashboard
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import setup_page_config, apply_custom_styling


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

setup_page_config()
apply_custom_styling()

# Initialize session state
state = StateManager()


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("## 🏥 Medical AI System")
    st.markdown("---")
    
    # Navigation menu
    selected_page = option_menu(
        menu_title="Navigation",
        options=["🔬 Patient Form", "📊 Results", "📋 History", "💬 Chatbot", "👨‍⚕️ Doctor Dashboard"],
        icons=["clipboard-pulse", "bar-chart", "table", "chat-dots", "graph-up"],
        menu_icon="house",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0a0e27"},
            "icon": {"color": "#00d4ff", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#1a1f3a",
            },
            "nav-link-selected": {"background-color": "#00d4ff", "color": "white"},
        }
    )
    
    st.markdown("---")
    
    # User info section
    st.markdown("### 👤 User Information")
    user_role = st.selectbox("Role", ["Clinician", "Patient", "Administrator"], label_visibility="collapsed")
    state.set_user_role(user_role)
    
    # Show session info
    st.markdown("---")
    st.markdown("### 📊 Session Info")
    if st.session_state.get("current_patient_id"):
        st.info(f"📌 Patient: {st.session_state.current_patient_id}")
    else:
        st.warning("ℹ️ No patient selected")
    
    # Backend status
    st.markdown("---")
    st.markdown("### 🔌 System Status")
    backend_status = state.check_backend_connection()
    if backend_status:
        st.success("✅ Backend: Connected")
    else:
        st.error("❌ Backend: Offline")


# ============================================================================
# PAGE ROUTING
# ============================================================================

page_map = {
    "🔬 Patient Form": patient_form.render,
    "📊 Results": results_dashboard.render,
    "📋 History": patient_history.render,
    "💬 Chatbot": ai_chatbot.render,
    "👨‍⚕️ Doctor Dashboard": doctor_dashboard.render,
}

# Render selected page
if selected_page in page_map:
    page_map[selected_page]()
else:
    st.error("Page not found")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.8em;'>
    <p>🏥 Medical AI Clinical Decision Support System</p>
    <p>v1.0 | Last Updated: April 2026</p>
    <p>⚠️ For Clinical Use Only - Requires Professional Review</p>
    </div>
    """,
    unsafe_allow_html=True
)
