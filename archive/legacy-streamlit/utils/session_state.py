"""
Session state management for the Streamlit app
Handles all global state across pages
"""

import streamlit as st
from typing import Any, Dict
from datetime import datetime


def init_session_state() -> None:
    """Initialize all session state variables"""
    
    defaults = {
        # ============= AUTHENTICATION =============
        "authenticated": False,
        "user_id": None,
        "user_name": "Guest",
        "user_role": "clinician",
        
        # ============= PATIENT DATA =============
        "current_patient": None,
        "patient_form_data": {},
        "patient_list": [],
        "selected_patient_id": None,
        
        # ============= PREDICTION DATA =============
        "last_prediction": None,
        "prediction_history": [],
        "shap_explanation": None,
        
        # ============= CHATBOT STATE =============
        "chat_history": [],
        "chat_mode": "clinical_assistant",
        
        # ============= UI STATE =============
        "theme": "dark",
        "show_advanced": False,
        "sidebar_collapsed": False,
        "shap_view": "Basic View",
        
        # ============= NOTIFICATIONS =============
        "notifications": [],
        "last_notification_time": None,
        
        # ============= ANALYTICS =============
        "analytics_data": {},
        "session_start_time": datetime.now(),
        
        # ============= FILTERS =============
        "active_filters": {
            "severity_range": [0, 100],
            "risk_range": [0, 100],
            "mutations": [],
            "risk_level": "all",
            "date_range": None
        },
        
        # ============= DISPLAY OPTIONS =============
        "view_mode": "grid",
        "items_per_page": 10,
        "current_page": 1,
        "selected_patients": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_var(key: str, default: Any = None) -> Any:
    """Safely get session variable"""
    return st.session_state.get(key, default)


def set_session_var(key: str, value: Any) -> None:
    """Set session variable"""
    st.session_state[key] = value


def clear_session_data() -> None:
    """Clear patient data from session"""
    st.session_state.current_patient = None
    st.session_state.patient_form_data = {}
    st.session_state.last_prediction = None
    st.session_state.shap_explanation = None


def update_patient_data(data: Dict[str, Any]) -> None:
    """Update patient data in session"""
    st.session_state.patient_data = data


def update_prediction_result(result: Dict[str, Any]) -> None:
    """Update prediction result in session"""
    st.session_state.prediction_result = result


def toggle_theme() -> None:
    """Toggle between dark and light theme"""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def add_chat_message(role: str, content: str) -> None:
    """Add message to chat history"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })


def clear_chat_history() -> None:
    """Clear chat history"""
    st.session_state.chat_history = []


def set_shap_view(view: str) -> None:
    """Set SHAP visualization view (Basic/Advanced)"""
    if view in ["Basic View", "Advanced View"]:
        st.session_state.shap_view = view
