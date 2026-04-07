"""
Session State Manager - Handles Streamlit session state and user data
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any
import requests


class StateManager:
    """Centralized session state management"""
    
    def __init__(self):
        """Initialize state manager"""
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize default session state"""
        default_keys = {
            "current_patient_id": None,
            "current_patient_data": {},
            "prediction_results": None,
            "user_role": "Clinician",
            "chat_history": [],
            "selected_date_range": (None, None),
            "filters_applied": False,
            "backend_available": True,
            "last_sync": None,
        }
        
        for key, value in default_keys.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # Patient Management
    
    def set_current_patient(self, patient_id: str, patient_data: Dict[str, Any]):
        """Set current patient in session"""
        st.session_state.current_patient_id = patient_id
        st.session_state.current_patient_data = patient_data
        st.session_state.last_sync = datetime.now().isoformat()
    
    def get_current_patient(self) -> Optional[Dict[str, Any]]:
        """Get current patient data"""
        return st.session_state.get("current_patient_data", {})
    
    def clear_current_patient(self):
        """Clear current patient"""
        st.session_state.current_patient_id = None
        st.session_state.current_patient_data = {}
    
    # Prediction Management
    
    def set_prediction_results(self, results: Dict[str, Any]):
        """Store prediction results"""
        st.session_state.prediction_results = results
        st.session_state.last_sync = datetime.now().isoformat()
    
    def get_prediction_results(self) -> Optional[Dict[str, Any]]:
        """Get last prediction results"""
        return st.session_state.get("prediction_results")
    
    def clear_predictions(self):
        """Clear prediction results"""
        st.session_state.prediction_results = None
    
    # User Management
    
    def set_user_role(self, role: str):
        """Set user role"""
        st.session_state.user_role = role
    
    def get_user_role(self) -> str:
        """Get user role"""
        return st.session_state.get("user_role", "Clinician")
    
    # Chat Management
    
    def add_chat_message(self, role: str, message: str):
        """Add message to chat history"""
        st.session_state.chat_history.append({
            "role": role,  # "user" or "assistant"
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_chat_history(self) -> list:
        """Get chat history"""
        return st.session_state.get("chat_history", [])
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
    
    # Filters & Date Range
    
    def set_date_range(self, start_date, end_date):
        """Set date range for filtering"""
        st.session_state.selected_date_range = (start_date, end_date)
        st.session_state.filters_applied = True
    
    def get_date_range(self):
        """Get selected date range"""
        return st.session_state.get("selected_date_range", (None, None))
    
    def clear_filters(self):
        """Clear all filters"""
        st.session_state.selected_date_range = (None, None)
        st.session_state.filters_applied = False
    
    # Backend Connection
    
    def check_backend_connection(self) -> bool:
        """Check if backend is available"""
        try:
            # Try to connect to FastAPI backend health endpoint
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            is_available = response.status_code == 200
            st.session_state.backend_available = is_available
            return is_available
        except (requests.ConnectionError, requests.Timeout):
            st.session_state.backend_available = False
            return False
    
    def is_backend_available(self) -> bool:
        """Get backend status"""
        return st.session_state.get("backend_available", False)
    
    # Generic Get/Set
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state"""
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in session state"""
        st.session_state[key] = value
    
    # Session Cleanup
    
    def reset_session(self):
        """Reset entire session"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._init_session_state()


# Initialize global state manager
def get_state() -> StateManager:
    """Get or create global state manager"""
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager()
    return st.session_state.state_manager
