"""
Session state management for Streamlit application.
Handles initialization and access to all session variables.
"""

import streamlit as st
from typing import Any, Optional, Dict, List
from constants import (
    SESSION_LOGGED_IN, SESSION_USER, SESSION_TOKEN,
    SESSION_DARK_MODE, SESSION_PATIENTS, SESSION_PREDICTION,
    SESSION_CHAT_HISTORY
)


def init_session_state() -> None:
    """Initialize all session state variables on app startup."""
    
    # Authentication state
    if SESSION_LOGGED_IN not in st.session_state:
        st.session_state[SESSION_LOGGED_IN] = False
    
    if SESSION_USER not in st.session_state:
        st.session_state[SESSION_USER] = None
    
    if SESSION_TOKEN not in st.session_state:
        st.session_state[SESSION_TOKEN] = None
    
    # UI state
    if SESSION_DARK_MODE not in st.session_state:
        st.session_state[SESSION_DARK_MODE] = False
    
    # Data state
    if SESSION_PATIENTS not in st.session_state:
        st.session_state[SESSION_PATIENTS] = []
    
    if SESSION_PREDICTION not in st.session_state:
        st.session_state[SESSION_PREDICTION] = None
    
    # Chat state
    if SESSION_CHAT_HISTORY not in st.session_state:
        st.session_state[SESSION_CHAT_HISTORY] = []


# Authentication helpers
def set_logged_in(user: Dict, token: str) -> None:
    """Set user as logged in with token."""
    st.session_state[SESSION_LOGGED_IN] = True
    st.session_state[SESSION_USER] = user
    st.session_state[SESSION_TOKEN] = token


def set_logged_out() -> None:
    """Clear login state."""
    st.session_state[SESSION_LOGGED_IN] = False
    st.session_state[SESSION_USER] = None
    st.session_state[SESSION_TOKEN] = None


def is_logged_in() -> bool:
    """Check if user is logged in."""
    return st.session_state.get(SESSION_LOGGED_IN, False)


def get_current_user() -> Optional[Dict]:
    """Get current logged in user."""
    return st.session_state.get(SESSION_USER)


def get_auth_token() -> Optional[str]:
    """Get current auth token."""
    return st.session_state.get(SESSION_TOKEN)


def get_user_name() -> str:
    """Get current user's name or 'Guest'."""
    user = get_current_user()
    if user and isinstance(user, dict):
        return user.get("name") or user.get("username") or "Guest"
    return "Guest"


def get_user_id() -> Optional[int]:
    """Get current user's ID."""
    user = get_current_user()
    if user and isinstance(user, dict):
        return user.get("id")
    return None


# UI state helpers
def toggle_dark_mode() -> None:
    """Toggle dark mode on/off."""
    st.session_state[SESSION_DARK_MODE] = not st.session_state.get(SESSION_DARK_MODE, False)


def is_dark_mode() -> bool:
    """Check if dark mode is enabled."""
    return st.session_state.get(SESSION_DARK_MODE, False)


def set_dark_mode(enabled: bool) -> None:
    """Set dark mode on/off."""
    st.session_state[SESSION_DARK_MODE] = enabled


# Patient data helpers
def set_patients(patients: List[Dict]) -> None:
    """Store patients list in session."""
    st.session_state[SESSION_PATIENTS] = patients


def get_patients() -> List[Dict]:
    """Get patients list from session."""
    return st.session_state.get(SESSION_PATIENTS, [])


def add_patient(patient: Dict) -> None:
    """Add a patient to the session."""
    patients = get_patients()
    patients.append(patient)
    set_patients(patients)


def clear_patients() -> None:
    """Clear all patients from session."""
    st.session_state[SESSION_PATIENTS] = []


def get_patient_count() -> int:
    """Get number of stored patients."""
    return len(get_patients())


# Prediction helpers
def set_prediction(prediction: Dict) -> None:
    """Store prediction result in session."""
    st.session_state[SESSION_PREDICTION] = prediction


def get_prediction() -> Optional[Dict]:
    """Get last prediction result from session."""
    return st.session_state.get(SESSION_PREDICTION)


def clear_prediction() -> None:
    """Clear prediction from session."""
    st.session_state[SESSION_PREDICTION] = None


def has_prediction() -> bool:
    """Check if there's a stored prediction."""
    return st.session_state.get(SESSION_PREDICTION) is not None


# Chat helpers
def add_chat_message(role: str, content: str) -> None:
    """Add a message to chat history."""
    chat_history = get_chat_history()
    chat_history.append({"role": role, "content": content})
    st.session_state[SESSION_CHAT_HISTORY] = chat_history


def get_chat_history() -> List[Dict]:
    """Get chat history."""
    return st.session_state.get(SESSION_CHAT_HISTORY, [])


def clear_chat_history() -> None:
    """Clear chat history."""
    st.session_state[SESSION_CHAT_HISTORY] = []


def get_chat_message_count() -> int:
    """Get number of messages in chat."""
    return len(get_chat_history())


def get_last_chat_message() -> Optional[Dict]:
    """Get the last message in chat history."""
    history = get_chat_history()
    return history[-1] if history else None


# Generic session helpers
def set_session_var(key: str, value: Any) -> None:
    """Set a generic session variable."""
    st.session_state[key] = value


def get_session_var(key: str, default: Any = None) -> Any:
    """Get a generic session variable."""
    return st.session_state.get(key, default)


def has_session_var(key: str) -> bool:
    """Check if session variable exists."""
    return key in st.session_state


def clear_session_var(key: str) -> None:
    """Clear a session variable."""
    if key in st.session_state:
        del st.session_state[key]


def get_all_session_state() -> Dict:
    """Get entire session state as dictionary (for debugging)."""
    return dict(st.session_state)


def clear_all_session_state() -> None:
    """Clear all session state (logout + reset UI)."""
    for key in list(st.session_state.keys()):
        if key.startswith("_"):  # Keep internal Streamlit keys
            continue
        del st.session_state[key]
    init_session_state()
