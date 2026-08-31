"""
Streamlit Pages - Multi-page components
"""

from . import patient_form
from . import results_dashboard
from . import patient_history
from . import ai_chatbot
from . import doctor_dashboard

__all__ = [
    "patient_form",
    "results_dashboard",
    "patient_history",
    "ai_chatbot",
    "doctor_dashboard",
]
