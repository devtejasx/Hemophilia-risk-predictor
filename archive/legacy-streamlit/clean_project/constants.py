"""
Application constants and configuration values.
Centralized configuration for easy maintenance.
"""

# Application info
APP_NAME = "Hemophilia Clinical Decision Support"
APP_VERSION = "2.0"
APP_EMOJI = "⚕️"

# URLs and ports
BACKEND_URL = "http://localhost:8000"
FRONTEND_PORT = 8501
BACKEND_PORT = 8000

# ML Model settings
MIN_CLOTTING_FACTOR = 0
MAX_CLOTTING_FACTOR = 100
MIN_ACTIVITY_LEVEL = 1
MAX_ACTIVITY_LEVEL = 10
MIN_COMPLIANCE = 0.0
MAX_COMPLIANCE = 1.0
MIN_AGE = 0
MAX_AGE = 120
MAX_BLEEDS = 50

# Risk thresholds
RISK_LOW_THRESHOLD = 0.4
RISK_MEDIUM_THRESHOLD = 0.7
RISK_HIGH_THRESHOLD = 1.0

# Session state keys
SESSION_LOGGED_IN = "logged_in"
SESSION_USER = "user"
SESSION_TOKEN = "token"
SESSION_DARK_MODE = "dark_mode"
SESSION_PATIENTS = "patients"
SESSION_PREDICTION = "prediction_result"
SESSION_CHAT_HISTORY = "chat_history"

# API endpoints
ENDPOINT_REGISTER = "/api/auth/register"
ENDPOINT_LOGIN = "/api/auth/login"
ENDPOINT_PATIENTS = "/api/patients"
ENDPOINT_PREDICTIONS = "/api/predictions"
ENDPOINT_CHAT = "/api/chat"
ENDPOINT_ANALYTICS = "/api/analytics"
ENDPOINT_HEALTH = "/health"

# Chat settings
CHAT_MAX_MESSAGES = 50
AI_RESPONSE_TIMEOUT = 10  # seconds

# Feature flags
ENABLE_AUTHENTICATION = True
ENABLE_ML_PREDICTIONS = True
ENABLE_CHAT = True
ENABLE_ANALYTICS = True

# Demo data for testing
DEMO_PATIENTS = {
    "PAT001": {"name": "John Doe", "age": 45, "risk": 0.72},
    "PAT002": {"name": "Jane Smith", "age": 38, "risk": 0.45},
    "PAT003": {"name": "Michael Johnson", "age": 52, "risk": 0.88},
}

# AI chat keywords
CHAT_KEYWORDS = {
    "help": "I can help with clinical decisions, patient analysis, and treatment recommendations.",
    "risk": "Risk assessment depends on clotting factor, activity level, and medical history.",
    "treatment": "Treatment should be personalized based on individual patient profiles.",
    "protocol": "Please refer to latest clinical guidelines for hemophilia management.",
}
