"""
Services package for business logic and external integrations.
Modular service layers for ML, chatbot, and data operations.
"""

from .ml_service import (
    MLService,
    get_ml_service,
    predict_risk,
    get_feature_importance,
)

from .chatbot_service import (
    ChatbotService,
    get_chatbot_service,
    get_response,
    get_clinical_guidance,
)

from .shap_service import (
    SHAPService,
    get_shap_service,
    explain_prediction,
    generate_feature_importance as generate_shap_importance,
)

__all__ = [
    # ML Service
    "MLService",
    "get_ml_service",
    "predict_risk",
    "get_feature_importance",
    # Chatbot Service
    "ChatbotService",
    "get_chatbot_service",
    "get_response",
    "get_clinical_guidance",
    # SHAP Service
    "SHAPService",
    "get_shap_service",
    "explain_prediction",
    "generate_shap_importance",
]
