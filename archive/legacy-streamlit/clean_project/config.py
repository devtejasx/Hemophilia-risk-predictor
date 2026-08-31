"""
Application configuration management.
Centralized configuration for all application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Base configuration class."""
    
    # Environment
    ENV = os.getenv("ENVIRONMENT", "development")
    DEBUG = ENV == "development"
    
    # Application
    APP_NAME = "Hemophilia Clinical Decision Support"
    APP_VERSION = "2.0"
    APP_DESCRIPTION = "AI-powered clinical decision support for hemophilia management"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app_data.db")
    DATABASE_PATH = Path(os.getenv("DB_PATH", "app_data.db"))
    
    # Streamlit configuration
    STREAMLIT_CONFIG = {
        "page_config": {
            "page_title": APP_NAME,
            "page_icon": "⚕️",
            "layout": "wide",
            "initial_sidebar_state": "expanded",
        },
        "theme": {
            "primaryColor": "#667eea",
            "backgroundColor": "#f8f9fa",
            "secondaryBackgroundColor": "#ffffff",
            "textColor": "#262730",
            "font": "sans serif",
        }
    }
    
    # Authentication
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    TOKEN_EXPIRATION = int(os.getenv("TOKEN_EXPIRATION_HOURS", 24))
    JWT_ALGORITHM = "HS256"
    HASH_ALGORITHM = "bcrypt"
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")
    
    # Streamlit app configuration
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # ML Model configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/risk_model.pkl")
    PREDICTION_THRESHOLD_LOW = 0.4
    PREDICTION_THRESHOLD_HIGH = 0.7
    
    # Feature bounds
    FEATURE_BOUNDS = {
        "age": {"min": 0, "max": 120},
        "clotting_factor": {"min": 0, "max": 100},
        "activity_level": {"min": 1, "max": 10},
        "compliance": {"min": 0.0, "max": 1.0},
        "bleeds": {"min": 0, "max": 50},
    }
    
    # Chat configuration
    CHAT_MAX_MESSAGES = int(os.getenv("CHAT_MAX_MESSAGES", 50))
    CHAT_AI_MODEL = os.getenv("CHAT_AI_MODEL", "clinical-llm")
    CHAT_AI_TIMEOUT = int(os.getenv("CHAT_AI_TIMEOUT", 10))
    
    # Feature flags
    FEATURES = {
        "authentication": True,
        "ml_predictions": True,
        "chat_interface": True,
        "analytics_dashboard": True,
        "shap_explainability": True,
        "user_management": True,
        "data_export": True,
    }
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    # Session configuration
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 30))
    
    # Email configuration (for notifications)
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    
    # External services
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Security
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        return getattr(cls, key, default)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///test_app.db"


def get_config() -> Config:
    """Get appropriate configuration based on environment.
    
    Returns:
        Configuration instance
    """
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Global configuration instance
config = get_config()
