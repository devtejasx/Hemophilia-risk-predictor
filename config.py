"""
Production Configuration Module
Centralized configuration for all environments
"""

import os
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings:
    """
    Application settings with environment-based configuration
    """
    
    # Environment
    ENV: Environment = Environment(os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Configuration
    API_TITLE: str = "Hemophilia AI Platform API"
    API_VERSION: str = "2.0.0"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Frontend Configuration
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8501")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./hemophilia_clinic.db")
    DB_CONNECTION_TIMEOUT: int = 30
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_TIMEOUT: int = 60
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 60
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if ENV == Environment.PRODUCTION else "DEBUG")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    
    # ML Models
    MODELS_PATH: str = os.getenv("MODELS_PATH", "./models")
    RF_MODEL_PATH: str = os.path.join(MODELS_PATH, "rf.pkl")
    XGB_MODEL_PATH: str = os.path.join(MODELS_PATH, "xgb.pkl")
    COLUMNS_PATH: str = os.path.join(MODELS_PATH, "columns.pkl")
    
    # Batch Processing
    BATCH_SIZE_PREDICTIONS: int = 32
    MAX_BATCH_SIZE: int = 1000
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:8501",
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "http://localhost:8501"),
    ]
    
    # Health Check
    HEALTH_CHECK_INTERVAL: int = 60
    
    @classmethod
    def get_settings(cls) -> "Settings":
        """Get current settings instance"""
        return cls()
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        return True


# Singleton instance
settings = Settings()
