"""
Configuration Module
Environment and application settings
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # Application
    APP_NAME: str = "Medical AI Platform API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "hemophilia_clinic.db")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Models
    RF_MODEL_PATH: str = os.getenv("RF_MODEL_PATH", "rf.pkl")
    XGB_MODEL_PATH: str = os.getenv("XGB_MODEL_PATH", "xgb.pkl")
    COLUMNS_PATH: str = os.getenv("COLUMNS_PATH", "columns.pkl")
    
    @property
    def database_url(self) -> str:
        """SQLite connection string"""
        return f"sqlite:///{self.DATABASE_PATH}"


# Global settings instance
settings = Settings()
