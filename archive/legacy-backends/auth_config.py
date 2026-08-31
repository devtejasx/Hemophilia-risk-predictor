"""
Authentication Configuration
JWT settings, security settings, and secrets configuration
"""

from datetime import timedelta
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings for authentication"""
    
    # JWT Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-12345678")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # Security
    PASSWORD_MIN_LENGTH: int = 8
    BCRYPT_ROUNDS: int = 12
    
    # Allowed roles
    ALLOWED_ROLES: list = ["doctor", "admin", "patient"]
    
    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit
        "http://localhost:8000",
    ]
    
    @property
    def access_token_expire(self) -> timedelta:
        """Get access token expiration as timedelta"""
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    @property
    def refresh_token_expire(self) -> timedelta:
        """Get refresh token expiration as timedelta"""
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)


settings = Settings()
