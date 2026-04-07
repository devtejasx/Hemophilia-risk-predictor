"""
Authentication Database Models
SQLAlchemy ORM models for user storage
"""

from sqlalchemy import Column, String, DateTime, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    """User role enumeration"""
    DOCTOR = "doctor"
    ADMIN = "admin"
    PATIENT = "patient"


class User(Base):
    """User database model"""
    
    __tablename__ = "users"
    
    # Primary key
    user_id = Column(String(36), primary_key=True, unique=True, index=True)
    
    # User information
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    
    # Role and status
    role = Column(
        Enum(UserRole),
        default=UserRole.DOCTOR,
        nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, email={self.email}, role={self.role})>"
    
    def to_dict(self) -> dict:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class RefreshToken(Base):
    """Refresh token storage for token revocation"""
    
    __tablename__ = "refresh_tokens"
    
    # Primary key
    token_id = Column(String(36), primary_key=True, unique=True, index=True)
    
    # User reference
    user_id = Column(String(36), nullable=False, index=True)
    
    # Token data
    token = Column(String(500), nullable=False)
    
    # Status
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    def is_valid(self) -> bool:
        """Check if refresh token is valid"""
        return (
            not self.is_revoked
            and datetime.utcnow() < self.expires_at
        )
    
    def __repr__(self) -> str:
        return f"<RefreshToken(token_id={self.token_id}, user_id={self.user_id})>"
