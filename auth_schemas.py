"""
Authentication Pydantic Models
Request and response schemas for API endpoints
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """User creation schema"""
    password: str = Field(..., min_length=8, max_length=100)
    role: str = Field(default="doctor", regex="^(doctor|admin|patient)$")
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength"""
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char in "!@#$%^&*" for char in v):
            raise ValueError("Password must contain at least one special character (!@#$%^&*)")
        return v


class UserUpdate(BaseModel):
    """User update schema"""
    full_name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    

class UserResponse(UserBase):
    """User response schema"""
    user_id: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserDetailResponse(UserResponse):
    """Detailed user response"""
    updated_at: datetime


# ============================================================================
# LOGIN/AUTHENTICATION SCHEMAS
# ============================================================================

class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        if not any(char in "!@#$%^&*" for char in v):
            raise ValueError("Password must contain at least one special character (!@#$%^&*)")
        return v


# ============================================================================
# TOKEN SCHEMAS
# ============================================================================

class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class TokenData(BaseModel):
    """JWT token data"""
    user_id: str
    email: str
    role: str
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None


# ============================================================================
# ADMIN SCHEMAS
# ============================================================================

class AdminUserCreate(UserCreate):
    """Admin user creation with role assignment"""
    role: str = Field(..., regex="^(doctor|admin|patient)$")


class AdminUserUpdate(BaseModel):
    """Admin user update schema"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = Field(None, regex="^(doctor|admin|patient)$")
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class AdminListResponse(BaseModel):
    """Admin list users response"""
    total: int
    users: List[UserDetailResponse]


# ============================================================================
# ERROR RESPONSE SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    errors: Optional[List[dict]] = None
