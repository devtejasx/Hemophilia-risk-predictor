"""
Authentication module for JWT-based auth with bcrypt password hashing.
Handles user authentication, token generation, and password management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # CHANGE THIS IN PRODUCTION
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Role definitions
VALID_ROLES = {"doctor", "patient", "admin"}
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "doctor": ["read", "write", "delete", "view_patients"],
    "patient": ["read", "write"],  # Can read/write own data
}


class TokenData(BaseModel):
    """JWT token payload"""
    sub: str  # username
    role: str
    exp: datetime
    iat: datetime
    type: str  # access or refresh


class Token(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class UserCredentials(BaseModel):
    """User login credentials"""
    username: str
    password: str
    remember_me: Optional[bool] = False


class UserRegister(BaseModel):
    """User registration"""
    username: str
    password: str
    email: str
    full_name: str
    role: str = "patient"  # Default role


class UserUpdate(BaseModel):
    """User profile update"""
    email: Optional[str] = None
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> tuple[str, datetime]:
    """
    Create JWT access token
    Returns: (token, expiration_datetime)
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "sub": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expire


def create_refresh_token(username: str, role: str) -> tuple[str, datetime]:
    """
    Create JWT refresh token
    Returns: (token, expiration_datetime)
    """
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "sub": username,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, expire


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate JWT token
    Returns: Token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def validate_token_type(payload: Dict[str, Any], expected_type: str) -> bool:
    """Validate that token is of expected type (access or refresh)"""
    return payload.get("type") == expected_type


def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Extract user info from valid access token"""
    payload = decode_token(token)
    if payload is None:
        return None
    
    if not validate_token_type(payload, "access"):
        return None
    
    return {
        "username": payload.get("sub"),
        "role": payload.get("role")
    }


def refresh_access_token(refresh_token: str) -> Optional[Token]:
    """Create new access token from refresh token"""
    payload = decode_token(refresh_token)
    if payload is None:
        return None
    
    if not validate_token_type(payload, "refresh"):
        return None
    
    username = payload.get("sub")
    role = payload.get("role")
    
    access_token, _ = create_access_token(username, role)
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def check_permission(role: str, permission: str) -> bool:
    """Check if role has permission"""
    return permission in ROLE_PERMISSIONS.get(role, [])


def validate_role(role: str) -> bool:
    """Validate that role is valid"""
    return role in VALID_ROLES
