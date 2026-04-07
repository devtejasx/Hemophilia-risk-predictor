"""
Security Utilities
Password hashing, JWT token generation, and verification
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
import uuid
from auth_config import settings


# ============================================================================
# PASSWORD HASHING
# ============================================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS
)


def hash_password(password: str) -> str:
    """Hash password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

class JWTManager:
    """JWT token management"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> tuple[str, datetime]:
        """Create JWT access token
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            Tuple of (token, expiration_datetime)
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + settings.access_token_expire
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        return encoded_jwt, expire
    
    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> tuple[str, datetime]:
        """Create JWT refresh token
        
        Args:
            data: Token payload data
            expires_delta: Custom expiration time
            
        Returns:
            Tuple of (token, expiration_datetime)
        """
        to_encode = data.copy()
        to_encode["type"] = "refresh"
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + settings.refresh_token_expire
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        return encoded_jwt, expire
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None if invalid/expired
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            print(f"Invalid token: {e}")
            return None
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """Decode JWT token without verification
        
        Args:
            token: JWT token to decode
            
        Returns:
            Token payload if decodable, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            return payload
        except Exception as e:
            print(f"Failed to decode token: {e}")
            return None


# ============================================================================
# USER ID GENERATION
# ============================================================================

def generate_user_id() -> str:
    """Generate unique user ID using UUID"""
    return str(uuid.uuid4())


def generate_token_id() -> str:
    """Generate unique token ID using UUID"""
    return str(uuid.uuid4())


# ============================================================================
# TOKEN DATA
# ============================================================================

def create_token_data(user_id: str, email: str, role: str) -> Dict[str, Any]:
    """Create token payload data
    
    Args:
        user_id: User unique identifier
        email: User email
        role: User role
        
    Returns:
        Dictionary with token data
    """
    return {
        "user_id": user_id,
        "email": email,
        "role": role,
    }


# ============================================================================
# PASSWORD VALIDATION
# ============================================================================

def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters"
    
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit"
    
    if not any(char in "!@#$%^&*" for char in password):
        return False, "Password must contain at least one special character (!@#$%^&*)"
    
    return True, None
