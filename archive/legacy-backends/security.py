"""
Security Module - Production-Grade Authentication & Authorization
JWT tokens, password hashing, and role-based access control (RBAC)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from config import settings
from logging_config import app_logger


# ==================== Password Hashing ====================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Previously hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ==================== JWT Token Management ====================

class TokenManager:
    """
    Manages JWT token creation, validation, and refresh
    """
    
    @staticmethod
    def create_access_token(
        subject: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token
        
        Args:
            subject: Token payload (user_id, username, role, etc)
            expires_delta: Custom expiration time
            
        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": str(subject.get("user_id")),
            "username": subject.get("username"),
            "role": subject.get("role"),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                settings.SECRET_KEY,
                algorithm=settings.ALGORITHM
            )
            app_logger.info(f"Access token created for user: {subject.get('username')}")
            return encoded_jwt
        except Exception as e:
            app_logger.error(f"Error creating access token: {str(e)}")
            raise
    
    @staticmethod
    def create_refresh_token(user_id: int, username: str) -> str:
        """
        Create JWT refresh token (longer expiration)
        
        Args:
            user_id: User ID
            username: Username
            
        Returns:
            Encoded refresh token
        """
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                settings.SECRET_KEY,
                algorithm=settings.ALGORITHM
            )
            return encoded_jwt
        except Exception as e:
            app_logger.error(f"Error creating refresh token: {str(e)}")
            raise
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify JWT token and extract payload
        
        Args:
            token: JWT token
            token_type: Type of token (access/refresh)
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                app_logger.warning(f"Invalid token type: expected {token_type}")
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            app_logger.warning(f"Token expired")
            return None
        except jwt.InvalidTokenError as e:
            app_logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            app_logger.error(f"Error verifying token: {str(e)}")
            return None
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[str]:
        """
        Create new access token using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if valid, None otherwise
        """
        payload = TokenManager.verify_token(refresh_token, token_type="refresh")
        if not payload:
            return None
        
        return TokenManager.create_access_token({
            "user_id": int(payload["sub"]),
            "username": payload["username"],
            "role": payload.get("role", "user")
        })


# ==================== API Key Management ====================

class APIKeyManager:
    """
    Manages API key generation and validation for service-to-service auth
    """
    
    @staticmethod
    def generate_api_key(service_name: str) -> str:
        """
        Generate API key for service
        
        Args:
            service_name: Name of the service
            
        Returns:
            Generated API key
        """
        import secrets
        api_key = secrets.token_urlsafe(32)
        app_logger.info(f"API key generated for service: {service_name}")
        return api_key
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return pwd_context.hash(api_key)
    
    @staticmethod
    def verify_api_key(plain_key: str, hashed_key: str) -> bool:
        """Verify API key"""
        return pwd_context.verify(plain_key, hashed_key)


# ==================== Role-Based Access Control ====================

class RBAC:
    """
    Role-Based Access Control
    Defines permissions for each role
    """
    
    ROLE_PERMISSIONS: Dict[str, list] = {
        "admin": [
            "create_user", "read_user", "update_user", "delete_user",
            "create_patient", "read_patient", "update_patient", "delete_patient",
            "manage_roles", "view_analytics", "manage_system"
        ],
        "doctor": [
            "read_patient", "update_patient", "create_doctor_note",
            "read_doctor_note", "update_doctor_note",
            "create_prescription", "view_analytics", "chat_support"
        ],
        "nurse": [
            "read_patient", "update_patient", "create_monitoring_record",
            "read_monitoring_record", "chat_support"
        ],
        "lab_tech": [
            "read_patient", "create_monitoring_record",
            "read_monitoring_record", "view_analytics"
        ],
        "patient": [
            "read_own_patient", "view_own_records", "chat_support"
        ]
    }
    
    @staticmethod
    def has_permission(role: str, permission: str) -> bool:
        """
        Check if role has permission
        
        Args:
            role: User role
            permission: Required permission
            
        Returns:
            True if role has permission
        """
        permissions = RBAC.ROLE_PERMISSIONS.get(role, [])
        return permission in permissions
    
    @staticmethod
    def has_any_permission(role: str, permissions: list) -> bool:
        """Check if role has ANY of the permissions"""
        role_permissions = RBAC.ROLE_PERMISSIONS.get(role, [])
        return any(p in role_permissions for p in permissions)
    
    @staticmethod
    def has_all_permissions(role: str, permissions: list) -> bool:
        """Check if role has ALL of the permissions"""
        role_permissions = RBAC.ROLE_PERMISSIONS.get(role, [])
        return all(p in role_permissions for p in permissions)


# ==================== Session Management ====================

class SessionManager:
    """
    Manage user sessions (in production, use Redis or database)
    """
    
    _active_sessions: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def create_session(user_id: int, username: str, role: str, 
                      ip_address: str) -> str:
        """
        Create new session
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            ip_address: Client IP address
            
        Returns:
            Session ID
        """
        import uuid
        session_id = str(uuid.uuid4())
        SessionManager._active_sessions[session_id] = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "ip_address": ip_address,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        app_logger.info(f"Session created for user: {username} (Session: {session_id})")
        return session_id
    
    @staticmethod
    def validate_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and update last activity"""
        session = SessionManager._active_sessions.get(session_id)
        if session:
            session["last_activity"] = datetime.utcnow()
            return session
        return None
    
    @staticmethod
    def invalidate_session(session_id: str) -> bool:
        """Invalidate session (logout)"""
        if session_id in SessionManager._active_sessions:
            del SessionManager._active_sessions[session_id]
            app_logger.info(f"Session invalidated: {session_id}")
            return True
        return False
    
    @staticmethod
    def cleanup_expired_sessions(max_inactive_minutes: int = 30) -> int:
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in SessionManager._active_sessions.items()
            if (now - session["last_activity"]).total_seconds() > max_inactive_minutes * 60
        ]
        for sid in expired:
            del SessionManager._active_sessions[sid]
        
        if expired:
            app_logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)
