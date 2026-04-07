"""
Authentication Dependencies
FastAPI dependencies for route protection and authorization
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from sqlalchemy.orm import Session
from typing import Optional
from auth_security import JWTManager
from auth_models import User
from auth_database import UserManager


# Security scheme for Swagger documentation
security = HTTPBearer(
    description="JWT Bearer token for authentication"
)


async def get_db() -> Session:
    """Get database session
    
    This should be implemented based on your database setup.
    Example for SQLAlchemy:
    """
    # from database import SessionLocal
    # db = SessionLocal()
    # try:
    #     yield db
    # finally:
    #     db.close()
    pass


async def get_current_user(
    credentials: HTTPAuthCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer credentials
        db: Database session
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    
    # Verify token
    payload = JWTManager.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = UserManager.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None
    
    Args:
        credentials: HTTP Bearer credentials
        db: Database session
        
    Returns:
        Current user or None if not authenticated
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = JWTManager.verify_token(token)
    
    if not payload:
        return None
    
    user_id = payload.get("user_id")
    if not user_id:
        return None
    
    user = UserManager.get_user_by_id(db, user_id)
    if user and user.is_active:
        return user
    
    return None


async def require_role(
    required_role: str
):
    """Factory function to create role-based dependency
    
    Args:
        required_role: Required user role (e.g., "admin", "doctor")
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Check if user has required role
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User if role matches
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        if current_user.role.value != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires {required_role} role",
            )
        return current_user
    
    return role_checker


async def require_roles(
    allowed_roles: list
):
    """Factory function to create multi-role dependency
    
    Args:
        allowed_roles: List of allowed roles
        
    Returns:
        Dependency function
    """
    async def roles_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Check if user has one of allowed roles
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User if role matches
            
        Raises:
            HTTPException: If user doesn't have allowed role
        """
        if current_user.role.value not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires one of: {', '.join(allowed_roles)}",
            )
        return current_user
    
    return roles_checker


async def require_verified(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require user to be verified
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if verified
        
    Raises:
        HTTPException: If user is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    return current_user


async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require admin role
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


async def require_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require doctor role
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if doctor
        
    Raises:
        HTTPException: If user is not doctor
    """
    if current_user.role.value != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor role required",
        )
    return current_user


# ============================================================================
# COMBINED DEPENDENCIES
# ============================================================================

async def require_admin_verified(
    current_user: User = Depends(require_admin)
) -> User:
    """Require admin and verified
    
    Args:
        current_user: Current admin user
        
    Returns:
        User if admin and verified
        
    Raises:
        HTTPException: If conditions not met
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    return current_user


async def require_doctor_verified(
    current_user: User = Depends(require_doctor)
) -> User:
    """Require doctor and verified
    
    Args:
        current_user: Current doctor user
        
    Returns:
        User if doctor and verified
        
    Raises:
        HTTPException: If conditions not met
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )
    return current_user
