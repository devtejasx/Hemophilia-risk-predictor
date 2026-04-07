"""
Security dependencies for FastAPI route protection.
Implements JWT verification and role-based access control.
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from backend.auth import get_user_from_token, check_permission

security = HTTPBearer()


class CurrentUser:
    """Dependency to extract and validate current user from JWT token"""
    
    def __init__(self, required_roles: Optional[List[str]] = None):
        """
        Args:
            required_roles: List of allowed roles. If None, any authenticated user allowed.
        """
        self.required_roles = required_roles or []
    
    async def __call__(self, credentials: HTTPAuthCredentials = Depends(security)):
        token = credentials.credentials
        
        user = get_user_from_token(token)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check role if required_roles specified
        if self.required_roles and user["role"] not in self.required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires one of these roles: {', '.join(self.required_roles)}"
            )
        
        return user


async def get_current_user(
    credentials: HTTPAuthCredentials = Depends(security)
) -> dict:
    """Get current authenticated user"""
    token = credentials.credentials
    user = get_user_from_token(token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_admin(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current user and verify admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_doctor(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current user and verify doctor role"""
    if current_user["role"] not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor access required"
        )
    return current_user


async def get_current_patient_or_doctor(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Get current user and verify patient or doctor role"""
    if current_user["role"] not in ["admin", "doctor", "patient"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Patient or doctor access required"
        )
    return current_user


def require_permission(permission: str):
    """
    Dependency factory to require specific permission
    Usage: @app.get("/protected", dependencies=[Depends(require_permission("read"))])
    """
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        if not check_permission(current_user["role"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker


def require_role(*roles: str):
    """
    Dependency factory to require specific roles
    Usage: @app.get("/admin", dependencies=[Depends(require_role("admin"))])
    """
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of these roles: {', '.join(roles)}"
            )
        return current_user
    
    return role_checker
