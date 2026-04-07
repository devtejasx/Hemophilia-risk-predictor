"""
Authentication API Endpoints
Login, signup, refresh token, and user management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from auth_schemas import (
    UserCreate, UserResponse, UserDetailResponse, UserUpdate,
    LoginRequest, TokenResponse, RefreshTokenRequest,
    ChangePasswordRequest, AdminUserCreate, AdminUserUpdate,
    AdminListResponse, ErrorResponse
)
from auth_security import (
    hash_password, verify_password, JWTManager,
    validate_password_strength, create_token_data
)
from auth_database import UserManager, RefreshTokenManager
from auth_dependencies import (
    get_current_user, get_current_user_optional, 
    require_admin, require_doctor, require_verified,
    require_admin_verified, require_doctor_verified,
    get_db
)
from auth_models import User
from auth_config import settings

# Create router
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "User already exists"},
        500: {"model": ErrorResponse, "description": "Server error"},
    }
)
async def signup(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register new user
    
    Args:
        user_create: User registration data
        db: Database session
        
    Returns:
        Token response with access and refresh tokens
        
    Raises:
        HTTPException: 400 validation error, 409 conflict
    """
    # Validate password strength
    is_valid, error = validate_password_strength(user_create.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Create user
    user, error = UserManager.create_user(db, user_create)
    
    if error:
        if "already registered" in error or "already taken" in error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Create tokens
    token_data = create_token_data(user.user_id, user.email, user.role.value)
    access_token, access_exp = JWTManager.create_access_token(token_data)
    refresh_token, refresh_exp = JWTManager.create_refresh_token(token_data)
    
    # Store refresh token
    RefreshTokenManager.create_refresh_token(db, user.user_id, refresh_token, refresh_exp)
    
    # Update last login
    UserManager.update_last_login(db, user)
    
    expire_in = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expire_in,
        user=UserResponse.from_orm(user)
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        403: {"model": ErrorResponse, "description": "Account inactive"},
        404: {"model": ErrorResponse, "description": "User not found"},
    }
)
async def login(
    login_request: LoginRequest,
    db: Session = Depends(get_db)
):
    """User login
    
    Args:
        login_request: Email and password
        db: Database session
        
    Returns:
        Token response with access and refresh tokens
        
    Raises:
        HTTPException: 401 invalid credentials, 403 inactive account
    """
    # Find user by email
    user = UserManager.get_user_by_email(db, login_request.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Verify password
    if not verify_password(login_request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens
    token_data = create_token_data(user.user_id, user.email, user.role.value)
    access_token, access_exp = JWTManager.create_access_token(token_data)
    refresh_token, refresh_exp = JWTManager.create_refresh_token(token_data)
    
    # Store refresh token
    RefreshTokenManager.create_refresh_token(db, user.user_id, refresh_token, refresh_exp)
    
    # Update last login
    UserManager.update_last_login(db, user)
    
    expire_in = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expire_in,
        user=UserResponse.from_orm(user)
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid refresh token"},
    }
)
async def refresh_access_token(
    refresh_request: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token
    
    Args:
        refresh_request: Refresh token
        db: Database session
        
    Returns:
        New token response
        
    Raises:
        HTTPException: 401 invalid/revoked token
    """
    # Verify refresh token
    payload = JWTManager.verify_token(refresh_request.refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get refresh token record
    token_record = RefreshTokenManager.get_refresh_token(
        db, refresh_request.refresh_token
    )
    
    if not token_record or not token_record.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired"
        )
    
    # Get user
    user = UserManager.get_user_by_id(db, payload.get("user_id"))
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer active"
        )
    
    # Create new access token
    token_data = create_token_data(user.user_id, user.email, user.role.value)
    access_token, _ = JWTManager.create_access_token(token_data)
    
    expire_in = int(settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_request.refresh_token,
        expires_in=expire_in,
        user=UserResponse.from_orm(user)
    )


# ============================================================================
# PROTECTED ENDPOINTS - USER
# ============================================================================

@router.get(
    "/me",
    response_model=UserDetailResponse,
)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User details
    """
    return UserDetailResponse.from_orm(current_user)


@router.put(
    "/me",
    response_model=UserDetailResponse,
)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user information
    
    Args:
        user_update: Update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated user details
    """
    updated_user, error = UserManager.update_user(db, current_user, user_update)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return UserDetailResponse.from_orm(updated_user)


@router.post(
    "/change-password",
    status_code=status.HTTP_200_OK,
)
async def change_password(
    change_pwd: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password
    
    Args:
        change_pwd: Current and new passwords
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    # Verify current password
    if not verify_password(change_pwd.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Verify new password != old password
    if verify_password(change_pwd.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Validate new password strength
    is_valid, error = validate_password_strength(change_pwd.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    # Verify passwords match
    if change_pwd.new_password != change_pwd.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password confirmation does not match"
        )
    
    # Update password
    current_user.hashed_password = hash_password(change_pwd.new_password)
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
)
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout (revoke all user tokens)
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    RefreshTokenManager.revoke_user_tokens(db, current_user.user_id)
    return {"message": "Logged out successfully"}


# ============================================================================
# PROTECTED ENDPOINTS - ADMIN
# ============================================================================

@router.post(
    "/admin/users",
    response_model=UserDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def admin_create_user(
    admin_user_create: AdminUserCreate,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin create user
    
    Args:
        admin_user_create: User creation data
        current_admin: Current admin user
        db: Database session
        
    Returns:
        Created user
    """
    user, error = UserManager.create_user(db, admin_user_create)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return UserDetailResponse.from_orm(user)


@router.get(
    "/admin/users",
    response_model=AdminListResponse,
)
async def admin_list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = None,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin list users
    
    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        role: Filter by role
        current_admin: Current admin user
        db: Database session
        
    Returns:
        List of users
    """
    users, total = UserManager.list_users(db, skip, limit, role)
    
    return AdminListResponse(
        total=total,
        users=[UserDetailResponse.from_orm(u) for u in users]
    )


@router.get(
    "/admin/users/{user_id}",
    response_model=UserDetailResponse,
)
async def admin_get_user(
    user_id: str,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin get user details
    
    Args:
        user_id: User ID
        current_admin: Current admin user
        db: Database session
        
    Returns:
        User details
    """
    user = UserManager.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserDetailResponse.from_orm(user)


@router.put(
    "/admin/users/{user_id}",
    response_model=UserDetailResponse,
)
async def admin_update_user(
    user_id: str,
    admin_update: AdminUserUpdate,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin update user
    
    Args:
        user_id: User ID
        admin_update: Update data
        current_admin: Current admin user
        db: Database session
        
    Returns:
        Updated user
    """
    user = UserManager.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    updated_user, error = UserManager.admin_update_user(db, user, admin_update)
    
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return UserDetailResponse.from_orm(updated_user)


@router.delete(
    "/admin/users/{user_id}",
    status_code=status.HTTP_200_OK,
)
async def admin_delete_user(
    user_id: str,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Admin delete (deactivate) user
    
    Args:
        user_id: User ID
        current_admin: Current admin user
        db: Database session
        
    Returns:
        Success message
    """
    user = UserManager.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    success, error = UserManager.delete_user(db, user)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {"message": f"User {user_id} has been deactivated"}


@router.get(
    "/admin/stats",
    response_model=dict,
)
async def admin_get_stats(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get authentication statistics
    
    Args:
        current_admin: Current admin user
        db: Database session
        
    Returns:
        Statistics
    """
    total_users = UserManager.count_users(db)
    
    return {
        "total_users": total_users,
        "timestamp": datetime.utcnow().isoformat(),
    }
