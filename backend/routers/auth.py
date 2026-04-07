"""
Authentication router for login, registration, and token management
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime
from backend.models import (
    UserLogin, UserRegister, UserResponse, TokenResponse,
    RefreshTokenRequest, ChangePasswordRequest, PasswordResetRequest
)
from backend.auth import create_access_token, create_refresh_token, refresh_access_token
from backend.users import (
    authenticate_user, create_user, get_user_by_id, update_last_login,
    change_password, reset_password, init_user_database, create_admin_if_not_exists
)
from backend.security import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Initialize database on startup
try:
    init_user_database()
    create_admin_if_not_exists()
except Exception as e:
    print(f"Warning: Could not initialize auth database: {e}")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Authenticate user with username and password, returns JWT tokens"
)
async def login(credentials: UserLogin):
    """
    Login endpoint.
    
    **Returns:**
    - `access_token`: JWT token for API requests
    - `refresh_token`: Token to refresh access token
    - `expires_in`: Expiration time in seconds
    - `user`: User profile information
    
    **Example:**
    ```
    POST /auth/login
    {
        "username": "doctor1",
        "password": "securepassword"
    }
    ```
    """
    # Authenticate user
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token, _ = create_access_token(user["username"], user["role"])
    refresh_token, _ = create_refresh_token(user["username"], user["role"])
    
    # Update last login
    update_last_login(user["id"])
    
    # Get fresh user data
    user_data = get_user_by_id(user["id"])
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60,  # 30 minutes in seconds
        user=UserResponse(**user_data)
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description="Register new user account"
)
async def register(user_data: UserRegister):
    """
    Register new user.
    
    **Roles available:**
    - `patient`: Regular patient user
    - `doctor`: Medical professional
    - `admin`: System administrator (admin only)
    
    **Password requirements:**
    - Minimum 8 characters
    - Recommended: Mix of uppercase, lowercase, numbers, and special characters
    
    **Example:**
    ```
    POST /auth/register
    {
        "username": "patient1",
        "email": "patient@example.com",
        "password": "SecurePass123!",
        "full_name": "John Doe",
        "role": "patient"
    }
    ```
    """
    # Validate role (patients can only register as patient)
    if user_data.role not in ["patient", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only register as 'patient' or 'doctor'. Admin accounts must be created by administrators."
        )
    
    try:
        # Create user
        new_user = create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=user_data.role
        )
        
        if not new_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not create user account"
            )
        
        # Create tokens
        access_token, _ = create_access_token(new_user["username"], new_user["role"])
        refresh_token, _ = create_refresh_token(new_user["username"], new_user["role"])
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=30 * 60,
            user=UserResponse(**new_user)
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error registering user"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Token",
    description="Get new access token using refresh token"
)
async def refresh(request: RefreshTokenRequest):
    """
    Refresh access token.
    
    Use this endpoint to get a new access token when your current token is about to expire.
    
    **Example:**
    ```
    POST /auth/refresh
    {
        "refresh_token": "eyJhbGc..."
    }
    ```
    """
    new_token = refresh_access_token(request.refresh_token)
    
    if not new_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user info from token
    from backend.auth import get_user_from_token
    user_info = get_user_from_token(new_token.access_token)
    
    if user_info:
        user = get_user_by_id(user_info.get("id")) or get_user_by_username(user_info["username"])
        if user:
            return TokenResponse(
                access_token=new_token.access_token,
                token_type="bearer",
                expires_in=30 * 60,
                user=UserResponse(**user)
            )
    
    return new_token


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User",
    description="Get information about currently logged-in user"
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile.
    
    Requires authentication (Bearer token).
    
    **Example response:**
    ```json
    {
        "id": 1,
        "username": "doctor1",
        "email": "doctor@hospital.com",
        "full_name": "Dr. Jane Smith",
        "role": "doctor",
        "created_at": "2026-04-01T12:00:00",
        "last_login": "2026-04-02T10:30:00",
        "is_active": true
    }
    ```
    """
    user = get_user_by_id(current_user["id"]) or get_user_by_username(current_user["username"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(**user)


@router.post(
    "/change-password",
    summary="Change Password",
    description="Change user password"
)
async def change_user_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Change password for current user.
    
    Requires authentication and current password verification.
    
    **Example:**
    ```
    POST /auth/change-password
    {
        "current_password": "OldPass123!",
        "new_password": "NewPass456!"
    }
    ```
    """
    success = change_password(
        current_user["id"],
        request.current_password,
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid current password or error changing password"
        )
    
    return {"message": "Password changed successfully"}


@router.post(
    "/forgot-password",
    summary="Request Password Reset",
    description="Request password reset link (for future implementation)"
)
async def forgot_password(request: PasswordResetRequest):
    """
    Request password reset.
    
    Note: Current implementation is placeholder. In production, this should:
    1. Generate secure reset token
    2. Store in database with expiration
    3. Send email with reset link
    4. Require token to actually reset password
    
    **Example:**
    ```
    POST /auth/forgot-password
    {
        "email": "user@example.com"
    }
    ```
    """
    # Check if user exists
    from backend.users import get_user_by_email
    user = get_user_by_email(request.email)
    
    if not user:
        # Don't reveal if email exists (security best practice)
        return {"message": "If email exists, password reset link has been sent"}
    
    # TODO: Generate reset token and send email
    # For now, just log the attempt
    from backend.users import log_audit
    log_audit("password_reset_requested", f"Email: {request.email}", user["id"])
    
    return {"message": "If email exists, password reset link has been sent"}


@router.post(
    "/logout",
    summary="Logout",
    description="Logout current user (invalidate tokens)"
)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout user.
    
    In a stateless JWT system, this endpoint doesn't actually do anything on the backend.
    The client should delete the stored tokens.
    
    To fully implement logout with token blacklisting, you would need:
    - Token blacklist in Redis or database
    - Middleware to check blacklist on each request
    - Token expiration validation
    
    **Example:**
    ```
    POST /auth/logout
    Authorization: Bearer <your_token>
    ```
    """
    from backend.users import log_audit
    log_audit("logout", f"User logged out", current_user["id"])
    
    return {"message": "Logged out successfully. Please delete your tokens from the client."}


# Helper function for importing in other files
def get_user_by_username(username: str):
    """Helper to import user lookup"""
    from backend.users import get_user_by_username as _get_user_by_username
    return _get_user_by_username(username)
