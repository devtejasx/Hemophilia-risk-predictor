"""Authentication: register, login, current user."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import db
from backend.core import security
from backend.core.config import settings
from backend.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Resolve the bearer token to a user, or 401."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = security.decode_access_token(credentials.credentials)
    except security.AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = db.get_user_by_id(int(payload.get("sub", 0)))
    if user is None or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is unavailable."
        )
    return user


@router.post("/register", response_model=TokenResponse,
             status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> TokenResponse:
    problems = security.password_problems(payload.password)
    if problems:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Password {', and '.join(problems)}.",
        )

    if db.get_user_by_email(payload.email) is not None:
        # Deliberately the same shape as any other conflict; no probing for
        # which addresses are registered beyond what registration must reveal.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user = db.create_user(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=security.hash_password(payload.password),
    )
    db.write_audit_log(user["id"], "user.register", "users", user["id"])
    logger.info("Registered user id=%s", user["id"])
    return TokenResponse(
        access_token=security.create_access_token(user["id"], {"email": user["email"]}),
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = db.get_user_by_email(payload.email)

    # Same message and roughly the same work whether the account exists or the
    # password is wrong, so the endpoint does not enumerate accounts.
    if user is None or not security.verify_password(
        payload.password, user["password_hash"]
    ):
        logger.info("Failed login attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled."
        )

    db.touch_last_login(user["id"])
    db.write_audit_log(user["id"], "user.login", "users", user["id"])
    return TokenResponse(
        access_token=security.create_access_token(user["id"], {"email": user["email"]}),
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"],
    )
