"""Password hashing and JWT handling.

Replaces the unsalted SHA-256 hashing on the previous live path (audit B-20).
bcrypt is used because it is salted and deliberately slow.

We call the `bcrypt` library directly rather than through passlib: passlib
1.7.4 reads `bcrypt.__about__.__version__`, which bcrypt 5 removed, and its
backend probe then dies inside a wrap-bug check. Direct use is also one fewer
dependency.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from backend.core.config import settings

logger = logging.getLogger(__name__)

# Resolved once at import so a misconfigured production process fails at
# startup rather than on the first login attempt.
_SECRET = settings.resolved_jwt_secret()

MIN_PASSWORD_LENGTH = 12

COMMON_PASSWORDS = {
    "password123", "password", "changeme", "admin", "letmein",
    "administrator", "hemophilia", "welcome123",
}


def _prepare(password: str) -> bytes:
    """Reduce any password to a fixed 44 bytes before bcrypt sees it.

    bcrypt silently truncates (or, in v5, rejects) input beyond 72 bytes, so a
    long passphrase would otherwise be weakened without warning. SHA-256 then
    base64 keeps the full entropy of the original and always fits.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


class AuthError(Exception):
    """Authentication or token failure."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password, returning False rather than raising on a bad hash."""
    try:
        return bcrypt.checkpw(_prepare(plain), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # e.g. a legacy SHA-256 or MD5 digest sitting in an old database row.
        logger.warning("Stored password hash is not a valid bcrypt hash; rejecting.")
        return False


def password_problems(password: str) -> list[str]:
    """Human-readable reasons a password is unacceptable, empty if it is fine."""
    problems = []
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"must be at least {MIN_PASSWORD_LENGTH} characters")
    if password.lower() in COMMON_PASSWORDS:
        problems.append("is a commonly used password")
    return problems


def create_access_token(subject: str | int, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        **(extra or {}),
    }
    return jwt.encode(payload, _SECRET, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, _SECRET, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Your session has expired. Please sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid authentication credentials.") from exc
