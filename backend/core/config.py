"""Environment-driven configuration. One source of truth for the backend.

Evolved from the previous root-level config.py, which had the right shape but
pointed MODELS_PATH at a ./models directory that never existed and defaulted
SECRET_KEY to a literal string.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


class ConfigError(RuntimeError):
    """Configuration is missing or unsafe for the selected environment."""


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = _bool("DEBUG", False)

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Exact origins only. "*" with credentials is rejected by browsers and was
    # what the archived backends used.
    cors_origins: list[str] = field(
        default_factory=lambda: _list(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        )
    )

    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    database_path: str = os.getenv("DATABASE_PATH", str(REPO_ROOT / "hemophilia.db"))

    artifacts_dir: str = os.getenv("ML_ARTIFACTS_DIR", str(REPO_ROOT / "ml" / "artifacts"))
    model_version: str = os.getenv("ML_MODEL_VERSION", "champ-v1")

    bootstrap_admin_email: str = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "")
    bootstrap_admin_password: str = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def resolved_jwt_secret(self) -> str:
        """The signing key, or a clear failure.

        In production a missing key is fatal — a predictable secret means anyone
        can mint a valid token. In development a random per-process key is
        generated so `uvicorn` starts without setup; tokens simply do not
        survive a restart.
        """
        if self.jwt_secret_key:
            if len(self.jwt_secret_key) < 32 and self.is_production:
                raise ConfigError(
                    "JWT_SECRET_KEY must be at least 32 characters in production."
                )
            return self.jwt_secret_key
        if self.is_production:
            raise ConfigError(
                "JWT_SECRET_KEY is required in production. Generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return secrets.token_urlsafe(48)

    def validate(self) -> list[str]:
        """Non-fatal warnings worth logging at startup."""
        warnings: list[str] = []
        if self.is_production and self.debug:
            warnings.append("DEBUG is enabled in production.")
        if "*" in self.cors_origins:
            warnings.append(
                "CORS_ORIGINS contains '*', which browsers reject alongside "
                "credentials. Set explicit origins."
            )
        if self.model_version == "legacy-synthetic-v0":
            warnings.append(
                "ML_MODEL_VERSION is the legacy synthetic model, which was "
                "trained on fabricated data and cannot be served."
            )
        return warnings


settings = Settings()
