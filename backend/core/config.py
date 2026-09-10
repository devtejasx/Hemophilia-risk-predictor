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

    # --- data + model -----------------------------------------------------
    # Paths are configurable so the application works from any checkout. The
    # ml package reads DATA_DIR / MMC2_PATH / MMC3_PATH directly; they are
    # declared here so the whole configuration surface is visible in one place.
    data_dir: str = os.getenv("DATA_DIR", str(REPO_ROOT / "ml" / "data"))
    mmc2_path: str = os.getenv(
        "MMC2_PATH",
        str(REPO_ROOT / "ml" / "data" / "BVTH_VTH-2024-000215-mmc2.csv"),
    )
    mmc3_path: str = os.getenv(
        "MMC3_PATH",
        str(REPO_ROOT / "ml" / "data" / "BVTH_VTH-2024-000215-mmc3.csv"),
    )

    artifacts_dir: str = os.getenv("ML_ARTIFACTS_DIR", str(REPO_ROOT / "ml" / "artifacts"))
    #: Which prediction mode the API serves when the caller does not choose one.
    default_feature_set: str = os.getenv("ML_DEFAULT_FEATURE_SET", "merged")
    #: Explicitly pinning a version overrides the feature-set routing entirely.
    model_version: str = os.getenv("ML_MODEL_VERSION", "")

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
        if self.model_version:
            # A positive check, not a blocklist: pinning anything other than a
            # current MMC2/MMC3 artifact is almost always a stale value carried
            # over from a previous deployment, and it silently disables mode
            # routing. Naming the known-good versions means a new artifact has
            # to be introduced deliberately rather than by typo.
            from ml.inference import FEATURE_SET_VERSIONS

            known = set(FEATURE_SET_VERSIONS.values())
            if self.model_version not in known:
                warnings.append(
                    f"ML_MODEL_VERSION={self.model_version!r} is not one of the "
                    f"MMC2/MMC3 artifacts ({', '.join(sorted(known))}). Leave it "
                    "empty to serve every prediction mode."
                )
        if self.default_feature_set not in {"genomic", "clinical", "merged"}:
            warnings.append(
                f"ML_DEFAULT_FEATURE_SET={self.default_feature_set!r} is not one "
                "of genomic, clinical, merged."
            )
        for label, path in (
            ("MMC2_PATH", self.mmc2_path),
            ("MMC3_PATH", self.mmc3_path),
        ):
            if not Path(path).is_file():
                warnings.append(
                    f"{label} does not exist ({path}). Training and dataset "
                    "validation will fail; serving a built model will not."
                )
        return warnings


settings = Settings()
