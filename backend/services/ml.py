"""Backend-side adapter over the ML package.

The backend owns no feature logic of its own. It calls ml.inference and
ml.explainability, which are the same code paths the training script and the
tests use. Services are constructed once at application startup.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.config import settings
from ml.artifacts import ArtifactError
from ml.explainability.service import ExplanationService
from ml.inference import PredictionService

logger = logging.getLogger(__name__)

_prediction: PredictionService | None = None
_explanation: ExplanationService | None = None
_load_error: str | None = None


def startup() -> None:
    """Load the configured model once. Records the failure rather than raising,
    so /health can report an unhealthy model instead of the process dying."""
    global _prediction, _explanation, _load_error
    try:
        _prediction = PredictionService(settings.model_version, settings.artifacts_dir)
        _explanation = ExplanationService(_prediction.bundle)
        _load_error = None
        logger.info(
            "ML ready: %s (threshold %.4f)",
            _prediction.version,
            _prediction.bundle.threshold,
        )
    except (ArtifactError, Exception) as exc:  # noqa: BLE001 - reported via /health
        _prediction = None
        _explanation = None
        _load_error = str(exc)
        logger.error("ML model failed to load: %s", exc)


def shutdown() -> None:
    global _prediction, _explanation
    _prediction = None
    _explanation = None


def is_ready() -> bool:
    return _prediction is not None


def load_error() -> str | None:
    return _load_error


def prediction_service() -> PredictionService:
    if _prediction is None:
        raise ArtifactError(
            f"Prediction model is unavailable: {_load_error or 'not loaded'}"
        )
    return _prediction


def explanation_service() -> ExplanationService:
    if _explanation is None:
        raise ArtifactError(
            f"Explanation service is unavailable: {_load_error or 'not loaded'}"
        )
    return _explanation


def model_version() -> str | None:
    return _prediction.version if _prediction else None


def interpretation(probability: float, threshold: float, category: str) -> str:
    """Probabilistic wording for the UI. Never a verdict, never a treatment
    recommendation — the archived pipeline returned strings such as
    "Consider hospitalization for immune tolerance induction"."""
    percent = probability * 100
    return (
        f"The model estimates a {percent:.1f}% probability that this F8 variant "
        f"has a reported history of inhibitor development, against a decision "
        f"threshold of {threshold * 100:.1f}% ({category.lower()}). This is an "
        f"estimate attributed to the variant, not a prediction about an "
        f"individual patient, and it does not indicate any course of treatment."
    )


def input_schema() -> dict[str, Any]:
    return prediction_service().input_schema()
