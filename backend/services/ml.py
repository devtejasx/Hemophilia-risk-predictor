"""Backend-side adapter over the ML package.

The backend owns no feature logic of its own. It calls ml.inference and
ml.explainability, which are the same code paths the training script and the
tests use. Services are constructed once at application startup.

There is one model per prediction mode — genomic (MMC2 only), clinical (MMC3
only) and merged (both, the default) — so this module holds a small registry
rather than a single service. A mode whose artifact is missing is recorded as
unavailable and reported by /health; it does not stop the others from serving.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.config import settings
from ml.artifacts import ArtifactError
from ml.explainability.service import ExplanationService
from ml.inference import FEATURE_SET_VERSIONS, PredictionService

logger = logging.getLogger(__name__)

_predictions: dict[str, PredictionService] = {}
_explanations: dict[str, ExplanationService] = {}
_errors: dict[str, str] = {}


def default_feature_set() -> str:
    return settings.default_feature_set


def available_feature_sets() -> list[str]:
    """Modes with a loaded model, in registry order."""
    return [name for name in FEATURE_SET_VERSIONS if name in _predictions]


def startup() -> None:
    """Load every configured model once.

    Failures are recorded rather than raised, so /health can report an
    unhealthy model instead of the process dying at boot.
    """
    _predictions.clear()
    _explanations.clear()
    _errors.clear()

    # An explicit ML_MODEL_VERSION pins one artifact and disables mode routing;
    # otherwise every feature set is loaded and the caller picks per request.
    pinned = settings.model_version.strip()
    targets = (
        {settings.default_feature_set: pinned} if pinned else dict(FEATURE_SET_VERSIONS)
    )

    for feature_set, version in targets.items():
        try:
            service = PredictionService(version, settings.artifacts_dir)
            _predictions[feature_set] = service
            _explanations[feature_set] = ExplanationService(service.bundle)
            logger.info(
                "ML ready: %s serves the '%s' mode (%d features, threshold %.4f)",
                service.version,
                feature_set,
                len(service.bundle.feature_names),
                service.bundle.threshold,
            )
        except Exception as exc:  # noqa: BLE001 - reported via /health
            _errors[feature_set] = str(exc)
            logger.error("Model for '%s' failed to load: %s", feature_set, exc)


def shutdown() -> None:
    _predictions.clear()
    _explanations.clear()
    _errors.clear()


def is_ready() -> bool:
    """True when the default mode can serve. Other modes may still be missing."""
    return default_feature_set() in _predictions


def load_error() -> str | None:
    if not _errors:
        return None
    return "; ".join(f"{name}: {reason}" for name, reason in sorted(_errors.items()))


def _resolve(feature_set: str | None) -> str:
    name = feature_set or default_feature_set()
    if name not in FEATURE_SET_VERSIONS:
        raise ArtifactError(
            f"Unknown prediction mode '{name}'. "
            f"Available: {', '.join(sorted(FEATURE_SET_VERSIONS))}."
        )
    return name


def prediction_service(feature_set: str | None = None) -> PredictionService:
    name = _resolve(feature_set)
    service = _predictions.get(name)
    if service is None:
        raise ArtifactError(
            f"Prediction model for '{name}' is unavailable: "
            f"{_errors.get(name, 'not loaded')}"
        )
    return service


def explanation_service(feature_set: str | None = None) -> ExplanationService:
    name = _resolve(feature_set)
    service = _explanations.get(name)
    if service is None:
        raise ArtifactError(
            f"Explanation service for '{name}' is unavailable: "
            f"{_errors.get(name, 'not loaded')}"
        )
    return service


def model_version(feature_set: str | None = None) -> str | None:
    try:
        return prediction_service(feature_set).version
    except ArtifactError:
        return None


def model_versions() -> dict[str, str]:
    return {name: svc.version for name, svc in _predictions.items()}


#: How each mode describes its own input. The wording has to name the block the
#: model was actually fitted on: telling a caller that a genomic-only estimate
#: used "this genomic and clinical description" claims an input the model never
#: saw.
_DESCRIPTION_OF_INPUT = {
    "genomic": "with this genomic description",
    "clinical": "whose reported clinical records look like this",
    "merged": "with this genomic description and these reported clinical records",
}


def interpretation(
    probability: float, threshold: float, category: str, feature_set: str
) -> str:
    """Probabilistic wording for the UI. Never a verdict, never a treatment
    recommendation — the archived pipeline returned strings such as
    "Consider hospitalization for immune tolerance induction"."""
    percent = probability * 100
    described = _DESCRIPTION_OF_INPUT.get(
        feature_set, "with this description"
    )
    return (
        f"The model estimates a {percent:.1f}% probability that an F8 mutation "
        f"{described} is reported with inhibitor development, against a decision "
        f"threshold of {threshold * 100:.1f}% ({category.lower()}). This is an "
        f"estimate about the mutation as the source literature reports it, not a "
        f"prediction about an individual patient, and it does not indicate any "
        f"course of treatment."
    )


def input_schema(feature_set: str | None = None) -> dict[str, Any]:
    return prediction_service(feature_set).input_schema()
