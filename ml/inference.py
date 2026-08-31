"""The single inference entrypoint for the whole system.

Backend, scripts and tests all call ``PredictionService.predict``. There is no
second copy of the feature-building logic anywhere — that duplication is what
allowed training and serving to drift apart in the previous implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.artifacts import ArtifactBundle, ArtifactError, load_bundle
from ml.preprocessing import champ

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "champ-v1"

#: Risk bands. The boundary between "lower" and "elevated" is the model's own
#: calibrated threshold, not a hardcoded 0.5. The wording is deliberately
#: probabilistic: this is decision support, not a verdict.
RISK_LOWER = "Lower estimated risk"
RISK_ELEVATED = "Elevated estimated risk"


class InputValidationError(ValueError):
    """Input cannot be mapped onto the feature space the model was fitted on."""

    def __init__(self, message: str, field: str | None = None,
                 allowed: list[str] | None = None) -> None:
        super().__init__(message)
        self.field = field
        self.allowed = allowed or []


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    risk_category: str
    threshold: float
    model_version: str
    preprocessing_version: str
    features_used: list[str]
    provenance_warning: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "risk_category": self.risk_category,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "preprocessing_version": self.preprocessing_version,
            "features_used": self.features_used,
            "provenance_warning": self.provenance_warning,
        }


class PredictionService:
    """Loads one model version once and serves predictions from it."""

    def __init__(
        self,
        version: str = DEFAULT_MODEL_VERSION,
        artifacts_dir: str | None = None,
    ) -> None:
        self.bundle: ArtifactBundle = load_bundle(version, artifacts_dir)
        self._categories = champ.fitted_categories(self.bundle.preprocessor)

    # -- introspection ----------------------------------------------------

    @property
    def version(self) -> str:
        return self.bundle.version

    def input_schema(self) -> dict[str, Any]:
        """What a caller must supply, and which values are accepted.

        The frontend builds its form from this, so the UI cannot offer a value
        the model has never been fitted on.
        """
        return {
            "categorical": {
                col: sorted(v for v in values if v != champ.MISSING_CATEGORY)
                for col, values in self._categories.items()
            },
            "numeric": {
                "exon_number": {
                    "description": "Exon or intron number carrying the variant",
                    "required": False,
                },
                "codon_number": {
                    "description": "Codon position of the variant",
                    "required": False,
                },
            },
            "boolean": {
                "is_intron": {
                    "description": "True if the variant lies in an intron rather than an exon",
                    "required": False,
                }
            },
            "required": list(champ.CATEGORICAL_FEATURES),
        }

    # -- validation -------------------------------------------------------

    def validate(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Turn a raw input dict into a single-row frame in training column order.

        Every categorical value is checked against the vocabulary the
        preprocessor was actually fitted on. An unknown value raises with the
        allowed list attached, so the API can return a useful 422 instead of a
        confident prediction about a row the model has never seen.
        """
        if not isinstance(payload, dict):
            raise InputValidationError("Input must be an object of feature values.")

        row: dict[str, Any] = {}

        for col in champ.CATEGORICAL_FEATURES:
            raw = payload.get(col, payload.get(_snake(col)))
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                raise InputValidationError(
                    f"'{col}' is required.", field=col,
                    allowed=sorted(v for v in self._categories[col]
                                   if v != champ.MISSING_CATEGORY),
                )
            value = str(raw).strip()
            if col in champ._CASE_INSENSITIVE_COLUMNS:
                value = champ.CASE_CANONICAL_FORMS.get(value.casefold(), value)
            allowed = self._categories[col]
            if value not in allowed:
                raise InputValidationError(
                    f"'{value}' is not a value of '{col}' that this model was "
                    f"trained on.",
                    field=col,
                    allowed=sorted(v for v in allowed if v != champ.MISSING_CATEGORY),
                )
            row[col] = value

        for col in champ.NUMERIC_FEATURES:
            raw = payload.get(col, payload.get(_snake(col)))
            if raw is None or raw == "":
                row[col] = np.nan  # imputed with the training median
                continue
            try:
                row[col] = float(raw)
            except (TypeError, ValueError):
                raise InputValidationError(
                    f"'{col}' must be a number, got {raw!r}.", field=col
                ) from None

        raw_intron = payload.get("is_intron", payload.get("isIntron", 0))
        row["is_intron"] = int(bool(raw_intron)) if not isinstance(raw_intron, str) \
            else int(raw_intron.strip().lower() in {"1", "true", "yes", "y"})

        return pd.DataFrame([row], columns=champ.FEATURE_COLUMNS)

    # -- prediction -------------------------------------------------------

    def transform(self, payload: dict[str, Any]) -> np.ndarray:
        """Validated input -> the exact matrix the estimator was fitted on."""
        frame = self.validate(payload)
        return self.bundle.preprocessor.transform(frame)

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        """Estimate the probability of inhibitor development for a variant.

        Note this is a *variant-attributable* estimate. CHAMP rows are F8
        variants and its inhibitor field is a registry aggregation, so the
        result is not an individual patient's probability.

        Raises InputValidationError for bad input and ArtifactError for a
        broken model. It never falls back to a hand-written formula: a caller
        must be able to distinguish a prediction from a failure.
        """
        matrix = self.transform(payload)

        try:
            probability = float(self.bundle.model.predict_proba(matrix)[0][1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ArtifactError(
                f"Model '{self.version}' failed to produce a probability: {exc}"
            ) from exc

        threshold = self.bundle.threshold
        category = RISK_ELEVATED if probability >= threshold else RISK_LOWER

        return PredictionResult(
            probability=round(probability, 6),
            risk_category=category,
            threshold=threshold,
            model_version=self.version,
            preprocessing_version=str(
                self.bundle.metadata.get("preprocessing_version", "unknown")
            ),
            features_used=list(champ.FEATURE_COLUMNS),
            provenance_warning=self.bundle.provenance_warning,
        )


def _snake(column: str) -> str:
    """'Variant Type' -> 'variant_type', so JSON clients may use either form."""
    return column.lower().replace(" ", "_")


_default_service: PredictionService | None = None


def get_prediction_service(
    version: str | None = None, artifacts_dir: str | None = None
) -> PredictionService:
    """Process-wide singleton, so the model is loaded once per process."""
    global _default_service
    if _default_service is None or (version and _default_service.version != version):
        _default_service = PredictionService(
            version or DEFAULT_MODEL_VERSION, artifacts_dir
        )
    return _default_service


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper around the default service."""
    return get_prediction_service().predict(payload).as_dict()
