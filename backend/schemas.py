"""Pydantic request/response models — the API contract the frontend types mirror."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator

DISCLAIMER = (
    "Research decision-support prototype. This estimate is not intended for "
    "standalone diagnosis or treatment decisions."
)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=12, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str


# --------------------------------------------------------------------------
# Patients
# --------------------------------------------------------------------------


class PatientCreate(BaseModel):
    """Only identity fields. Clinical and genomic variables live on the case
    record instead, because those are the fields the model actually consumes."""

    identifier: str = Field(min_length=1, max_length=64,
                            description="Your own reference for this patient")
    display_name: str = Field(min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


class PatientResponse(BaseModel):
    id: int
    identifier: str
    display_name: str
    notes: str | None = None
    created_at: str
    updated_at: str | None = None
    prediction_count: int = 0
    latest_probability: float | None = None
    latest_risk_category: str | None = None
    latest_risk: str | None = None


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


FEATURE_SETS = ("genomic", "clinical", "merged")


class CaseInput(BaseModel):
    """One MMC2/MMC3-shaped record to score.

    The accepted keys are not fixed by this class: they are the source columns
    of whichever feature set the request targets, reported by
    ``GET /api/predictions/schema``. Pinning them here would mean editing the
    API every time the dataset or the feature resolution changes, and would let
    the two disagree — which is exactly the drift the ml package exists to
    prevent. Validation of the values themselves happens once, in
    ``ml.inference.PredictionService.validate``, against the vocabulary the
    served preprocessor was actually fitted on.

    ``features`` therefore carries the raw column-name keys, e.g.::

        {"mut_type": "Point", "mut_effect": "Missense", "cli_phe": "Severe"}
    """

    feature_set: str = Field(
        default="merged",
        description="Prediction mode: genomic, clinical or merged.",
    )
    features: dict[str, Any] = Field(
        description="Source column name -> value. See GET /api/predictions/schema."
    )
    mutation_label: str | None = Field(
        default=None,
        max_length=200,
        description="Your own label for this mutation. Never reaches the model.",
    )

    @field_validator("feature_set")
    @classmethod
    def _known_feature_set(cls, value: str) -> str:
        name = (value or "").strip().lower()
        if name not in FEATURE_SETS:
            raise ValueError(f"must be one of {', '.join(FEATURE_SETS)}")
        return name

    @field_validator("features")
    @classmethod
    def _not_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("must contain at least one feature")
        return value

    def to_features(self) -> dict[str, Any]:
        """The dict shape ml.inference.PredictionService expects.

        Blank strings are dropped rather than forwarded: an untouched optional
        form field must read as "not supplied", not as the empty category.
        """
        return {
            key: value
            for key, value in self.features.items()
            if not (isinstance(value, str) and not value.strip())
            and value is not None
        }


class PredictionResponse(BaseModel):
    id: int
    patient_id: int
    prediction: int = Field(description="1 = inhibitor risk above the threshold")
    risk: str = Field(description='"High" or "Low"')
    probability: float = Field(description="Calibrated probability in [0, 1]")
    risk_category: str
    threshold: float = Field(description="The model's own decision threshold")
    model_version: str
    feature_set: str
    preprocessing_version: str
    created_at: str
    interpretation: str
    features: dict[str, Any] = Field(default_factory=dict)
    mutation_label: str | None = None
    disclaimer: str = DISCLAIMER


class ContributionResponse(BaseModel):
    feature: str
    label: str = ""
    value: Any = None
    supplied: bool = True
    contribution: float
    direction: str


class MethodExplanation(BaseModel):
    available: bool
    reason: str | None = None
    basis: str | None = None
    base_value: float | None = None
    local_prediction: float | None = None
    contributions: list[ContributionResponse] = Field(default_factory=list)


class ExplanationResponse(BaseModel):
    prediction_id: int
    model_version: str
    feature_set: str = ""
    unit_of_explanation: str
    shap: MethodExplanation | None = None
    lime: MethodExplanation | None = None
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------
# Analytics / health
# --------------------------------------------------------------------------


class AnalyticsResponse(BaseModel):
    total_patients: int
    total_predictions: int
    mean_probability: float | None = None
    risk_distribution: dict[str, int]
    mutation_type_distribution: dict[str, int]
    feature_set_distribution: dict[str, int]
    model_version: str
    note: str = (
        "Counts are computed from predictions stored by this account. They "
        "describe usage of this prototype, not clinical outcomes."
    )


class HealthResponse(BaseModel):
    status: str
    database: str
    model: str
    model_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    field: str | None = None
    allowed_values: list[str] | None = None
