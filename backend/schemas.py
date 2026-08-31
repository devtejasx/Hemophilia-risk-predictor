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
    """Only identity fields. No clinical variables are stored, because none of
    them feed the model — the predictive input is the genomic profile."""

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


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


class GenomicInput(BaseModel):
    """A CHAMP-compatible F8 variant description.

    Field names use the CHAMP column names via aliases so the payload is
    self-describing. Accepted values come from GET /api/predictions/schema,
    which reports the vocabulary the model was actually fitted on.
    """

    variant_type: str = Field(alias="Variant Type")
    mechanism: str = Field(alias="Mechanism")
    domain: str = Field(alias="Domain")
    subtype: str = Field(alias="Subtype")
    in_poly_a: str = Field(alias="In Poly A")
    reported_clinical_severity: str = Field(alias="Reported Clinical Severity")
    exon_number: float | None = Field(default=None, ge=0, le=200)
    codon_number: float | None = Field(default=None, ge=0, le=10000)
    is_intron: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("variant_type", "mechanism", "domain", "subtype",
                     "in_poly_a", "reported_clinical_severity")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    def to_features(self) -> dict[str, Any]:
        """The dict shape ml.inference.PredictionService expects."""
        return {
            "Variant Type": self.variant_type,
            "Mechanism": self.mechanism,
            "Domain": self.domain,
            "Subtype": self.subtype,
            "In Poly A": self.in_poly_a,
            "Reported Clinical Severity": self.reported_clinical_severity,
            "exon_number": self.exon_number,
            "codon_number": self.codon_number,
            "is_intron": int(self.is_intron),
        }


class PredictionResponse(BaseModel):
    id: int
    patient_id: int
    probability: float = Field(description="Calibrated probability in [0, 1]")
    risk_category: str
    threshold: float = Field(description="The model's own decision threshold")
    model_version: str
    preprocessing_version: str
    created_at: str
    interpretation: str
    disclaimer: str = DISCLAIMER


class ContributionResponse(BaseModel):
    feature: str
    value: Any = None
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
    variant_type_distribution: dict[str, int]
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
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    field: str | None = None
    allowed_values: list[str] | None = None
