"""Prediction and explanation endpoints.

The prediction path stays fast: it validates, transforms, predicts and persists.
Explanations are computed at their own endpoint and cached in the database, so
a slow explainer never delays a prediction.

Three prediction modes are served, matching the feature blocks in
`genomic` (MMC2 only), `clinical` (MMC3 only) and `merged`
(both). Each is a separately trained artifact; the caller picks one per request
and the response says which answered.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend import db
from backend.routers.auth import get_current_user
from backend.routers.patients import _require_patient
from backend.schemas import (
    CaseInput,
    ExplanationResponse,
    MethodExplanation,
    PredictionResponse,
)
from backend.services import ml
from ml.artifacts import ArtifactError
from ml.inference import InputValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Predictions"])

VALID_METHODS = {"shap", "lime"}


@router.get("/predictions/schema")
def prediction_input_schema(
    feature_set: str | None = Query(
        None, description="genomic, clinical or merged. Defaults to the server's mode."
    ),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """The fields one prediction mode accepts, and the values it was fitted on.

    The frontend builds its form from this, so the UI cannot offer a category
    the model was never fitted on, and never has to hardcode a column list.
    """
    try:
        return {
            "model_version": ml.model_version(feature_set),
            "available_feature_sets": ml.available_feature_sets(),
            "default_feature_set": ml.default_feature_set(),
            **ml.input_schema(feature_set),
        }
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post(
    "/patients/{patient_id}/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_prediction(
    patient_id: int,
    payload: CaseInput,
    current_user: dict = Depends(get_current_user),
) -> PredictionResponse:
    _require_patient(patient_id, current_user["id"])

    try:
        service = ml.prediction_service(payload.feature_set)
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    features = payload.to_features()
    try:
        result = service.predict(features).as_dict()
    except InputValidationError as exc:
        # 422 with the accepted values, never a prediction from a guessed input.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_case_input",
                "detail": str(exc),
                "field": exc.field,
                "allowed_values": exc.allowed,
            },
        ) from exc
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    record_id = db.create_case_record(
        patient_id, result["feature_set"], features, payload.mutation_label
    )
    prediction_id = db.create_prediction(patient_id, record_id, result)
    db.write_audit_log(
        current_user["id"], "prediction.create", "predictions", prediction_id
    )

    stored = db.get_prediction(prediction_id, current_user["id"])
    assert stored is not None
    return _to_response(stored)


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
def get_prediction(
    prediction_id: int, current_user: dict = Depends(get_current_user)
) -> PredictionResponse:
    stored = db.get_prediction(prediction_id, current_user["id"])
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found."
        )
    return _to_response(stored)


@router.get("/patients/{patient_id}/history", response_model=list[PredictionResponse])
def prediction_history(
    patient_id: int,
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
) -> list[PredictionResponse]:
    _require_patient(patient_id, current_user["id"])
    rows = db.list_predictions_for_patient(patient_id, current_user["id"], limit=limit)
    return [_to_response(row) for row in rows]


@router.get("/predictions/{prediction_id}/explanation",
            response_model=ExplanationResponse)
def get_explanation(
    prediction_id: int,
    refresh: bool = Query(False, description="Recompute instead of using the stored copy"),
    current_user: dict = Depends(get_current_user),
) -> ExplanationResponse:
    stored = db.get_prediction(prediction_id, current_user["id"])
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found."
        )

    if not refresh:
        cached = db.get_explanations(prediction_id)
        if {"shap", "lime"} <= set(cached):
            return ExplanationResponse(
                prediction_id=prediction_id,
                model_version=stored["model_version"],
                feature_set=stored["feature_set"],
                unit_of_explanation=cached["shap"].get("unit_of_explanation", ""),
                shap=MethodExplanation(**_strip(cached["shap"])),
                lime=MethodExplanation(**_strip(cached["lime"])),
            )

    feature_set = stored["feature_set"]
    try:
        service = ml.prediction_service(feature_set)
        explainer = ml.explanation_service(feature_set)
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    if service.version != stored["model_version"]:
        # Explaining a stored prediction with a different model would attribute
        # one model's probability to another model's feature effects.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"This prediction was made with model '{stored['model_version']}' "
                f"and the server now serves '{service.version}' for the "
                f"'{feature_set}' mode. Re-run the prediction to explain it."
            ),
        )

    features = stored["features"]
    try:
        matrix = service.transform(features)
    except InputValidationError as exc:
        # A stored record that no longer validates means the served vocabulary
        # changed since the prediction was made.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"The stored input for prediction {prediction_id} is no longer "
                f"valid for model '{service.version}': {exc}"
            ),
        ) from exc

    computed = explainer.explain(matrix, features)
    unit = computed["unit_of_explanation"]
    for method in VALID_METHODS:
        db.save_explanation(
            prediction_id, method, {**computed[method], "unit_of_explanation": unit}
        )

    return ExplanationResponse(
        prediction_id=prediction_id,
        model_version=stored["model_version"],
        feature_set=feature_set,
        unit_of_explanation=unit,
        shap=MethodExplanation(**_strip(computed["shap"])),
        lime=MethodExplanation(**_strip(computed["lime"])),
    )


@router.get("/explanations/global")
def global_explanation(
    feature_set: str | None = Query(None),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Model-wide feature importance. Describes the model, not any patient."""
    try:
        explainer = ml.explanation_service(feature_set)
    except ArtifactError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return {
        "model_version": ml.model_version(feature_set),
        **explainer.global_importance(),
    }


# --------------------------------------------------------------------------


def _strip(payload: dict) -> dict:
    """Keep only the fields MethodExplanation declares."""
    allowed = set(MethodExplanation.model_fields)
    return {k: v for k, v in payload.items() if k in allowed}


def _to_response(row: dict) -> PredictionResponse:
    return PredictionResponse(
        id=row["id"],
        patient_id=row["patient_id"],
        prediction=row["prediction"],
        risk=row["risk"],
        probability=row["probability"],
        risk_category=row["risk_category"],
        threshold=row["threshold"],
        model_version=row["model_version"],
        feature_set=row["feature_set"],
        preprocessing_version=row["preprocessing_version"],
        created_at=row["created_at"],
        features=row.get("features", {}),
        mutation_label=row.get("mutation_label"),
        interpretation=ml.interpretation(
            row["probability"], row["threshold"], row["risk_category"]
        ),
    )
