"""Patient records. Every query is scoped to the authenticated user."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend import db
from backend.routers.auth import get_current_user
from backend.schemas import PatientCreate, PatientResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/patients", tags=["Patients"])


def _require_patient(patient_id: int, user_id: int) -> dict:
    patient = db.get_patient(patient_id, user_id)
    if patient is None:
        # 404 rather than 403: a user must not learn that another account's
        # patient id exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )
    return patient


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: PatientCreate, current_user: dict = Depends(get_current_user)
) -> PatientResponse:
    try:
        patient = db.create_patient(
            user_id=current_user["id"],
            identifier=payload.identifier,
            display_name=payload.display_name,
            notes=payload.notes,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You already have a patient with identifier "
                   f"'{payload.identifier}'.",
        ) from None
    db.write_audit_log(current_user["id"], "patient.create", "patients", patient["id"])
    return PatientResponse(**patient)


@router.get("", response_model=list[PatientResponse])
def list_patients(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
) -> list[PatientResponse]:
    return [
        PatientResponse(**row)
        for row in db.list_patients(current_user["id"], limit=limit, offset=offset)
    ]


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int, current_user: dict = Depends(get_current_user)
) -> PatientResponse:
    patient = _require_patient(patient_id, current_user["id"])
    enriched = next(
        (
            row
            for row in db.list_patients(current_user["id"], limit=500)
            if row["id"] == patient_id
        ),
        patient,
    )
    return PatientResponse(**enriched)


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int, current_user: dict = Depends(get_current_user)
) -> None:
    if not db.delete_patient(patient_id, current_user["id"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found."
        )
    db.write_audit_log(current_user["id"], "patient.delete", "patients", patient_id)
