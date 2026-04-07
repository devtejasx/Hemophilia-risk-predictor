"""
Patient Management API routes
Requires: Authenticated user (doctor/admin for full access, patient for own data)
Roles: doctor, admin (full CRUD), patient (limited access)
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from database
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from database import (
    add_patient, get_patient, get_all_patients, 
    update_patient, delete_patient
)

from ..models import PatientResponse, PatientCreateRequest, PatientUpdateRequest
from ..security import get_current_user, get_current_doctor, get_current_admin

router = APIRouter(prefix="/patients", tags=["Patients"])
logger = logging.getLogger(__name__)


@router.get("", response_model=List[PatientResponse])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    severity: Optional[str] = None,
    risk_above: Optional[float] = None,
    current_user: dict = Depends(get_current_doctor)
) -> List[PatientResponse]:
    """
    List all patients with optional filtering
    
    **Authentication:** Required
    **Roles:** doctor, admin
    
    Query parameters:
    - skip: Number of records to skip (pagination)
    - limit: Number of records to return (max 100)
    - severity: Filter by severity (Mild, Moderate, Severe)
    - risk_above: Filter patients with risk above threshold
    """
    try:
        patients = get_all_patients()
        
        # Apply filters
        if severity:
            patients = [p for p in patients if p.get('Severity', '').lower() == severity.lower()]
        
        if risk_above is not None:
            patients = [p for p in patients if p.get('Risk', 0) >= risk_above]
        
        # Apply pagination
        total = len(patients)
        patients = patients[skip:skip + limit]
        
        # Convert to response format
        responses = []
        for p in patients:
            responses.append(PatientResponse(
                id=p.get('id', 0),
                patient_data=p,
                created_at=datetime.fromisoformat(p.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(p.get('updated_at', datetime.now().isoformat())),
                last_prediction=None,
                last_risk_score=p.get('Risk')
            ))
        
        logger.info(f"[{current_user['username']}] Listed {len(responses)} patients (total: {total})")
        return responses
    
    except Exception as e:
        logger.error(f"Error listing patients: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list patients: {str(e)}")


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient_detail(
    patient_id: int,
    current_user: dict = Depends(get_current_doctor)
) -> PatientResponse:
    """
    Get specific patient details
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patient = get_patient(patient_id)
        
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        logger.info(f"[{current_user['username']}] Retrieved patient {patient_id}")
        
        return PatientResponse(
            id=patient_id,
            patient_data=patient,
            created_at=datetime.fromisoformat(patient.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(patient.get('updated_at', datetime.now().isoformat())),
            last_prediction=None,
            last_risk_score=patient.get('Risk')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get patient: {str(e)}")


@router.post("", response_model=PatientResponse)
async def create_patient(
    request: PatientCreateRequest,
    current_user: dict = Depends(get_current_doctor)
) -> PatientResponse:
    """
    Create new patient record
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patient_dict = request.patient_data.dict()
        
        # Add to database
        patient_id = add_patient(patient_dict, request.notes)
        
        # Retrieve created patient
        patient = get_patient(patient_id)
        
        logger.info(f"[{current_user['username']}] Created new patient {patient_id}: {patient.get('name')}")
        
        return PatientResponse(
            id=patient_id,
            patient_data=patient,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prediction=None,
            last_risk_score=patient.get('Risk')
        )
    
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient_record(
    patient_id: int,
    request: PatientUpdateRequest,
    current_user: dict = Depends(get_current_doctor)
) -> PatientResponse:
    """
    Update existing patient record
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        # Check if patient exists
        patient = get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Update patient
        patient_dict = request.patient_data.dict()
        update_patient(patient_id, patient_dict, current_user["id"] if "id" in current_user else None)
        
        # Retrieve updated patient
        updated = get_patient(patient_id)
        
        logger.info(f"[{current_user['username']}] Updated patient {patient_id}")
        
        return PatientResponse(
            id=patient_id,
            patient_data=updated,
            created_at=datetime.fromisoformat(updated.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.now(),
            last_prediction=None,
            last_risk_score=updated.get('Risk')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update patient: {str(e)}")


@router.delete("/{patient_id}")
async def delete_patient_record(
    patient_id: int,
    current_user: dict = Depends(get_current_admin)
):
    """
    Delete patient record
    
    **Authentication:** Required
    **Roles:** admin only
    """
    try:
        # Check if patient exists
        patient = get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Delete patient
        delete_patient(patient_id)
        
        logger.info(f"[{current_user['username']}] Deleted patient {patient_id}")
        
        return {
            "status": "deleted",
            "patient_id": patient_id,
            "message": f"Patient {patient_id} has been deleted"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete patient: {str(e)}")


@router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: int,
    limit: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(get_current_doctor)
):
    """
    Get patient's prediction/visit history
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patient = get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        logger.info(f"[{current_user['username']}] Retrieved patient {patient_id} history")
        
        # Return placeholder (implement with database history)
        return {
            "patient_id": patient_id,
            "history": [],
            "message": "History tracking to be implemented with audit log"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
