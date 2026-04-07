"""
Patients Router
Endpoints for patient CRUD operations
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from models import PatientCreate, PatientUpdate, PatientResponse
from services.patient_service import PatientService, get_patient_service
from database import get_db
from exceptions import PatientNotFound, DatabaseException, exception_to_http_exception

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    db = Depends(get_db)
):
    """
    Create new patient
    
    Args:
        patient: Patient data
        db: Database connection
        
    Returns:
        Created patient with ID
    """
    try:
        service = get_patient_service(db)
        return service.create_patient(patient)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db = Depends(get_db)
):
    """
    Get patient by ID
    
    Args:
        patient_id: Patient ID
        db: Database connection
        
    Returns:
        Patient data
    """
    try:
        service = get_patient_service(db)
        return service.get_patient(patient_id)
    except PatientNotFound as e:
        raise exception_to_http_exception(e)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/", response_model=List[PatientResponse])
async def list_patients(
    skip: int = 0,
    limit: int = 100,
    db = Depends(get_db)
):
    """
    List all patients with pagination
    
    Args:
        skip: Number of patients to skip
        limit: Maximum number of patients to return
        db: Database connection
        
    Returns:
        List of patients
    """
    try:
        service = get_patient_service(db)
        return service.get_all_patients(skip, limit)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_update: PatientUpdate,
    db = Depends(get_db)
):
    """
    Update patient data
    
    Args:
        patient_id: Patient ID
        patient_update: Updated patient data
        db: Database connection
        
    Returns:
        Updated patient
    """
    try:
        service = get_patient_service(db)
        return service.update_patient(patient_id, patient_update)
    except PatientNotFound as e:
        raise exception_to_http_exception(e)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: int,
    db = Depends(get_db)
):
    """
    Delete patient
    
    Args:
        patient_id: Patient ID
        db: Database connection
    """
    try:
        service = get_patient_service(db)
        service.delete_patient(patient_id)
    except PatientNotFound as e:
        raise exception_to_http_exception(e)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/search/by-severity")
async def search_by_severity(
    severity: str,
    db = Depends(get_db)
):
    """
    Search patients by severity
    
    Args:
        severity: Severity level (severe, moderate, mild)
        db: Database connection
        
    Returns:
        List of matching patients
    """
    try:
        service = get_patient_service(db)
        return service.search_patients(severity=severity)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/search/by-mutation")
async def search_by_mutation(
    mutation: str,
    db = Depends(get_db)
):
    """
    Search patients by mutation type
    
    Args:
        mutation: Mutation type
        db: Database connection
        
    Returns:
        List of matching patients
    """
    try:
        service = get_patient_service(db)
        return service.search_patients(mutation=mutation)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
