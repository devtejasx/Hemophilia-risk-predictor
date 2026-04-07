"""
ML Prediction API routes
Requires: Authenticated user (patient or doctor)
Roles: All authenticated users can make predictions
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from ..models import PredictionRequest, PredictionResponse
from ..ml_utils import predict_inhibitor_risk
from ..security import get_current_user, get_current_doctor

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)


@router.post("", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
) -> PredictionResponse:
    """
    Predict inhibitor development risk using ML models
    
    **Authentication:** Required (any authenticated user)
    
    **Roles:** patient, doctor, admin
    
    Takes clinical parameters and returns risk prediction with recommendations.
    """
    try:
        result = predict_inhibitor_risk(
            age=request.age,
            dose=request.dose,
            exposure=request.exposure,
            severity=request.severity,
            mutation=request.mutation,
            ethnicity=request.ethnicity,
            blood_type=request.blood_type,
            hla_typing=request.hla_typing,
            product_type=request.product_type,
            treatment_adherence=request.treatment_adherence,
            family_history=request.family_history,
            previous_inhibitor=request.previous_inhibitor,
            joint_damage_score=request.joint_damage_score,
            bleeding_episodes=request.bleeding_episodes,
            baseline_factor_level=request.baseline_factor_level,
            immunosuppression=request.immunosuppression,
            active_infection=request.active_infection,
            vaccination_status=request.vaccination_status,
            physical_activity=request.physical_activity,
            stress_level=request.stress_level,
            comorbidities=request.comorbidities
        )
        
        logger.info(f"[{current_user['username']}] Prediction: {result['risk_category']} risk")
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/batch")
async def batch_predict(
    patient_ids: list[int],
    current_user: dict = Depends(get_current_doctor)
) -> dict:
    """
    Batch predict risk for multiple patients
    
    **Authentication:** Required
    **Roles:** doctor, admin
    
    Stub for future implementation
    """
    return {
        "message": "Batch prediction endpoint",
        "patients": len(patient_ids),
        "status": "Not implemented yet"
    }


@router.get("/history/{patient_id}")
async def get_prediction_history(
    patient_id: int,
    current_user: dict = Depends(get_current_doctor)
):
    """
    Get prediction history for a patient
    
    **Authentication:** Required
    **Roles:** doctor, admin
    
    Returns historical predictions for a patient
    """
    return {
        "patient_id": patient_id,
        "predictions": [],
        "status": "Not implemented yet"
    }
