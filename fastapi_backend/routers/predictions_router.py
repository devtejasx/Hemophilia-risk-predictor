"""
Prediction Router
Endpoints for ML predictions
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from datetime import datetime
from models import PredictionInput, PredictionOutput
from services.prediction_service import prediction_service
from services.patient_service import get_patient_service, PatientService
from database import get_db
from exceptions import PredictionException, exception_to_http_exception

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


@router.post("/", response_model=PredictionOutput)
async def create_prediction(
    prediction_input: PredictionInput,
    patient_id: int = None,
    db = Depends(get_db)
):
    """
    Generate ML prediction for patient
    
    Args:
        prediction_input: Patient clinical data
        patient_id: Optional patient ID
        db: Database connection
        
    Returns:
        Prediction with risk score and explanations
    """
    try:
        # Generate prediction
        result = prediction_service.predict(prediction_input, patient_id)
        
        # Save to database if patient_id provided
        if patient_id and db:
            try:
                query = '''
                    INSERT INTO predictions 
                    (patient_id, risk_score, risk_category, confidence, model_version)
                    VALUES (?, ?, ?, ?, ?)
                '''
                cursor = db.cursor()
                cursor.execute(query, (
                    patient_id,
                    result['risk_score'],
                    result['risk_category'],
                    result['confidence'],
                    "v1.0"
                ))
                
                # Update patient risk score
                cursor.execute(
                    'UPDATE patients SET risk_score = ? WHERE id = ?',
                    (result['risk_score'], patient_id)
                )
            except Exception as e:
                print(f"Warning: Failed to save prediction to database: {e}")
        
        return PredictionOutput(**result)
        
    except PredictionException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/batch", response_model=List[PredictionOutput])
async def batch_predictions(
    predictions: List[PredictionInput]
):
    """
    Generate predictions for multiple patients
    
    Args:
        predictions: List of patient data
        
    Returns:
        List of predictions
    """
    try:
        results = []
        for pred_input in predictions:
            result = prediction_service.predict(pred_input)
            results.append(PredictionOutput(**result))
        
        return results
        
    except PredictionException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/patient/{patient_id}")
async def get_patient_predictions(
    patient_id: int,
    db = Depends(get_db)
):
    """
    Get prediction history for patient
    
    Args:
        patient_id: Patient ID
        db: Database connection
        
    Returns:
        List of past predictions
    """
    try:
        query = '''
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
        '''
        cursor = db.cursor()
        cursor.execute(query, (patient_id,))
        predictions = cursor.fetchall()
        
        if not predictions:
            return {"predictions": [], "patient_id": patient_id}
        
        return {
            "predictions": [dict(row) for row in predictions],
            "patient_id": patient_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
