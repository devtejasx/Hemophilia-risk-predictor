"""
Example Protected Routes
Demonstrates how to implement protected endpoints using authentication dependencies
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from auth_dependencies import (
    get_current_user, require_admin, require_doctor,
    require_verified, get_db, get_current_user_optional
)
from auth_models import User
from pydantic import BaseModel

# Create router
router = APIRouter(prefix="/api/protected", tags=["Protected Routes"])


# ============================================================================
# EXAMPLE SCHEMAS
# ============================================================================

class PatientData(BaseModel):
    """Example patient data"""
    patient_id: str
    name: str
    age: int
    diagnosis: str


class PredictionResult(BaseModel):
    """Example prediction result"""
    risk_score: float
    risk_label: str
    confidence: float


# ============================================================================
# EXAMPLE 1: BASIC AUTHENTICATION REQUIRED
# ============================================================================

@router.get(
    "/patient-data",
    response_model=list[PatientData],
    summary="Get patient data (authenticated users)",
)
async def get_patient_data(
    current_user: User = Depends(get_current_user)
):
    """Get patient data - requires authentication
    
    Any authenticated user can access this
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List of patient data
    """
    # Example: Return user-specific data
    return [
        PatientData(
            patient_id="P001",
            name="John Doe",
            age=52,
            diagnosis="Hypertension"
        ),
        PatientData(
            patient_id="P002",
            name="Jane Smith",
            age=45,
            diagnosis="Diabetes"
        ),
    ]


# ============================================================================
# EXAMPLE 2: DOCTOR ROLE REQUIRED
# ============================================================================

@router.post(
    "/predict",
    response_model=PredictionResult,
    summary="Make prediction (doctors only)",
)
async def make_prediction(
    patient_data: PatientData,
    current_doctor: User = Depends(require_doctor)
):
    """Make ML prediction - requires doctor role
    
    Only users with doctor role can access this
    
    Args:
        patient_data: Patient data for prediction
        current_doctor: Current authenticated doctor user
        
    Returns:
        Prediction result
    """
    # Example: Mock ML prediction
    return PredictionResult(
        risk_score=0.65,
        risk_label="HIGH",
        confidence=0.92
    )


# ============================================================================
# EXAMPLE 3: ADMIN ROLE REQUIRED
# ============================================================================

@router.delete(
    "/patient-data/{patient_id}",
    summary="Delete patient (admin only)",
)
async def delete_patient_data(
    patient_id: str,
    current_admin: User = Depends(require_admin)
):
    """Delete patient data - requires admin role
    
    Only admins can permanently delete data
    
    Args:
        patient_id: Patient ID to delete
        current_admin: Current authenticated admin user
        
    Returns:
        Success message
    """
    return {
        "message": f"Patient {patient_id} deleted successfully",
        "deleted_by": current_admin.email
    }


# ============================================================================
# EXAMPLE 4: ROLE-BASED WITH VERIFICATION
# ============================================================================

@router.post(
    "/sensitive-operation",
    summary="Sensitive operation (verified doctors only)",
)
async def sensitive_operation(
    data: dict,
    current_doctor: User = Depends(require_doctor),
):
    """Sensitive operation - requires doctor role and verification
    
    Note: require_verified is in auth_dependencies
    
    Args:
        data: Operation data
        current_doctor: Current authenticated verified doctor
        
    Returns:
        Operation result
    """
    if not current_doctor.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required for this operation"
        )
    
    return {
        "message": "Operation completed successfully",
        "performed_by": current_doctor.email,
        "role": current_doctor.role.value
    }


# ============================================================================
# EXAMPLE 5: OPTIONAL AUTHENTICATION
# ============================================================================

@router.get(
    "/public-data",
    summary="Public data (auth optional)",
)
async def get_public_data(
    current_user: User = Depends(get_current_user_optional)
):
    """Get public data - authentication optional
    
    Endpoint works for both authenticated and anonymous users
    
    Args:
        current_user: Current user if authenticated, None otherwise
        
    Returns:
        Data (filtered based on authentication status)
    """
    base_data = {
        "public_info": "Available to everyone",
        "timestamp": "2026-04-07T10:00:00Z"
    }
    
    if current_user:
        # Add user-specific data
        base_data["user_info"] = {
            "authenticated": True,
            "username": current_user.username,
            "role": current_user.role.value
        }
    else:
        base_data["user_info"] = {
            "authenticated": False,
            "note": "Login to see personalized data"
        }
    
    return base_data


# ============================================================================
# EXAMPLE 6: CHECKING USER PROPERTIES
# ============================================================================

@router.get(
    "/user-stats",
    summary="Get user statistics",
)
async def get_user_stats(
    current_user: User = Depends(get_current_user)
):
    """Get user statistics - demonstrate property access
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User statistics
    """
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "username": current_user.username,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
    }


# ============================================================================
# EXAMPLE 7: DATABASE OPERATIONS IN PROTECTED ROUTE
# ============================================================================

@router.post(
    "/save-data",
    summary="Save user-specific data",
)
async def save_user_data(
    data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save data for current user
    
    Demonstrates database access in protected route
    
    Args:
        data: Data to save
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Saved data info
    """
    # Example: Update user data in database
    from sqlalchemy import text
    
    try:
        # This is just an example - implement actual logic
        return {
            "message": "Data saved successfully",
            "user_id": current_user.user_id,
            "data": data,
            "saved_by": current_user.email
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving data: {str(e)}"
        )


# ============================================================================
# EXAMPLE 8: ROLE-SPECIFIC DASHBOARD
# ============================================================================

@router.get(
    "/dashboard",
    summary="Get role-specific dashboard",
)
async def get_dashboard(
    current_user: User = Depends(get_current_user)
):
    """Get dashboard based on user role
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Role-specific dashboard data
    """
    role = current_user.role.value
    
    if role == "admin":
        return {
            "dashboard_type": "Admin Dashboard",
            "sections": [
                "User Management",
                "System Statistics",
                "Security Logs",
                "Configuration",
            ],
            "widgets": {
                "total_users": 127,
                "active_sessions": 23,
                "system_health": "Healthy"
            }
        }
    
    elif role == "doctor":
        return {
            "dashboard_type": "Doctor Dashboard",
            "sections": [
                "My Patients",
                "Recent Predictions",
                "Clinical Alerts",
                "Patient History",
            ],
            "widgets": {
                "patients_today": 14,
                "pending_reviews": 3,
                "model_accuracy": "89%"
            }
        }
    
    else:  # patient
        return {
            "dashboard_type": "Patient Dashboard",
            "sections": [
                "My Health",
                "Appointments",
                "Results",
                "Messages",
            ],
            "widgets": {
                "last_checkup": "2026-03-28",
                "recent_risk_score": 0.42,
                "status": "Stable"
            }
        }


# ============================================================================
# EXAMPLE 9: MULTI-ROLE ACCESS
# ============================================================================

@router.get(
    "/reports",
    summary="Access reports (doctors and admin)",
)
async def get_reports(
    current_user: User = Depends(get_current_user)
):
    """Access reports - allowed for doctors and admins
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Reports (or error if not authorized)
    """
    allowed_roles = ["doctor", "admin"]
    
    if current_user.role.value not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Only {', '.join(allowed_roles)} can access reports"
        )
    
    return {
        "reports": [
            {"id": "R001", "title": "Monthly Summary", "created": "2026-04-01"},
            {"id": "R002", "title": "Patient Outcomes", "created": "2026-03-31"},
            {"id": "R003", "title": "Risk Analysis", "created": "2026-03-28"},
        ],
        "accessed_by": current_user.email
    }
