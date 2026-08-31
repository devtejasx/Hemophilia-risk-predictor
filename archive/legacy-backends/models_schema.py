"""
Pydantic Models for API Request/Response Validation
Type-safe API contracts with automatic documentation
"""

from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class SeverityLevel(str, Enum):
    """Hemophilia severity classification"""
    SEVERE = "severe"
    MODERATE = "moderate"
    MILD = "mild"


class MutationType(str, Enum):
    """Common hemophilia mutation types"""
    INTRON_22 = "intron22"
    INTRON_1 = "intron1"
    EXON_DELETION = "exon_deletion"
    POINT_MUTATION = "point_mutation"
    UNKNOWN = "unknown"


class UserRole(str, Enum):
    """User roles for access control"""
    DOCTOR = "doctor"
    NURSE = "nurse"
    LAB_TECH = "lab_tech"
    ADMIN = "admin"
    PATIENT = "patient"


# ==================== Patient Models ====================

class PatientBase(BaseModel):
    """Base patient data"""
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(M|F|Other)$")
    ethnicity: Optional[str] = None
    severity: SeverityLevel
    mutation: MutationType
    dose_intensity: float = Field(..., ge=0.0, le=10000.0)
    exposure_days: int = Field(..., ge=0, le=10000)
    treatment_adherence: Optional[float] = Field(None, ge=0.0, le=100.0)
    previous_inhibitor: Optional[bool] = False
    
    class Config:
        use_enum_values = True


class PatientCreate(PatientBase):
    """Create patient request"""
    pass


class PatientUpdate(BaseModel):
    """Update patient request (all fields optional)"""
    name: Optional[str] = None
    age: Optional[int] = None
    dose_intensity: Optional[float] = None
    treatment_adherence: Optional[float] = None
    previous_inhibitor: Optional[bool] = None


class PatientResponse(PatientBase):
    """Patient response model"""
    id: int
    risk_score: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== Prediction Models ====================

class PredictionInput(BaseModel):
    """ML prediction input"""
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    dose_intensity: float = Field(..., ge=0.0, description="Treatment dose intensity")
    exposure_days: int = Field(..., ge=0, description="Days of factor exposure")
    severity: SeverityLevel = Field(..., description="Hemophilia severity")
    mutation: MutationType = Field(..., description="Mutation type")
    hemoglobin: Optional[float] = Field(None, ge=5.0, le=20.0)
    white_blood_cells: Optional[float] = Field(None, ge=0.0, le=30.0)
    platelets: Optional[float] = Field(None, ge=0.0, le=500.0)
    
    class Config:
        use_enum_values = True


class FeatureImportance(BaseModel):
    """Feature importance scores"""
    feature: str
    importance_score: float
    impact: str  # "High", "Medium", "Low"


class PredictionOutput(BaseModel):
    """ML prediction output"""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    risk_category: str = Field(..., description="Low/Medium/High")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    top_features: List[FeatureImportance]
    explanation: str
    timestamp: datetime
    model_version: str = "2.0.0"


# ==================== ChatBot Models ====================

class ChatMessage(BaseModel):
    """Chat message"""
    user_id: int
    patient_id: Optional[int] = None
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[str] = Field(None, description="Chat mode (diagnosis, treatment, etc)")


class ChatResponse(BaseModel):
    """Chat response"""
    message_id: str
    response: str = Field(..., description="AI response")
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    timestamp: datetime
    

class ConversationHistory(BaseModel):
    """Conversation history"""
    patient_id: int
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime


# ==================== Analytics Models ====================

class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_patients: int
    high_risk_patients: int
    medium_risk_patients: int
    low_risk_patients: int
    average_risk_score: float
    total_predictions: int
    inhibitor_rate: float
    avg_treatment_adherence: float
    predictions_today: int
    active_chats: int
    timestamp: datetime


class PatientTrend(BaseModel):
    """Patient trend data"""
    date: datetime
    risk_score: float
    hemoglobin: Optional[float]
    treatment_adherence: Optional[float]
    notes: Optional[str]


# ==================== User Models ====================

class UserRegister(BaseModel):
    """User registration"""
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, description="Must be 8+ chars, include uppercase, lowercase, number, special char")
    full_name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.NURSE
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        if not any(c in "!@#$%^&*" for c in v):
            raise ValueError("Password must contain special character (!@#$%^&*)")
        return v


class UserLogin(BaseModel):
    """User login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    username: str
    role: UserRole


class UserResponse(BaseModel):
    """User response model"""
    id: int
    username: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ValidationError(BaseModel):
    """Validation error response"""
    field: str
    message: str
    value: Optional[Any] = None


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details"""
    errors: List[ValidationError]


# ==================== Monitoring Models ====================

class MonitoringRecord(BaseModel):
    """Patient monitoring record"""
    patient_id: int
    hemoglobin: float
    white_blood_cells: float
    platelets: float
    factor_level: Optional[float] = None
    pt_inr: Optional[float] = None
    aptt: Optional[float] = None
    bleeding_episodes: int = 0
    recorded_at: datetime
    recorded_by: Optional[int] = None
    notes: Optional[str] = None


class TreatmentRecord(BaseModel):
    """Treatment administration record"""
    patient_id: int
    product_name: str
    dose_units: float
    administration_date: datetime
    administered_by: int
    route: str = "IV"
    indications: Optional[str] = None
    response: Optional[str] = None
    adverse_events: Optional[str] = None
