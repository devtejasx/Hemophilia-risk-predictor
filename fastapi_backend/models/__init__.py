"""
Pydantic Models/Schemas
Request and response data models with validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class SeverityLevel(str, Enum):
    """Hemophilia severity levels"""
    SEVERE = "severe"
    MODERATE = "moderate"
    MILD = "mild"


class MutationType(str, Enum):
    """Common mutation types"""
    INTRON_22 = "intron22"
    INTRON_1 = "intron1"
    EXON_DELETION = "exon_deletion"
    POINT_MUTATION = "point_mutation"
    UNKNOWN = "unknown"


# ==================== Patient Models ====================

class PatientBase(BaseModel):
    """Base patient data"""
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(M|F|Other)$")
    severity: SeverityLevel
    mutation: MutationType
    dose_intensity: float = Field(..., ge=0.0)
    exposure_days: int = Field(..., ge=0)
    
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
    exposure_days: Optional[int] = None


class PatientResponse(PatientBase):
    """Patient response with additional fields"""
    id: int
    risk_score: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ==================== Prediction Models ====================

class PredictionInput(BaseModel):
    """ML prediction input"""
    age: int = Field(..., ge=0, le=150, description="Patient age")
    dose_intensity: float = Field(..., ge=0.0, description="Treatment dose")
    exposure_days: int = Field(..., ge=0, description="Days of exposure")
    severity: SeverityLevel = Field(..., description="Severity level")
    mutation: MutationType = Field(..., description="Mutation type")
    hemoglobin: Optional[float] = Field(None, ge=5.0, le=20.0)
    white_blood_cells: Optional[float] = Field(None, ge=0.0)
    platelets: Optional[float] = Field(None, ge=0.0)
    
    class Config:
        use_enum_values = True


class FeatureImportance(BaseModel):
    """Feature importance in prediction"""
    feature: str
    importance_score: float
    impact: str  # "High", "Medium", "Low"


class PredictionOutput(BaseModel):
    """ML prediction output"""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    risk_category: str = Field(..., description="Low/Medium/High")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_features: List[FeatureImportance]
    explanation: str
    timestamp: datetime


# ==================== Chat Models ====================

class ChatMessage(BaseModel):
    """Chat message"""
    patient_id: Optional[int] = None
    message: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ChatResponse(BaseModel):
    """Chat response"""
    message_id: str
    response: str
    confidence: Optional[float] = None
    timestamp: datetime


class ConversationHistory(BaseModel):
    """Conversation history"""
    patient_id: int
    messages: List[Dict[str, Any]]
    created_at: datetime


# ==================== Analytics Models ====================

class RiskDistribution(BaseModel):
    """Risk score distribution"""
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int


class AnalyticsResponse(BaseModel):
    """Analytics summary"""
    total_patients: int
    average_risk_score: float
    risk_distribution: RiskDistribution
    high_risk_patients: int
    inhibitor_rate: Optional[float] = None
    average_adherence: Optional[float] = None
    timestamp: datetime


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


# ==================== Health Models ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    database_connected: bool
    models_loaded: bool
