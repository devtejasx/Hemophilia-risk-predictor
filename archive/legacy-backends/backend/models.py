"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, List, Any
from datetime import datetime


# ============= AUTHENTICATION MODELS =============

class UserLogin(BaseModel):
    """User login credentials"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="Password")


class UserRegister(BaseModel):
    """User registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")
    full_name: str = Field(..., min_length=2, description="Full name")
    role: str = Field(default="patient", description="Role: patient, doctor, admin")


class UserResponse(BaseModel):
    """User profile response (public data, no password)"""
    id: int
    username: str
    email: str
    full_name: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = None
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Expiration in seconds")
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str = Field(..., description="Refresh token from login")


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: str = Field(..., description="Email address")


# ============= PREDICTION MODELS =============

class PredictionRequest(BaseModel):
    """ML prediction request model"""
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    dose: float = Field(..., gt=0, description="Treatment dose in units")
    exposure: int = Field(..., ge=0, description="Days of treatment exposure")
    severity: str = Field(..., description="Severity: Mild, Moderate, Severe")
    mutation: str = Field(..., description="Gene mutation name")
    ethnicity: Optional[str] = None
    blood_type: Optional[str] = None
    hla_typing: Optional[str] = None
    product_type: Optional[str] = None
    treatment_adherence: Optional[float] = Field(None, ge=0, le=100)
    family_history: Optional[str] = None
    previous_inhibitor: Optional[bool] = None
    joint_damage_score: Optional[int] = None
    bleeding_episodes: Optional[int] = None
    baseline_factor_level: Optional[float] = None
    immunosuppression: Optional[bool] = None
    active_infection: Optional[bool] = None
    vaccination_status: Optional[str] = None
    physical_activity: Optional[str] = None
    stress_level: Optional[str] = None
    comorbidities: Optional[str] = None


class PredictionResponse(BaseModel):
    """ML prediction response model"""
    risk_score: float = Field(..., description="Predicted inhibitor risk (0-1)")
    risk_category: str = Field(..., description="Risk category: Low, Medium, High, Critical")
    confidence: float = Field(..., description="Prediction confidence")
    contributing_factors: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    model_used: str = Field(default="ensemble", description="Model type: rf, xgb, ensemble")


# ============= CHAT/CLINICAL ASSISTANT MODELS =============

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="user or assistant")
    content: str = Field(...)
    mode: Optional[str] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    question: str = Field(..., description="User question")
    mode: str = Field(
        default="diagnosis_support",
        description="Clinical mode: diagnosis_support, treatment_recommendation, risk_explanation, monitoring_analysis"
    )
    patient_data: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    user_id: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI-generated response")
    mode_used: str = Field(..., description="Mode used for response")
    disclaimer: str = Field(..., description="Safety disclaimer")
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None


# ============= PATIENT MODELS =============

class PatientData(BaseModel):
    """Patient clinical data"""
    name: str
    age: int
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    severity: Optional[str] = None
    mutation: Optional[str] = None
    blood_type: Optional[str] = None
    hla_type: Optional[str] = None
    dose: Optional[float] = None
    exposure: Optional[int] = None
    product: Optional[str] = None
    adherence: Optional[float] = None
    family_history: Optional[str] = None
    previous_inhibitor: Optional[bool] = None
    joint_damage: Optional[int] = None
    bleeding_episodes: Optional[int] = None
    factor_level: Optional[float] = None
    immunosuppression: Optional[bool] = None
    active_infection: Optional[bool] = None
    vaccination: Optional[str] = None
    activity_level: Optional[str] = None
    stress_level: Optional[str] = None
    risk: Optional[float] = None


class PatientCreateRequest(BaseModel):
    """Create patient request"""
    patient_data: PatientData
    notes: Optional[str] = None


class PatientUpdateRequest(BaseModel):
    """Update patient request"""
    patient_data: PatientData
    updated_by: Optional[int] = None


class PatientResponse(BaseModel):
    """Patient response model"""
    id: int
    patient_data: PatientData
    created_at: datetime
    updated_at: datetime
    last_prediction: Optional[datetime] = None
    last_risk_score: Optional[float] = None


# ============= ANALYTICS MODELS =============

class AnalyticsRequest(BaseModel):
    """Analytics request model"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metric_type: Optional[str] = None  # "prediction", "risk", "treatment", etc
    group_by: Optional[str] = None  # "severity", "mutation", "age_group", etc


class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_patients: int
    predictions_this_month: int
    average_risk_score: float
    high_risk_count: int
    treatment_adherence_avg: float
    inhibitor_rate: float


class RiskDistribution(BaseModel):
    """Risk score distribution"""
    low: int  # 0-0.3
    medium: int  # 0.3-0.6
    high: int  # 0.6-0.8
    critical: int  # 0.8-1.0


class AnalyticsResponse(BaseModel):
    """Analytics response model"""
    dashboard_stats: DashboardStats
    risk_distribution: RiskDistribution
    predictions_trend: List[Dict[str, Any]]
    top_factors: List[Dict[str, float]]
    recommendations: List[str]


# ============= EVALUATION MODELS =============

class EvaluationRequest(BaseModel):
    """ML model evaluation request"""
    model_names: Optional[List[str]] = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    metrics: Optional[List[str]] = None  # accuracy, precision, recall, f1, roc_auc, etc


class MetricsResponse(BaseModel):
    """Model metrics response"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Dict[str, float]]


class EvaluationResponse(BaseModel):
    """Complete evaluation response"""
    models_evaluated: List[str]
    metrics: List[MetricsResponse]
    best_model: str
    best_metric: float
    training_details: Dict[str, Any]


# ============= ERROR MODELS =============

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ============= HEALTH CHECK =============

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    database: str = "connected"
    ml_models: str = "loaded"
