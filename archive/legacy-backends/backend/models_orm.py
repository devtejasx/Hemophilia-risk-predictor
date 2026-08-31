"""
SQLAlchemy ORM Models for Hemophilia AI Platform
PostgreSQL-based database with comprehensive relationships and indexing
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, ForeignKey, Index, Enum, JSON, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum

Base = declarative_base()


# ============= ENUMS =============

class SeverityLevel(enum.Enum):
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


class RiskCategory(enum.Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class UserRole(enum.Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class ConversationType(enum.Enum):
    DIAGNOSIS_SUPPORT = "diagnosis_support"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    RISK_EXPLANATION = "risk_explanation"
    MONITORING_ANALYSIS = "monitoring_analysis"


# ============= MODELS =============

class Doctor(Base):
    """Doctor/Healthcare Professional Model"""
    __tablename__ = "doctors"
    __table_args__ = (
        Index('idx_doctor_user_id', 'user_id'),
        Index('idx_doctor_hospital', 'hospital_id'),
        Index('idx_doctor_active', 'is_active'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True, nullable=False)
    specialization = Column(String(100))
    hospital_id = Column(String(100))
    license_number = Column(String(100), unique=True)
    phone = Column(String(20))
    bio = Column(Text)
    years_experience = Column(Integer)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patients = relationship("Patient", back_populates="assigned_doctor")
    conversations = relationship("Conversation", back_populates="doctor")
    summaries = relationship("PatientSummary", back_populates="doctor")


class Patient(Base):
    """Patient Model"""
    __tablename__ = "patients"
    __table_args__ = (
        Index('idx_patient_doctor_id', 'doctor_id'),
        Index('idx_patient_risk_score', 'current_risk_score'),
        Index('idx_patient_severity', 'severity'),
        Index('idx_patient_created_at', 'created_at'),
        Index('idx_patient_last_prediction', 'last_prediction_date'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    email = Column(String(100), unique=True, index=True)
    phone = Column(String(20))
    
    # Clinical Data
    severity = Column(Enum(SeverityLevel), default=SeverityLevel.MODERATE)
    mutation = Column(String(200))
    blood_type = Column(String(10))
    hla_type = Column(String(50))
    
    # Treatment History
    dose = Column(Float)
    exposure_days = Column(Integer)
    product_type = Column(String(100))
    treatment_adherence = Column(Float, default=100.0)  # Percentage
    
    # Risk Management
    current_risk_score = Column(Float, default=0.0, index=True)
    risk_category = Column(Enum(RiskCategory), default=RiskCategory.LOW)
    previous_inhibitor = Column(Boolean, default=False)
    inhibitor_detection_date = Column(DateTime)
    
    # Clinical Factors
    joint_damage_score = Column(Integer, default=0)
    bleeding_episodes_count = Column(Integer, default=0)
    baseline_factor_level = Column(Float)
    active_infection = Column(Boolean, default=False)
    immunosuppression = Column(Boolean, default=False)
    vaccination_status = Column(String(50))
    
    # Lifestyle
    physical_activity = Column(String(50))
    stress_level = Column(String(50))
    comorbidities = Column(Text)
    
    # Tracking
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    last_prediction_date = Column(DateTime)
    last_prediction_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Extra metadata
    notes = Column(Text)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    assigned_doctor = relationship("Doctor", back_populates="patients")
    treatment_history = relationship("TreatmentHistory", back_populates="patient", cascade="all, delete-orphan")
    monitoring_records = relationship("MonitoringRecord", back_populates="patient", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="patient", cascade="all, delete-orphan")
    risk_predictions = relationship("RiskPrediction", back_populates="patient", cascade="all, delete-orphan")
    summaries = relationship("PatientSummary", back_populates="patient", cascade="all, delete-orphan")


class TreatmentHistory(Base):
    """Patient Treatment History Model"""
    __tablename__ = "treatment_history"
    __table_args__ = (
        Index('idx_treatment_patient_id', 'patient_id'),
        Index('idx_treatment_date', 'treatment_date'),
        Index('idx_treatment_type', 'treatment_type'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    treatment_type = Column(String(100), nullable=False)
    dose = Column(Float)
    route = Column(String(50))  # IV, SC, etc.
    product_name = Column(String(200))
    treatment_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    indication = Column(String(200))  # Reason for treatment
    response = Column(String(50))  # Good, Fair, Poor
    adverse_events = Column(Text)
    
    recorded_by = Column(String(100))
    recorded_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="treatment_history")


class MonitoringRecord(Base):
    """Patient Monitoring/Lab Records Model"""
    __tablename__ = "monitoring_records"
    __table_args__ = (
        Index('idx_monitoring_patient_id', 'patient_id'),
        Index('idx_monitoring_date', 'record_date'),
        Index('idx_monitoring_test_type', 'test_type'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    test_type = Column(String(100), nullable=False)  # Inhibitor test, Factor level, etc.
    test_date = Column(DateTime, default=datetime.utcnow, index=True)
    record_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    result_value = Column(Float)
    result_unit = Column(String(50))
    reference_range = Column(String(100))
    is_abnormal = Column(Boolean, default=False)
    
    clinical_notes = Column(Text)
    lab_name = Column(String(200))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="monitoring_records")


class RiskPrediction(Base):
    """ML Risk Predictions History Model"""
    __tablename__ = "risk_predictions"
    __table_args__ = (
        Index('idx_prediction_patient_id', 'patient_id'),
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_prediction_score', 'risk_score'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    risk_score = Column(Float, nullable=False, index=True)
    risk_category = Column(Enum(RiskCategory), nullable=False)
    confidence = Column(Float)
    
    model_used = Column(String(100))  # rf, xgb, ensemble
    model_version = Column(String(50))
    
    input_features = Column(JSON)  # Store input parameters
    contributing_factors = Column(JSON)  # Store top factors
    recommendations = Column(JSON)  # Store recommendations
    
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="risk_predictions")


class Conversation(Base):
    """AI Chat Conversations Model"""
    __tablename__ = "conversations"
    __table_args__ = (
        Index('idx_conversation_patient_id', 'patient_id'),
        Index('idx_conversation_doctor_id', 'doctor_id'),
        Index('idx_conversation_date', 'created_at'),
        Index('idx_conversation_type', 'conversation_type'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    conversation_type = Column(Enum(ConversationType), nullable=False)
    
    question = Column(Text, nullable=False)
    answer = Column(Text)
    
    ai_model = Column(String(50))  # GPT-4, etc.
    confidence = Column(Float)
    
    user_rating = Column(Integer)  # 1-5 stars
    feedback = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Extra metadata
    session_id = Column(String(100), index=True)  # Group conversations
    is_archived = Column(Boolean, default=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="conversations")
    doctor = relationship("Doctor", back_populates="conversations")


class PatientSummary(Base):
    """AI-Generated Patient Clinical Summaries Model"""
    __tablename__ = "patient_summaries"
    __table_args__ = (
        Index('idx_summary_patient_id', 'patient_id'),
        Index('idx_summary_doctor_id', 'doctor_id'),
        Index('idx_summary_date', 'generated_at'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    # Summary Components
    clinical_summary = Column(Text, nullable=False)  # AI-generated overview
    risk_assessment = Column(Text)  # Risk analysis
    treatment_recommendations = Column(Text)  # AI recommendations
    monitoring_suggestions = Column(Text)  # What to monitor
    
    # Metadata
    summary_type = Column(String(50))  # routine, urgent, follow_up
    ai_model = Column(String(50))
    confidence_score = Column(Float)
    
    # Tracking
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime)
    approved_by_doctor = Column(Boolean, default=False)
    doctor_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="summaries")
    doctor = relationship("Doctor", back_populates="summaries")


class HighRiskAlert(Base):
    """High-Risk Patient Alerts Model"""
    __tablename__ = "high_risk_alerts"
    __table_args__ = (
        Index('idx_alert_patient_id', 'patient_id'),
        Index('idx_alert_doctor_id', 'doctor_id'),
        Index('idx_alert_created_at', 'created_at'),
        Index('idx_alert_status', 'alert_status'),
        Index('idx_alert_severity', 'severity'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    alert_type = Column(String(100), nullable=False)  # high_risk_score, risk_increase, etc.
    severity = Column(String(50))  # critical, high, moderate
    
    message = Column(Text, nullable=False)
    trigger_value = Column(Float)  # Value that triggered alert
    threshold = Column(Float)  # What was exceeded
    
    alert_status = Column(String(50), default="active", index=True)  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient")
    doctor = relationship("Doctor")


# ============= DATABASE SESSION =============

class DatabaseSession:
    """Database session management"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "postgresql://user:password@localhost/hemophilia"
        self.engine = None
        self.SessionLocal = None
    
    def init_engine(self):
        """Initialize database engine and create tables"""
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,  # Test connections before using
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        if self.SessionLocal is None:
            self.init_engine()
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        if session:
            session.close()


# ============= UTILITY FUNCTIONS =============

def init_database(database_url: str = None):
    """Initialize database with tables and indexes"""
    db_session = DatabaseSession(database_url)
    db_session.init_engine()
    return db_session
