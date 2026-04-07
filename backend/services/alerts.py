"""
High-Risk Patient Alert System
Identifies and manages critical patient conditions
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.models_orm import (
    Patient, HighRiskAlert, RiskPrediction, RiskCategory, 
    MonitoringRecord, TreatmentHistory
)


class AlertService:
    """Service for managing high-risk patient alerts"""
    
    # Alert Thresholds
    CRITICAL_RISK_SCORE = 0.8
    HIGH_RISK_SCORE = 0.6
    RISK_INCREASE_THRESHOLD = 0.15  # 15% increase
    PREVIOUS_INHIBITOR_FLAG = True
    
    @staticmethod
    def check_patient_risk(db: Session, patient_id: int) -> Dict[str, Any]:
        """
        Check patient risk and create/update alerts
        Returns: {alerts: List, risk_level: str, should_notify: bool}
        """
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            return {"alerts": [], "risk_level": "unknown", "should_notify": False}
        
        alerts = []
        
        # Check 1: Current risk score critical alert
        if patient.current_risk_score >= AlertService.CRITICAL_RISK_SCORE:
            alert = AlertService._create_or_update_alert(
                db, patient_id, patient.doctor_id,
                alert_type="critical_risk_score",
                severity="critical",
                message=f"CRITICAL: Patient risk score {patient.current_risk_score:.2f} exceeds critical threshold ({AlertService.CRITICAL_RISK_SCORE})",
                trigger_value=patient.current_risk_score,
                threshold=AlertService.CRITICAL_RISK_SCORE
            )
            alerts.append(alert)
        
        # Check 2: High risk score alert
        elif patient.current_risk_score >= AlertService.HIGH_RISK_SCORE:
            alert = AlertService._create_or_update_alert(
                db, patient_id, patient.doctor_id,
                alert_type="high_risk_score",
                severity="high",
                message=f"HIGH: Patient risk score {patient.current_risk_score:.2f} is elevated",
                trigger_value=patient.current_risk_score,
                threshold=AlertService.HIGH_RISK_SCORE
            )
            alerts.append(alert)
        
        # Check 3: Risk score increase
        if patient.last_prediction_score:
            risk_increase = patient.current_risk_score - patient.last_prediction_score
            if risk_increase >= AlertService.RISK_INCREASE_THRESHOLD:
                alert = AlertService._create_or_update_alert(
                    db, patient_id, patient.doctor_id,
                    alert_type="risk_score_increase",
                    severity="high",
                    message=f"ALERT: Risk increased from {patient.last_prediction_score:.2f} to {patient.current_risk_score:.2f}",
                    trigger_value=risk_increase,
                    threshold=AlertService.RISK_INCREASE_THRESHOLD
                )
                alerts.append(alert)
        
        # Check 4: Previous inhibitor history
        if patient.previous_inhibitor and patient.inhibitor_detection_date:
            days_since = (datetime.utcnow() - patient.inhibitor_detection_date).days
            if days_since < 365:  # Alert if within last year
                alert = AlertService._create_or_update_alert(
                    db, patient_id, patient.doctor_id,
                    alert_type="previous_inhibitor",
                    severity="high",
                    message=f"ALERT: Patient has inhibitor history ({days_since} days ago) - elevated re-development risk",
                    trigger_value=float(days_since),
                    threshold=365
                )
                alerts.append(alert)
        
        # Check 5: Low adherence
        if patient.treatment_adherence < 70:
            alert = AlertService._create_or_update_alert(
                db, patient_id, patient.doctor_id,
                alert_type="low_adherence",
                severity="moderate",
                message=f"WARNING: Treatment adherence is low ({patient.treatment_adherence:.0f}%)",
                trigger_value=patient.treatment_adherence,
                threshold=70
            )
            alerts.append(alert)
        
        # Check 6: Abnormal monitoring results
        recent_monitoring = db.query(MonitoringRecord)\
            .filter(MonitoringRecord.patient_id == patient_id)\
            .filter(MonitoringRecord.is_abnormal == True)\
            .filter(MonitoringRecord.record_date >= datetime.utcnow() - timedelta(days=7))\
            .all()
        
        if recent_monitoring:
            alert = AlertService._create_or_update_alert(
                db, patient_id, patient.doctor_id,
                alert_type="abnormal_monitoring",
                severity="moderate",
                message=f"ALERT: {len(recent_monitoring)} abnormal lab results in past 7 days",
                trigger_value=float(len(recent_monitoring)),
                threshold=0
            )
            alerts.append(alert)
        
        # Determine overall risk level
        has_critical = any(a.get("severity") == "critical" for a in alerts)
        has_high = any(a.get("severity") == "high" for a in alerts)
        
        risk_level = "critical" if has_critical else ("high" if has_high else "moderate" if alerts else "low")
        
        return {
            "alerts": alerts,
            "risk_level": risk_level,
            "should_notify": bool(alerts),
            "patient_id": patient_id
        }
    
    @staticmethod
    def _create_or_update_alert(
        db: Session,
        patient_id: int,
        doctor_id: int,
        alert_type: str,
        severity: str,
        message: str,
        trigger_value: float,
        threshold: float
    ) -> Dict[str, Any]:
        """Create or update alert"""
        
        # Check if active alert exists
        existing_alert = db.query(HighRiskAlert)\
            .filter(HighRiskAlert.patient_id == patient_id)\
            .filter(HighRiskAlert.alert_type == alert_type)\
            .filter(HighRiskAlert.alert_status == "active")\
            .first()
        
        if existing_alert:
            existing_alert.message = message
            existing_alert.trigger_value = trigger_value
            existing_alert.updated_at = datetime.utcnow()
            db.commit()
            return {
                "id": existing_alert.id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "status": "updated"
            }
        else:
            # Create new alert
            new_alert = HighRiskAlert(
                patient_id=patient_id,
                doctor_id=doctor_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                trigger_value=trigger_value,
                threshold=threshold,
                alert_status="active"
            )
            db.add(new_alert)
            db.commit()
            db.refresh(new_alert)
            
            return {
                "id": new_alert.id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "status": "created"
            }
    
    @staticmethod
    def get_doctor_alerts(db: Session, doctor_id: int, status: str = "active") -> List[Dict[str, Any]]:
        """Get all alerts for a doctor"""
        alerts = db.query(HighRiskAlert)\
            .filter(HighRiskAlert.doctor_id == doctor_id)\
            .filter(HighRiskAlert.alert_status == status)\
            .order_by(HighRiskAlert.created_at.desc())\
            .all()
        
        return [
            {
                "id": alert.id,
                "patient_id": alert.patient_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "status": alert.alert_status
            }
            for alert in alerts
        ]
    
    @staticmethod
    def get_patient_alerts(db: Session, patient_id: int) -> List[Dict[str, Any]]:
        """Get all active alerts for a patient"""
        alerts = db.query(HighRiskAlert)\
            .filter(HighRiskAlert.patient_id == patient_id)\
            .filter(HighRiskAlert.alert_status == "active")\
            .order_by(HighRiskAlert.severity.desc())\
            .all()
        
        return [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "created_at": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
    
    @staticmethod
    def acknowledge_alert(db: Session, alert_id: int, acknowledged_by: str = None) -> bool:
        """Acknowledge/resolve an alert"""
        alert = db.query(HighRiskAlert).filter(HighRiskAlert.id == alert_id).first()
        if alert:
            alert.alert_status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            db.commit()
            return True
        return False
    
    @staticmethod
    def resolve_alert(db: Session, alert_id: int, notes: str = None) -> bool:
        """Resolve an alert"""
        alert = db.query(HighRiskAlert).filter(HighRiskAlert.id == alert_id).first()
        if alert:
            alert.alert_status = "resolved"
            alert.resolution_notes = notes
            alert.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False
    
    @staticmethod
    def get_high_risk_patients(db: Session, doctor_id: int, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Get all high-risk patients for a doctor"""
        patients = db.query(Patient)\
            .filter(Patient.doctor_id == doctor_id)\
            .filter(Patient.current_risk_score >= threshold)\
            .order_by(Patient.current_risk_score.desc())\
            .all()
        
        return [
            {
                "id": patient.id,
                "name": f"{patient.first_name} {patient.last_name}",
                "risk_score": patient.current_risk_score,
                "risk_category": patient.risk_category.value,
                "severity": patient.severity.value,
                "last_prediction_date": patient.last_prediction_date.isoformat() if patient.last_prediction_date else None,
                "alert_count": len(AlertService.get_patient_alerts(db, patient.id))
            }
            for patient in patients
        ]


# ============= UTILITY FUNCTIONS =============

def check_all_patients(db: Session):
    """Check risk for all patients and update alerts"""
    patients = db.query(Patient).all()
    results = []
    for patient in patients:
        result = AlertService.check_patient_risk(db, patient.id)
        if result["alerts"]:
            results.append(result)
    return results
