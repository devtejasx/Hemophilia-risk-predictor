"""
Risk Trend Analysis Service
Analyzes patient risk trends over time
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.models_orm import Patient, RiskPrediction


class TrendService:
    """Service for analyzing risk trends"""
    
    @staticmethod
    def get_patient_risk_trend(
        db: Session,
        patient_id: int,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Get risk trend data for a patient over specified days
        Returns timeline of risk scores with trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        predictions = db.query(RiskPrediction)\
            .filter(RiskPrediction.patient_id == patient_id)\
            .filter(RiskPrediction.prediction_date >= cutoff_date)\
            .order_by(RiskPrediction.prediction_date.asc())\
            .all()
        
        if not predictions:
            return {
                "patient_id": patient_id,
                "trend_data": [],
                "trend_direction": "no_data",
                "average_risk": 0,
                "max_risk": 0,
                "min_risk": 0,
                "days_analyzed": days
            }
        
        # Extract data
        timeline = [
            {
                "date": pred.prediction_date.isoformat(),
                "risk_score": pred.risk_score,
                "risk_category": pred.risk_category.value,
                "model": pred.model_used,
                "confidence": pred.confidence
            }
            for pred in predictions
        ]
        
        risk_scores = [p.risk_score for p in predictions]
        
        # Calculate statistics
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)
        
        # Determine trend direction
        if len(risk_scores) >= 2:
            recent_avg = sum(risk_scores[-min(5, len(risk_scores)):]) / min(5, len(risk_scores))
            old_avg = sum(risk_scores[:min(5, len(risk_scores))]) / min(5, len(risk_scores))
            
            if recent_avg > old_avg * 1.1:  # 10% increase
                trend_direction = "increasing"
            elif recent_avg < old_avg * 0.9:  # 10% decrease
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            "patient_id": patient_id,
            "trend_data": timeline,
            "trend_direction": trend_direction,
            "average_risk": round(avg_risk, 3),
            "max_risk": round(max_risk, 3),
            "min_risk": round(min_risk, 3),
            "total_predictions": len(predictions),
            "days_analyzed": days
        }
    
    @staticmethod
    def get_cohort_risk_trend(
        db: Session,
        doctor_id: int,
        days: int = 90,
        severity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated risk trend for all patients under a doctor
        Shows cohort-level risk trends
        """
        from backend.models_orm import SeverityLevel
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all patient IDs
        query = db.query(Patient.id).filter(Patient.doctor_id == doctor_id)
        if severity:
            query = query.filter(Patient.severity == severity)
        
        patient_ids = [p[0] for p in query.all()]
        
        # Get predictions for all patients
        predictions = db.query(RiskPrediction)\
            .filter(RiskPrediction.patient_id.in_(patient_ids))\
            .filter(RiskPrediction.prediction_date >= cutoff_date)\
            .order_by(RiskPrediction.prediction_date.asc())\
            .all()
        
        if not predictions:
            return {
                "doctor_id": doctor_id,
                "cohort_size": len(patient_ids),
                "trend_data": [],
                "average_cohort_risk": 0,
                "days_analyzed": days
            }
        
        # Group by date and calculate averages
        date_groups = {}
        for pred in predictions:
            date_key = pred.prediction_date.date().isoformat()
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(pred.risk_score)
        
        timeline = [
            {
                "date": date,
                "average_risk": round(sum(scores) / len(scores), 3),
                "min_risk": round(min(scores), 3),
                "max_risk": round(max(scores), 3),
                "patients_assessed": len(scores)
            }
            for date, scores in sorted(date_groups.items())
        ]
        
        all_scores = [p.risk_score for p in predictions]
        cohort_avg_risk = sum(all_scores) / len(all_scores)
        
        return {
            "doctor_id": doctor_id,
            "cohort_size": len(patient_ids),
            "trend_data": timeline,
            "average_cohort_risk": round(cohort_avg_risk, 3),
            "high_risk_count": sum(1 for s in all_scores if s >= 0.6),
            "critical_risk_count": sum(1 for s in all_scores if s >= 0.8),
            "days_analyzed": days
        }
    
    @staticmethod
    def get_risk_change_summary(
        db: Session,
        patient_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get summary of risk changes over period
        Shows if patient is improving, worsening, or stable
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        predictions = db.query(RiskPrediction)\
            .filter(RiskPrediction.patient_id == patient_id)\
            .filter(RiskPrediction.prediction_date >= cutoff_date)\
            .order_by(RiskPrediction.prediction_date.asc())\
            .all()
        
        if len(predictions) < 2:
            return {
                "patient_id": patient_id,
                "period_days": days,
                "status": "insufficient_data",
                "predictions_count": len(predictions)
            }
        
        first_score = predictions[0].risk_score
        last_score = predictions[-1].risk_score
        change = last_score - first_score
        percent_change = (change / first_score * 100) if first_score > 0 else 0
        
        # Classify change
        if abs(percent_change) < 5:
            status = "stable"
            color = "yellow"
        elif percent_change > 0:
            status = "worsening"
            color = "red"
        else:
            status = "improving"
            color = "green"
        
        return {
            "patient_id": patient_id,
            "period_days": days,
            "start_date": predictions[0].prediction_date.isoformat(),
            "end_date": predictions[-1].prediction_date.isoformat(),
            "initial_risk": round(first_score, 3),
            "current_risk": round(last_score, 3),
            "absolute_change": round(change, 3),
            "percent_change": round(percent_change, 1),
            "status": status,
            "color": color,
            "predictions_count": len(predictions)
        }
    
    @staticmethod
    def predict_risk_trajectory(
        db: Session,
        patient_id: int,
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Simple prediction of future risk based on trend
        Uses linear regression on recent trend
        """
        recent_days = 90
        cutoff_date = datetime.utcnow() - timedelta(days=recent_days)
        
        predictions = db.query(RiskPrediction)\
            .filter(RiskPrediction.patient_id == patient_id)\
            .filter(RiskPrediction.prediction_date >= cutoff_date)\
            .order_by(RiskPrediction.prediction_date.asc())\
            .all()
        
        if len(predictions) < 2:
            return {
                "patient_id": patient_id,
                "status": "insufficient_data",
                "projected_date": None,
                "projected_risk": None
            }
        
        # Simple linear trend
        scores = [p.risk_score for p in predictions]
        x_values = list(range(len(scores)))
        
        # Calculate linear regression
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(scores) / n
        
        numerator = sum((x_values[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Project forward
        projected_x = len(scores) + (days_ahead / recent_days * len(scores))
        projected_risk = intercept + slope * projected_x
        projected_risk = max(0, min(1, projected_risk))  # Clamp to 0-1
        
        projected_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        return {
            "patient_id": patient_id,
            "analysis_period_days": recent_days,
            "projection_days": days_ahead,
            "current_risk": round(scores[-1], 3),
            "projected_risk": round(projected_risk, 3),
            "trend_slope": round(slope, 4),
            "projected_date": projected_date.isoformat(),
            "projection_confidence": "low" if len(predictions) < 5 else "moderate"
        }
    
    @staticmethod
    def get_risk_improvement_metrics(
        db: Session,
        doctor_id: int
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics on patient risk improvements
        Shows which patients are improving vs worsening
        """
        patients = db.query(Patient)\
            .filter(Patient.doctor_id == doctor_id)\
            .all()
        
        improvements = []
        deteriorations = []
        stable = []
        
        for patient in patients:
            summary = TrendService.get_risk_change_summary(db, patient.id, days=30)
            if summary.get("status") == "insufficient_data":
                continue
            
            if summary["status"] == "improving":
                improvements.append({
                    "patient_id": patient.id,
                    "name": f"{patient.first_name} {patient.last_name}",
                    "improvement": abs(summary["percent_change"]),
                    "current_risk": summary["current_risk"]
                })
            elif summary["status"] == "worsening":
                deteriorations.append({
                    "patient_id": patient.id,
                    "name": f"{patient.first_name} {patient.last_name}",
                    "deterioration": summary["percent_change"],
                    "current_risk": summary["current_risk"]
                })
            else:
                stable.append({
                    "patient_id": patient.id,
                    "name": f"{patient.first_name} {patient.last_name}",
                    "current_risk": summary["current_risk"]
                })
        
        return {
            "doctor_id": doctor_id,
            "total_patients": len(patients),
            "improving_count": len(improvements),
            "worsening_count": len(deteriorations),
            "stable_count": len(stable),
            "improving_patients": sorted(improvements, key=lambda x: x["improvement"], reverse=True),
            "worsening_patients": sorted(deteriorations, key=lambda x: x["deterioration"], reverse=True),
            "stable_patients": stable
        }
