"""
Analytics and Dashboard API routes
Requires: Authenticated doctor or admin
Roles: doctor (view analytics), admin (view + export)
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from database import get_all_patients

from ..models import (
    AnalyticsRequest, AnalyticsResponse, 
    DashboardStats, RiskDistribution
)
from ..security import get_current_doctor, get_current_admin

router = APIRouter(prefix="/analytics", tags=["Analytics"])
logger = logging.getLogger(__name__)


@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard_analytics(
    days: Optional[int] = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_doctor)
) -> AnalyticsResponse:
    """
    Get dashboard analytics and statistics
    
    **Authentication:** Required
    **Roles:** doctor, admin
    
    Returns overview statistics, risk distribution, predictions trend, etc.
    """
    try:
        patients = get_all_patients()
        
        if not patients:
            # Return empty analytics
            return AnalyticsResponse(
                dashboard_stats=DashboardStats(
                    total_patients=0,
                    predictions_this_month=0,
                    average_risk_score=0.0,
                    high_risk_count=0,
                    treatment_adherence_avg=0.0,
                    inhibitor_rate=0.0
                ),
                risk_distribution=RiskDistribution(low=0, medium=0, high=0, critical=0),
                predictions_trend=[],
                top_factors=[],
                recommendations=[]
            )
        
        # Calculate statistics
        total_patients = len(patients)
        
        # Risk scores
        risk_scores = [p.get('Risk', 0) for p in patients if p.get('Risk') is not None]
        average_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Risk distribution
        risk_low = sum(1 for r in risk_scores if r < 0.3)
        risk_medium = sum(1 for r in risk_scores if 0.3 <= r < 0.6)
        risk_high = sum(1 for r in risk_scores if 0.6 <= r < 0.8)
        risk_critical = sum(1 for r in risk_scores if r >= 0.8)
        
        # Treatment adherence
        adherence_scores = [p.get('Adherence', 100) for p in patients if p.get('Adherence') is not None]
        avg_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 100
        
        # Inhibitor history
        inhibitor_count = sum(1 for p in patients if p.get('Previous Inhibitor'))
        inhibitor_rate = inhibitor_count / total_patients if total_patients > 0 else 0
        
        # Create statistics
        dashboard_stats = DashboardStats(
            total_patients=total_patients,
            predictions_this_month=total_patients,  # Simplified
            average_risk_score=average_risk,
            high_risk_count=risk_high + risk_critical,
            treatment_adherence_avg=avg_adherence,
            inhibitor_rate=inhibitor_rate
        )
        
        # Risk distribution
        risk_distribution = RiskDistribution(
            low=risk_low,
            medium=risk_medium,
            high=risk_high,
            critical=risk_critical
        )
        
        # Top contributing factors
        top_factors = [
            {"factor": "Previous Inhibitor History", "weight": 0.15, "count": inhibitor_count},
            {"factor": "Low Treatment Adherence", "weight": 0.10, "count": sum(1 for a in adherence_scores if a < 50)},
            {"factor": "Severe Hemophilia", "weight": 0.12, "count": sum(1 for p in patients if p.get('Severity') == 'Severe')},
        ]
        
        # Recommendations based on analytics
        recommendations = []
        if average_risk > 0.6:
            recommendations.append("High average risk score detected - review monitoring protocols")
        if avg_adherence < 70:
            recommendations.append("Treatment adherence below 70% - consider adherence interventions")
        if inhibitor_rate > 0.2:
            recommendations.append("Inhibitor rate above 20% - heighten surveillance")
        if risk_critical > 0:
            recommendations.append(f"{risk_critical} patients with critical risk - urgent specialist review needed")
        
        logger.info(f"[{current_user['username']}] Dashboard analytics retrieved: {total_patients} patients analyzed")
        
        return AnalyticsResponse(
            dashboard_stats=dashboard_stats,
            risk_distribution=risk_distribution,
            predictions_trend=[],
            top_factors=top_factors,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/risk-distribution")
async def get_risk_distribution(current_user: dict = Depends(get_current_doctor)) -> dict:
    """
    Get risk score distribution across patient population
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patients = get_all_patients()
        risk_scores = [p.get('Risk', 0) for p in patients if p.get('Risk') is not None]
        
        if not risk_scores:
            return {"message": "No risk data available"}
        
        distribution = {
            "low": sum(1 for r in risk_scores if r < 0.3),
            "medium": sum(1 for r in risk_scores if 0.3 <= r < 0.6),
            "high": sum(1 for r in risk_scores if 0.6 <= r < 0.8),
            "critical": sum(1 for r in risk_scores if r >= 0.8),
            "average": sum(risk_scores) / len(risk_scores),
            "min": min(risk_scores),
            "max": max(risk_scores)
        }
        
        logger.info(f"[{current_user['username']}] Risk distribution calculated: {len(risk_scores)} patients")
        return distribution
    
    except Exception as e:
        logger.error(f"Error calculating distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate distribution: {str(e)}")


@router.get("/severity-breakdown")
async def get_severity_breakdown(current_user: dict = Depends(get_current_doctor)) -> dict:
    """
    Get patient breakdown by severity
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patients = get_all_patients()
        
        breakdown = {
            "mild": sum(1 for p in patients if p.get('Severity') == 'Mild'),
            "moderate": sum(1 for p in patients if p.get('Severity') == 'Moderate'),
            "severe": sum(1 for p in patients if p.get('Severity') == 'Severe'),
            "total": len(patients)
        }
        
        logger.info(f"[{current_user['username']}] Severity breakdown calculated")
        return breakdown
    
    except Exception as e:
        logger.error(f"Error calculating severity breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate breakdown: {str(e)}")


@router.get("/adherence-metrics")
async def get_adherence_metrics(current_user: dict = Depends(get_current_doctor)) -> dict:
    """
    Get treatment adherence statistics
    
    **Authentication:** Required
    **Roles:** doctor, admin
    """
    try:
        patients = get_all_patients()
        adherence_scores = [p.get('Adherence', 100) for p in patients if p.get('Adherence') is not None]
        
        if not adherence_scores:
            return {"message": "No adherence data available"}
        
        logger.info(f"[{current_user['username']}] Adherence metrics retrieved")
        
        return {
            "average": sum(adherence_scores) / len(adherence_scores),
            "high": sum(1 for a in adherence_scores if a >= 90),
            "medium": sum(1 for a in adherence_scores if 70 <= a < 90),
            "low": sum(1 for a in adherence_scores if a < 70),
            "total_patients": len(adherence_scores)
        }
    
    except Exception as e:
        logger.error(f"Error calculating adherence metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate metrics: {str(e)}")


@router.get("/export")
async def export_analytics(
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: dict = Depends(get_current_admin)
):
    """
    Export analytics data in specified format
    
    **Authentication:** Required
    **Roles:** admin only
    """
    try:
        patients = get_all_patients()
        
        logger.info(f"[{current_user['username']}] Analytics export: {format} format, {len(patients)} patients")
        
        if format == "csv":
            return {
                "message": "CSV export functionality to be implemented",
                "patients_count": len(patients)
            }
        else:
            return {
                "patients": patients,
                "count": len(patients),
                "exported_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error exporting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
