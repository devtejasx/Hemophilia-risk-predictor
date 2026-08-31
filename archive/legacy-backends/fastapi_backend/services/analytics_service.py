"""
Analytics Service
System-wide analytics and statistics
"""

from typing import Dict, Any, Optional
from datetime import datetime
from models import AnalyticsResponse, RiskDistribution
from exceptions import DatabaseException


class AnalyticsService:
    """Service for analytics and statistics"""
    
    def __init__(self, db):
        self.db = db
    
    def get_dashboard_stats(self) -> AnalyticsResponse:
        """Get dashboard statistics"""
        try:
            # Total patients
            total = self.db.execute_one('SELECT COUNT(*) as count FROM patients')
            total_patients = total['count'] if total else 0
            
            # Average risk score
            avg_risk = self.db.execute_one(
                'SELECT AVG(risk_score) as avg_risk FROM patients WHERE risk_score IS NOT NULL'
            )
            average_risk_score = float(avg_risk['avg_risk'] or 0.0)
            
            # Risk distribution
            low_risk = self.db.execute_one(
                'SELECT COUNT(*) as count FROM patients WHERE risk_score < 0.33'
            )
            medium_risk = self.db.execute_one(
                'SELECT COUNT(*) as count FROM patients WHERE risk_score >= 0.33 AND risk_score < 0.67'
            )
            high_risk = self.db.execute_one(
                'SELECT COUNT(*) as count FROM patients WHERE risk_score >= 0.67'
            )
            
            risk_distribution = RiskDistribution(
                low_risk_count=low_risk['count'] if low_risk else 0,
                medium_risk_count=medium_risk['count'] if medium_risk else 0,
                high_risk_count=high_risk['count'] if high_risk else 0
            )
            
            # High risk patients
            high_risk_patients = (high_risk['count'] if high_risk else 0)
            
            return AnalyticsResponse(
                total_patients=total_patients,
                average_risk_score=average_risk_score,
                risk_distribution=risk_distribution,
                high_risk_patients=high_risk_patients,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            raise DatabaseException(f"Failed to get analytics: {str(e)}")
    
    def get_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get risk score trends over time"""
        try:
            query = '''
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as count,
                    AVG(risk_score) as avg_risk
                FROM patients
                WHERE created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            '''
            rows = self.db.execute(query, (days,))
            
            trends = [
                {
                    "date": row['date'],
                    "count": row['count'],
                    "average_risk": float(row['avg_risk'] or 0.0)
                }
                for row in rows
            ]
            
            return {
                "period_days": days,
                "trends": trends,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            raise DatabaseException(f"Failed to get trends: {str(e)}")
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """Get statistics by mutation type"""
        try:
            query = '''
                SELECT 
                    mutation,
                    COUNT(*) as patient_count,
                    AVG(risk_score) as avg_risk,
                    MAX(risk_score) as max_risk,
                    MIN(risk_score) as min_risk
                FROM patients
                GROUP BY mutation
                ORDER BY patient_count DESC
            '''
            rows = self.db.execute(query)
            
            stats = {}
            for row in rows:
                stats[row['mutation']] = {
                    "patient_count": row['patient_count'],
                    "average_risk": float(row['avg_risk'] or 0.0),
                    "max_risk": float(row['max_risk'] or 0.0),
                    "min_risk": float(row['min_risk'] or 0.0)
                }
            
            return {
                "mutation_statistics": stats,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            raise DatabaseException(f"Failed to get mutation stats: {str(e)}")
    
    def get_severity_distribution(self) -> Dict[str, Any]:
        """Get patient distribution by severity"""
        try:
            query = '''
                SELECT 
                    severity,
                    COUNT(*) as count,
                    AVG(risk_score) as avg_risk
                FROM patients
                GROUP BY severity
            '''
            rows = self.db.execute(query)
            
            distribution = {}
            for row in rows:
                distribution[row['severity']] = {
                    "count": row['count'],
                    "average_risk": float(row['avg_risk'] or 0.0)
                }
            
            return {
                "severity_distribution": distribution,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            raise DatabaseException(f"Failed to get severity distribution: {str(e)}")
    
    def get_high_risk_patients(self, limit: int = 10) -> Dict[str, Any]:
        """Get highest risk patients"""
        try:
            query = '''
                SELECT id, name, age, severity, risk_score
                FROM patients
                WHERE risk_score IS NOT NULL
                ORDER BY risk_score DESC
                LIMIT ?
            '''
            rows = self.db.execute(query, (limit,))
            
            patients = [
                {
                    "id": row['id'],
                    "name": row['name'],
                    "age": row['age'],
                    "severity": row['severity'],
                    "risk_score": float(row['risk_score'])
                }
                for row in rows
            ]
            
            return {
                "high_risk_patients": patients,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            raise DatabaseException(f"Failed to get high risk patients: {str(e)}")


def get_analytics_service(db) -> AnalyticsService:
    """Dependency injection for analytics service"""
    return AnalyticsService(db)
