"""
Analytics Router
Endpoints for system analytics and statistics
"""

from fastapi import APIRouter, Depends, HTTPException, status
from models import AnalyticsResponse
from services.analytics_service import get_analytics_service, AnalyticsService
from database import get_db
from exceptions import DatabaseException, exception_to_http_exception

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard_stats(
    db = Depends(get_db)
):
    """
    Get dashboard summary statistics
    
    Args:
        db: Database connection
        
    Returns:
        Dashboard statistics including risk distribution
    """
    try:
        service = get_analytics_service(db)
        return service.get_dashboard_stats()
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/trends")
async def get_trends(
    days: int = 30,
    db = Depends(get_db)
):
    """
    Get risk score trends over time
    
    Args:
        days: Number of days to analyze (default: 30)
        db: Database connection
        
    Returns:
        Trends data with daily aggregations
    """
    try:
        service = get_analytics_service(db)
        return service.get_risk_trends(days)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/mutations")
async def get_mutation_stats(
    db = Depends(get_db)
):
    """
    Get statistics by mutation type
    
    Args:
        db: Database connection
        
    Returns:
        Statistics for each mutation type
    """
    try:
        service = get_analytics_service(db)
        return service.get_mutation_statistics()
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/severity")
async def get_severity_distribution(
    db = Depends(get_db)
):
    """
    Get patient distribution by severity
    
    Args:
        db: Database connection
        
    Returns:
        Count and average risk by severity level
    """
    try:
        service = get_analytics_service(db)
        return service.get_severity_distribution()
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/high-risk")
async def get_high_risk_patients(
    limit: int = 10,
    db = Depends(get_db)
):
    """
    Get highest risk patients
    
    Args:
        limit: Maximum number of patients to return (default: 10)
        db: Database connection
        
    Returns:
        List of highest risk patients
    """
    try:
        service = get_analytics_service(db)
        return service.get_high_risk_patients(limit)
    except DatabaseException as e:
        raise exception_to_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
