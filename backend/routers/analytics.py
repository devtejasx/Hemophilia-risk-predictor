"""Analytics computed from stored predictions. Nothing is synthesised."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend import db
from backend.routers.auth import get_current_user
from backend.schemas import AnalyticsResponse
from backend.services import ml

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("", response_model=AnalyticsResponse)
def analytics(current_user: dict = Depends(get_current_user)) -> AnalyticsResponse:
    data = db.analytics_for_user(current_user["id"])
    return AnalyticsResponse(**data, model_version=ml.model_version() or "unavailable")
