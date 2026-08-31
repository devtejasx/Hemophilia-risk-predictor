"""
FastAPI routers for different API endpoints
"""

from . import predict, chat, patients, analytics

__all__ = ["predict", "chat", "patients", "analytics"]
