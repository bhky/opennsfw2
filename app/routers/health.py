"""
Health check router.
"""
from typing import Dict, Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "OpenNSFW2 API",
        "version": "0.14.0"
    }


@router.get("/model")
async def model_health() -> Dict[str, Any]:
    """Check if the model is loaded and working."""
    try:
        from ..services.prediction_service import PredictionService
        service = PredictionService()

        return {
            "status": "healthy",
            "model_loaded": service.is_model_loaded(),
            "model_version": "0.14.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }
