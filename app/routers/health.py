"""
Health check router.
"""
from typing import Dict, Any

from fastapi import APIRouter

import opennsfw2

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "OpenNSFW2 API",
        "version": opennsfw2.__version__
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
            "version": opennsfw2.__version__
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }
