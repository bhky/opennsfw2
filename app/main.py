"""
FastAPI application for OpenNSFW2 HTTP service.
"""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import opennsfw2
from .routers import health, prediction


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Initialize the model on startup."""
    from .services.prediction_service import PredictionService
    # Initialize the singleton to load the model.
    PredictionService()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="OpenNSFW2 API",
    description="HTTP API for NSFW content detection using OpenNSFW2",
    version=opennsfw2.__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers.
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(prediction.router, prefix="/predict", tags=["prediction"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
