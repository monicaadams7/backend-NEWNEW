"""
System API endpoints
"""
from fastapi import APIRouter, Request
from datetime import datetime
import os
import logging

from app.models.system import HealthResponse, MetadataResponse

# Create logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["System"])

# Define endpoints
@router.get("/", response_model=MetadataResponse)
async def root():
    """Return basic application metadata"""
    return {
        "app": "FitGPT",
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is healthy"""
    return {"status": "ok", "version": "1.0.0"}
