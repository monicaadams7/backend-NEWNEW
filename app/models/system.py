"""
System-related data models
"""
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")

class MetadataResponse(BaseModel):
    """Response model for root endpoint metadata"""
    app: str = Field(..., description="Application name")
    status: str = Field(..., description="Application status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    timestamp: str = Field(..., description="Current server time")
