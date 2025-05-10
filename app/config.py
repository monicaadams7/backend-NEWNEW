"""
Configuration handling for FitGPT backend
Loads and validates environment variables
"""
import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application settings
    ENVIRONMENT: str = "development"
    PORT: int = 8000
    
    # Security settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]
    RATE_LIMIT: str = "20/minute"
    
    # Storage settings
    STORAGE_TYPE: str = "sqlite"
    SQLITE_PATH: str = "./data/conversations.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    
    # Logging settings
    LOG_DIR: str = "./logs"
    LOG_LEVEL: str = "INFO"
    
    # API keys
    OPENAI_API_KEY: Optional[str] = None
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse comma-separated origins into a list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment value"""
        allowed = ["development", "staging", "production", "test"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {', '.join(allowed)}")
        return v.lower()
    
    @validator("STORAGE_TYPE")
    def validate_storage_type(cls, v):
        """Validate storage type"""
        allowed = ["sqlite", "redis"]
        if v.lower() not in allowed:
            raise ValueError(f"Storage type must be one of: {', '.join(allowed)}")
        return v.lower()
    
    class Config:
        """Pydantic config class"""
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
