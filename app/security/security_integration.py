"""
Security Integration Module for FitGPT Backend
Entry point for importing security and infrastructure components
"""
import logging
from fastapi import FastAPI

# Import components
from .api_key_auth import (
    get_api_key, require_role, create_api_key, revoke_api_key,
    get_current_user_id, rate_limit_by_key
)

from .jwt_auth import (
    get_current_user, require_role as jwt_require_role,
    create_access_token, authenticate_user, UserCreate, UserInfo
)

from .security_middleware import (
    setup_security_middleware, SecurityHeadersMiddleware,
    XSSProtectionMiddleware, RequestValidationMiddleware
)

from .request_sanitizer import (
    SanitizedModel, sanitize_input, sanitize_request_body,
    validate_and_sanitize, get_pagination, get_sorting
)

from .user_manager import (
    setup_auth_routes, auth_router, user_router, api_key_router
)

from .health_check import (
    setup_health_routes, health_router
)

# Configure logger
logger = logging.getLogger(__name__)

def setup_security(app: FastAPI, enable_middleware: bool = True):
    """
    Set up all security components for a FastAPI app
    
    Args:
        app: The FastAPI app instance
        enable_middleware: Whether to enable security middleware
        
    Returns:
        The configured FastAPI app
        
    Usage:
        from fastapi import FastAPI
        from app.security import setup_security
        
        app = FastAPI()
        setup_security(app)
    """
    # Set up security middleware
    if enable_middleware:
        setup_security_middleware(app)
        
    # Set up authentication routes
    setup_auth_routes(app)
    
    # Set up health check routes
    setup_health_routes(app)
    
    logger.info("Security components initialized")
    
    return app

__all__ = [
    # API Key Authentication
    "get_api_key", "require_role", "create_api_key", 
    "revoke_api_key", "get_current_user_id", "rate_limit_by_key",
    
    # JWT Authentication
    "get_current_user", "jwt_require_role", "create_access_token",
    "authenticate_user", "UserCreate", "UserInfo",
    
    # Security Middleware
    "setup_security_middleware", "SecurityHeadersMiddleware",
    "XSSProtectionMiddleware", "RequestValidationMiddleware",
    
    # Request Sanitization
    "SanitizedModel", "sanitize_input", "sanitize_request_body",
    "validate_and_sanitize", "get_pagination", "get_sorting",
    
    # Routes
    "auth_router", "user_router", "api_key_router", "health_router",
    
    # Setup Functions
    "setup_auth_routes", "setup_health_routes", "setup_security"
]