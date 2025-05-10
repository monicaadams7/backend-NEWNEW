"""
Security utilities for the FitGPT backend
"""
import secrets
import hashlib
import logging
from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def generate_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """
    Create a secure hash of a password
    
    Note: In a production system, use a proper password hashing library
    like bcrypt, argon2, or passlib
    """
    # This is a placeholder - use a proper password hashing library in production
    return hashlib.sha256(password.encode()).hexdigest()

def validate_api_key(request: Request, api_key_header: str = "X-API-Key") -> bool:
    """
    Validate an API key from request headers
    """
    # Get API key from header
    api_key = request.headers.get(api_key_header)
    
    if not api_key:
        logger.warning("API key missing")
        return False
    
    # In a real application, you would validate against a sec

    # Continue creating security utilities
cat > app/utils/security.py << 'EOF'
"""
Security utilities for the FitGPT backend
"""
import secrets
import hashlib
import logging
from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def generate_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """
    Create a secure hash of a password
    
    Note: In a production system, use a proper password hashing library
    like bcrypt, argon2, or passlib
    """
    # This is a placeholder - use a proper password hashing library in production
    return hashlib.sha256(password.encode()).hexdigest()

def validate_api_key(request: Request, api_key_header: str = "X-API-Key") -> bool:
    """
    Validate an API key from request headers
    """
    # Get API key from header
    api_key = request.headers.get(api_key_header)
    
    if not api_key:
        logger.warning("API key missing")
        return False
    
    # In a real application, you would validate against a secure storage
    # This is just a placeholder
    valid_api_key = "your-api-key-here"
    
    return secrets.compare_digest(api_key, valid_api_key)

def api_key_auth(request: Request, api_key_header: str = "X-API-Key") -> None:
    """
    Dependency for endpoints that require API key authentication
    Raises HTTPException if authentication fails
    """
    if not validate_api_key(request, api_key_header):
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
