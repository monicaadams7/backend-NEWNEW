"""
Security Middleware for FitGPT Backend
Provides middleware for headers like CSP, X-Frame-Options, etc.
"""
import os
import logging
from typing import Callable, List, Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers
from starlette.types import ASGIApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Default security headers
DEFAULT_SECURITY_HEADERS = {
    # Prevent the app from being embedded in an iframe (clickjacking protection)
    "X-Frame-Options": "DENY",
    
    # Controls how much information the browser includes with navigation errors
    "X-Content-Type-Options": "nosniff",
    
    # Enables browser XSS filtering
    "X-XSS-Protection": "1; mode=block",
    
    # Prevents browser MIME type sniffing
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    
    # Referrer Policy controls how much information is sent with requests
    "Referrer-Policy": "strict-origin-when-cross-origin",
    
    # Feature Policy restricts browser features
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
}

# Default Content Security Policy
DEFAULT_CSP = {
    "default-src": ["'self'"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "script-src": ["'self'"],
    "img-src": ["'self'", "data:"],
    "connect-src": ["'self'"],
    "font-src": ["'self'"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
    "upgrade-insecure-requests": []
}

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses
    """
    def __init__(
        self, 
        app: ASGIApp, 
        security_headers: Optional[Dict[str, str]] = None,
        csp_directives: Optional[Dict[str, List[str]]] = None,
        enable_csp: bool = True
    ):
        super().__init__(app)
        self.security_headers = security_headers or DEFAULT_SECURITY_HEADERS.copy()
        self.csp_directives = csp_directives or DEFAULT_CSP.copy()
        self.enable_csp = enable_csp
        
        # Parse CSP from environment if available
        self._parse_env_csp()
        
        # Build CSP header
        if self.enable_csp:
            self.security_headers["Content-Security-Policy"] = self._build_csp_header()
            
        logger.info(f"Security headers middleware initialized with {len(self.security_headers)} headers")
        
    def _parse_env_csp(self):
        """Parse CSP directives from environment variables"""
        env_csp = os.getenv("CONTENT_SECURITY_POLICY")
        if env_csp:
            try:
                # Environment CSP is in the format "default-src 'self'; script-src 'self'"
                directives = {}
                for directive in env_csp.split(";"):
                    directive = directive.strip()
                    if not directive:
                        continue
                        
                    parts = directive.split()
                    if len(parts) >= 1:
                        name = parts[0]
                        values = parts[1:]
                        directives[name] = values
                
                if directives:
                    self.csp_directives = directives
                    logger.info(f"Loaded CSP from environment with {len(directives)} directives")
            except Exception as e:
                logger.error(f"Error parsing CSP from environment: {e}")

    def _build_csp_header(self) -> str:
        """Build the Content-Security-Policy header value from directives"""
        parts = []
        
        for directive, sources in self.csp_directives.items():
            if not sources:
                # Directives like upgrade-insecure-requests don't have values
                parts.append(directive)
            else:
                parts.append(f"{directive} {' '.join(sources)}")
                
        return "; ".join(parts)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to all responses"""
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
            
        return response

class XSSProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add X-XSS-Protection header to all responses
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate incoming requests
    """
    async def dispatch(self, request: Request, call_next):
        # Add basic input checks or validation logic here
        # Example: Log the request method and URL
        logger.info(f"Incoming request: {request.method} {request.url}")
        
        # Process the request
        response = await call_next(request)
        return response

def setup_security_middleware(app: FastAPI):
    """
    Setup security middleware for the FastAPI app.
    Adds security headers, XSS protection, and request validation to all responses.
    """
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(XSSProtectionMiddleware)
    app.add_middleware(RequestValidationMiddleware)