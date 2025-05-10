"""
Secure Entry Point for FitGPT Backend
Wraps the original app with security features
"""
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your main app - avoid circular imports
from app.main import app as base_app
from app.secured_endpoints import register_secured_endpoints
from app.security import setup_security

# Apply security to the base app
app = setup_security(base_app)

# Add secured endpoints if not already done in main.py
try:
    # Check if secured endpoints exist already
    has_secured_endpoints = any(route.path.startswith("/api/v1/secured") for route in app.routes)
    
    if not has_secured_endpoints:
        logger.info("Registering secured endpoints")
        register_secured_endpoints(app)
except Exception as e:
    logger.error(f"Error registering secured endpoints: {e}")

logger.info("Security components initialized")

# This is imported by the run_app.py script
# No need to run uvicorn here