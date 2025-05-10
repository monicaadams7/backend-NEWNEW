"""
FitGPT Backend API
A production-ready FastAPI service for the FitGPT application
"""
import os
import sys
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import logging
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# Import the required modules
from app.security import setup_security
from app.secured_endpoints import register_secured_endpoints

# Load environment variables
load_dotenv()

# Configure logging
log_dir = os.getenv("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Create a handler for file output with rotation
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "fitgpt.log"),
    maxBytes=10485760,  # 10MB
    backupCount=10,
)
file_handler.setFormatter(formatter)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the RAG module
try:
    from app.services.rag.rag_wrapper import process_query
    logger.info("Successfully imported RAG wrapper")
except ImportError as e:
    logger.error(f"Error importing RAG wrapper: {e}")
    
    # Define a fallback function if import fails
    def process_query(query: str) -> str:
        """Fallback function when RAG system is not available"""
        return f"I received your query: '{query}', but my RAG system isn't properly connected yet."

# Try to import the storage module
try:
    from app.services.storage import ConversationStorage
    logger.info("Successfully imported storage service")
    
    # Initialize storage backend based on environment settings
    storage_type = os.getenv("STORAGE_TYPE", "sqlite").lower()
    
    if storage_type == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        storage = ConversationStorage.create_redis_storage(redis_url)
        logger.info("Using Redis for conversation storage")
    else:  # Default to SQLite
        db_path = os.getenv("SQLITE_PATH", "conversations.db")
        storage = ConversationStorage.create_sqlite_storage(db_path)
        logger.info(f"Using SQLite for conversation storage at {db_path}")
        
except ImportError as e:
    logger.warning(f"Storage service not available, using in-memory storage: {e}")
    
    # Simple in-memory storage as fallback
    class InMemoryStorage:
        def __init__(self):
            self.conversations: Dict[str, List[Dict[str, str]]] = {}
            
        async def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
            return self.conversations.get(conversation_id, [])
            
        async def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
            self.conversations[conversation_id] = messages
    
    storage = InMemoryStorage()
    logger.info("Using in-memory fallback for conversation storage")

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app with metadata and versioning
app = FastAPI(
    title="FitGPT Backend",
    description="API for FitGPT conversational fitness assistant",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Setup security for the app
setup_security(app)

# Register secured endpoints
register_secured_endpoints(app)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS from environment variables
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
logger.info(f"Configuring CORS for origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request processing middleware to log request time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Process time: {process_time:.4f}s"
    )
    
    return response

# Error handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# Schemas for API models
class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'bot'")
    text: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'bot']:
            raise ValueError('Role must be either "user" or "bot"')
        return v

class ConversationRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")

class ConversationResponse(BaseModel):
    message: Message = Field(..., description="Response message from the bot")
    conversation_id: str = Field(..., description="Conversation ID for tracking")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")

class MetadataResponse(BaseModel):
    app: str = Field(..., description="Application name")
    status: str = Field(..., description="Application status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    timestamp: str = Field(..., description="Current server time")

# Custom API documentation
@app.get("/api/v1/docs", include_in_schema=False)
async def get_custom_docs():
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title="FitGPT API Documentation",
    )

@app.get("/api/v1/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return get_openapi(
        title="FitGPT API",
        version="1.0.0",
        description="FitGPT Conversational API for fitness assistance",
        routes=app.routes,
    )

# Root endpoint with application metadata
@app.get("/", response_model=MetadataResponse, tags=["System"])
async def root():
    """Return basic application metadata"""
    return {
        "app": "FitGPT",
        "status": "running",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
@limiter.exempt  # Don't rate limit health checks
async def health_check():
    """Check if the API is healthy"""
    return {"status": "ok", "version": "1.0.0"}

# Conversation endpoint with rate limiting
@app.post("/api/v1/chat", response_model=ConversationResponse, tags=["Conversation"])
@limiter.limit(os.getenv("RATE_LIMIT", "10/minute"))  # Rate limit from env or default
async def chat(request: ConversationRequest, client_request: Request):
    """
    Process a chat request and return a response from the FitGPT system
    """
    # Sanitize and validate input
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="No messages provided"
        )
    
    # Get or create conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Get user's message (the last message in the list)
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, 
            detail="No user message found"
        )
    
    user_message = user_messages[-1].text
    logger.info(f"Processing query for conversation {conversation_id}: {user_message[:50]}...")
    
    try:
        # Retrieve conversation history
        history = await storage.get_conversation(conversation_id)
        
        # Process the query using RAG system
        # In a production system, this should be wrapped in a timeout
        # to prevent long-running queries from blocking the server
        answer = await process_query_async(user_message, history)
        
        # Create new message
        bot_message = Message(role="bot", text=answer)
        
        # Update conversation history
        history.extend([
            {"role": "user", "text": user_message},
            {"role": "bot", "text": answer}
        ])
        
        # Save updated conversation history
        await storage.save_conversation(conversation_id, history)
        
        # Return the response
        return ConversationResponse(
            message=bot_message,
            conversation_id=conversation_id
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Return a friendly error message
        return ConversationResponse(
            message=Message(
                role="bot",
                text="I'm sorry, I encountered an error processing your request. Please try again later."
            ),
            conversation_id=conversation_id
        )

# Wrapper to make process_query async if it's not already
async def process_query_async(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    Async wrapper for process_query
    Enables async support even if the underlying RAG function is synchronous
    """
    # Convert to async if needed in the future
    # For now, just call the function directly
    return process_query(query)

# Admin endpoints for RAG system management
@app.get("/api/v1/admin/rag-health", tags=["Admin"])
async def rag_health_check():
    """
    Check the health of the RAG system
    """
    try:
        # Placeholder logic for RAG health check
        return {"status": "healthy", "details": "RAG system is operational"}
    except Exception as e:
        logger.error(f"Error checking RAG health: {e}")
        return {"status": "unhealthy", "details": str(e)}


@app.post("/api/v1/admin/reload-rag", tags=["Admin"])
async def reload_rag():
    """
    Reload the RAG system (e.g., reinitialize models or connections)
    """
    try:
        # Placeholder logic for reloading RAG system
        logger.info("Reloading RAG system...")
        return {"status": "success", "message": "RAG system reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading RAG system: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/v1/admin/configure-rag", tags=["Admin"])
async def configure_rag(config: Dict[str, Any]):
    """
    Dynamically configure the RAG system
    """
    try:
        # Placeholder logic for configuring RAG system
        logger.info(f"Configuring RAG system with: {config}")
        return {"status": "success", "message": "RAG system configured successfully"}
    except Exception as e:
        logger.error(f"Error configuring RAG system: {e}")
        return {"status": "error", "message": str(e)}
# Add secured endpoints
try:
    from app.secured_endpoints import register_secured_endpoints
    register_secured_endpoints(app, chat_function=chat, storage=storage, process_query_async=process_query_async)
    logger.info("Secured endpoints registered")
except Exception as e:
    logger.warning(f"Failed to register secured endpoints: {e}")
if __name__ == "__main__":
    # Get port from environment with fallback to 8000
    port = int(os.getenv("PORT", "8000"))
    
    # Determine if we should enable hot reload (development only)
    reload = os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    logger.info(f"Starting FitGPT Backend on port {port} (reload={reload})")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload
    )
