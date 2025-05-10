"""
Enhanced Health Check for FitGPT Backend
Provides health check endpoint with Redis and vector DB connectivity checks
"""
import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Health check router
health_router = APIRouter(prefix="/api/v1", tags=["System"])

class ComponentHealth(BaseModel):
    """Health status of a component"""
    status: str
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_checked: str

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    timestamp: str
    environment: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]

# Start time for uptime calculation
START_TIME = time.time()

# Cache health check results to prevent hammering services
# Format: {"component_name": {"status": "ok", "last_checked": timestamp}}
health_cache = {}
CACHE_TTL_SECONDS = int(os.getenv("HEALTH_CACHE_TTL", "60"))

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity"""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    component_name = "redis"
    
    # Check cache first
    cache_entry = health_cache.get(component_name)
    if cache_entry and (time.time() - cache_entry.get("timestamp", 0) < CACHE_TTL_SECONDS):
        return cache_entry
    
    start_time = time.time()
    status = "ok"
    message = None
    
    try:
        # Import Redis only when needed
        import redis
        
        # Test connection
        client = redis.Redis.from_url(redis_url)
        client.ping()
        
    except ImportError:
        status = "warning"
        message = "Redis package not installed"
        
    except redis.ConnectionError as e:
        status = "error"
        message = f"Redis connection error: {str(e)}"
        
    except Exception as e:
        status = "error"
        message = f"Redis error: {str(e)}"
        
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Cache result
    result = {
        "status": status,
        "latency_ms": latency_ms,
        "message": message,
        "timestamp": time.time(),
        "last_checked": datetime.utcnow().isoformat()
    }
    health_cache[component_name] = result
    
    return result

async def check_vector_db_health() -> Dict[str, Any]:
    """Check Vector DB connectivity"""
    component_name = "vector_db"
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    
    # Check cache first
    cache_entry = health_cache.get(component_name)
    if cache_entry and (time.time() - cache_entry.get("timestamp", 0) < CACHE_TTL_SECONDS):
        return cache_entry
    
    start_time = time.time()
    status = "ok"
    message = None
    
    try:
        # Check if Chroma is installed
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            status = "warning"
            message = "Chroma package not installed"
            raise ImportError(message)
        
        # Check if vector DB path exists
        if not os.path.exists(vector_db_path):
            status = "warning"
            message = f"Vector DB path not found: {vector_db_path}"
            raise FileNotFoundError(message)
            
        # Try to connect to vector DB
        # This is a simple check that just verifies the directory structure
        # without loading the entire DB, which could be expensive
        metadata_path = os.path.join(vector_db_path, "chroma.sqlite3")
        if not os.path.exists(metadata_path):
            status = "warning"
            message = f"Vector DB metadata not found at {metadata_path}"
            
        # Could add more checks here if needed
        
    except ImportError:
        # Already handled above
        pass
        
    except FileNotFoundError:
        # Already handled above
        pass
        
    except Exception as e:
        status = "error"
        message = f"Vector DB error: {str(e)}"
        
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Cache result
    result = {
        "status": status,
        "latency_ms": latency_ms,
        "message": message,
        "timestamp": time.time(),
        "last_checked": datetime.utcnow().isoformat()
    }
    health_cache[component_name] = result
    
    return result

async def check_rag_system_health() -> Dict[str, Any]:
    """Check RAG system health"""
    component_name = "rag_system"
    
    # Check cache first
    cache_entry = health_cache.get(component_name)
    if cache_entry and (time.time() - cache_entry.get("timestamp", 0) < CACHE_TTL_SECONDS):
        return cache_entry
    
    start_time = time.time()
    status = "ok"
    message = None
    
    try:
        # Try to import RAG wrapper
        try:
            from app.services.rag.rag_wrapper import get_system_health
            
            # Get system health
            health_metrics = get_system_health()
            
            # Check if system is healthy
            if not health_metrics.get("is_healthy", True):
                status = "error"
                message = "RAG system is unhealthy"
                
            # Add additional info to message
            if status == "ok":
                processed = health_metrics.get("total_queries", 0)
                failures = health_metrics.get("failed_queries", 0)
                message = f"Processed {processed} queries with {failures} failures"
                
        except ImportError:
            status = "warning"
            message = "RAG wrapper not found"
            
        except Exception as e:
            status = "error"
            message = f"RAG system error: {str(e)}"
            
    except Exception as e:
        status = "error"
        message = f"Error checking RAG system: {str(e)}"
        
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Cache result
    result = {
        "status": status,
        "latency_ms": latency_ms,
        "message": message,
        "timestamp": time.time(),
        "last_checked": datetime.utcnow().isoformat()
    }
    health_cache[component_name] = result
    
    return result

async def check_database_health() -> Dict[str, Any]:
    """Check SQLite or database connectivity"""
    component_name = "database"
    
    # Check cache first
    cache_entry = health_cache.get(component_name)
    if cache_entry and (time.time() - cache_entry.get("timestamp", 0) < CACHE_TTL_SECONDS):
        return cache_entry
    
    start_time = time.time()
    status = "ok"
    message = None
    
    try:
        # Check SQLite
        import sqlite3
        
        # Try to connect to database
        db_path = os.getenv("SQLITE_PATH", "conversations.db")
        
        # Just test if we can create a connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Simple query to test connection
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        message = f"SQLite version: {version}"
        
        conn.close()
        
    except ImportError:
        status = "warning"
        message = "SQLite package not installed"
        
    except sqlite3.Error as e:
        status = "error"
        message = f"Database error: {str(e)}"
        
    except Exception as e:
        status = "error"
        message = f"Error checking database: {str(e)}"
        
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Cache result
    result = {
        "status": status,
        "latency_ms": latency_ms,
        "message": message,
        "timestamp": time.time(),
        "last_checked": datetime.utcnow().isoformat()
    }
    health_cache[component_name] = result
    
    return result

@health_router.get("/health", response_model=HealthCheckResponse)
async def health_check(full: bool = False):
    """
    Enhanced health check endpoint
    
    Args:
        full: Whether to perform full health check (may be slower)
        
    Returns:
        Health check status for all components
    """
    # Basic health check
    components = {
        "api": {
            "status": "ok",
            "latency_ms": 0.0,
            "message": None,
            "last_checked": datetime.utcnow().isoformat()
        }
    }
    
    # Check Redis
    components["redis"] = await check_redis_health()
    
    # Check database
    components["database"] = await check_database_health()
    
    # Full health check
    if full:
        # Check Vector DB
        components["vector_db"] = await check_vector_db_health()
        
        # Check RAG system
        components["rag_system"] = await check_rag_system_health()
    
    # Determine overall status
    status = "ok"
    for component, health in components.items():
        if health["status"] == "error":
            status = "error"
            break
        elif health["status"] == "warning" and status != "error":
            status = "warning"
    
    return HealthCheckResponse(
        status=status,
        version=os.getenv("VERSION", "1.0.0"),
        timestamp=datetime.utcnow().isoformat(),
        environment=os.getenv("ENVIRONMENT", "development"),
        uptime_seconds=time.time() - START_TIME,
        components={
            name: ComponentHealth(
                status=info["status"],
                latency_ms=info["latency_ms"],
                message=info["message"],
                last_checked=info["last_checked"]
            )
            for name, info in components.items()
        }
    )

@health_router.get("/ready")
async def readiness_check():
    """
    Simple readiness check endpoint
    Returns 200 if the service is ready to accept traffic
    """
    # Check critical components
    redis_health = await check_redis_health()
    db_health = await check_database_health()
    
    # Service is ready if both Redis and database are ok or in warning state
    # But not in error state
    if redis_health["status"] == "error" or db_health["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    return {"status": "ready"}

@health_router.get("/live")
async def liveness_check():
    """
    Simple liveness check endpoint
    Returns 200 if the service is alive
    """
    return {"status": "alive"}

def setup_health_routes(app):
    """
    Add health check routes to the FastAPI app
    
    Args:
        app: The FastAPI app instance
        
    Usage:
        from app.health.health_check import setup_health_routes
        
        app = FastAPI()
        setup_health_routes(app)
    """
    app.include_router(health_router)
    return app