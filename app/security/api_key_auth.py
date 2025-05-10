"""
API Key Authentication for FitGPT Backend
Provides dependency for validating API keys from request headers or query parameters
"""
import os
import time
import logging
import hashlib
import hmac
from typing import Dict, List, Optional, Union, Callable, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# API key header and query parameter names
API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
API_KEY_QUERY = os.getenv("API_KEY_QUERY", "api_key")

# API key validation schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)

# API key storage options: in-memory, Redis, or SQLite
KEY_STORAGE = os.getenv("KEY_STORAGE", "memory").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create a data directory in the project root for SQLite files
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(data_dir, "api_keys.db"))

# Key prefix for Redis storage
REDIS_KEY_PREFIX = "fitgpt:api_key:"

# In-memory storage as fallback
api_keys: Dict[str, Dict[str, Any]] = {}

# Initialize key storage
if KEY_STORAGE == "redis" and REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()  # Test connection
        logger.info("Redis initialized for API key storage")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}, falling back to in-memory storage")
        KEY_STORAGE = "memory"
elif KEY_STORAGE == "sqlite":
    try:
        import sqlite3
        
        # Connect and create table if it doesn't exist
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                api_key TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                rate_limit TEXT DEFAULT '60/minute',
                created_at INTEGER NOT NULL,
                expires_at INTEGER,
                is_active INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("SQLite initialized for API key storage")
    except Exception as e:
        logger.warning(f"SQLite initialization failed: {e}, falling back to in-memory storage")
        KEY_STORAGE = "memory"
else:
    logger.info("Using in-memory storage for API keys")
    # Initialize with an admin key from environment if available
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key:
        api_keys[admin_key] = {
            "user_id": "admin",
            "role": "admin",
            "rate_limit": "1000/minute",
            "created_at": int(time.time()),
            "expires_at": None,
            "is_active": True
        }
        logger.info("Added admin API key from environment")

class APIKeyInfo(BaseModel):
    """Schema for API key information"""
    key_id: str
    user_id: str
    role: str = "user"
    rate_limit: str = "60/minute"
    created_at: int
    expires_at: Optional[int] = None
    is_active: bool = True

# Secure key generation function
def generate_api_key(key_id: str, secret: str = None) -> str:
    """Generate a secure API key with optional secret"""
    if not secret:
        secret = os.getenv("API_KEY_SECRET", "your-secret-key")
    
    timestamp = str(int(time.time()))
    
    # Create a unique, hard-to-guess key
    message = f"{key_id}:{timestamp}".encode('utf-8')
    signature = hmac.new(
        secret.encode('utf-8'),
        message,
        hashlib.sha256
    ).hexdigest()
    
    # Return a formatted key
    return f"fgpt_{key_id}_{timestamp[:5]}_{signature[:16]}"

# Redis methods for key management
def _redis_get_key_info(api_key: str) -> Optional[Dict[str, Any]]:
    """Get API key info from Redis"""
    if not redis_client:
        return None
    
    key_data = redis_client.get(f"{REDIS_KEY_PREFIX}{api_key}")
    if not key_data:
        return None
    
    try:
        return APIKeyInfo.parse_raw(key_data).dict()
    except Exception as e:
        logger.error(f"Error parsing API key data: {e}")
        return None

def _redis_save_key(api_key: str, key_info: Dict[str, Any]) -> bool:
    """Save API key to Redis"""
    if not redis_client:
        return False
    
    try:
        # Convert to APIKeyInfo for validation
        key_model = APIKeyInfo(**key_info)
        # Save with TTL if expires_at is set
        if key_model.expires_at:
            ttl = max(0, key_model.expires_at - int(time.time()))
            redis_client.setex(
                f"{REDIS_KEY_PREFIX}{api_key}",
                ttl,
                key_model.json()
            )
        else:
            redis_client.set(
                f"{REDIS_KEY_PREFIX}{api_key}",
                key_model.json()
            )
        return True
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        return False

def _redis_delete_key(api_key: str) -> bool:
    """Delete API key from Redis"""
    if not redis_client:
        return False
    
    try:
        return redis_client.delete(f"{REDIS_KEY_PREFIX}{api_key}") > 0
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        return False

# SQLite methods for key management
def _sqlite_get_key_info(api_key: str) -> Optional[Dict[str, Any]]:
    """Get API key info from SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key_id, user_id, role, rate_limit, created_at, expires_at, is_active "
            "FROM api_keys WHERE api_key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (api_key, int(time.time()))
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return {
            "key_id": row[0],
            "user_id": row[1],
            "role": row[2],
            "rate_limit": row[3],
            "created_at": row[4],
            "expires_at": row[5],
            "is_active": bool(row[6])
        }
    except Exception as e:
        logger.error(f"Error getting API key from SQLite: {e}")
        return None

def _sqlite_save_key(api_key: str, key_info: Dict[str, Any]) -> bool:
    """Save API key to SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Check if key exists
        cursor.execute("SELECT key_id FROM api_keys WHERE api_key = ?", (api_key,))
        exists = cursor.fetchone() is not None
        
        if exists:
            # Update existing key
            cursor.execute(
                "UPDATE api_keys SET user_id = ?, role = ?, rate_limit = ?, "
                "expires_at = ?, is_active = ? WHERE api_key = ?",
                (
                    key_info["user_id"],
                    key_info["role"],
                    key_info["rate_limit"],
                    key_info["expires_at"],
                    1 if key_info["is_active"] else 0,
                    api_key
                )
            )
        else:
            # Insert new key
            cursor.execute(
                "INSERT INTO api_keys (key_id, api_key, user_id, role, rate_limit, "
                "created_at, expires_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key_info["key_id"],
                    api_key,
                    key_info["user_id"],
                    key_info["role"],
                    key_info["rate_limit"],
                    key_info["created_at"],
                    key_info["expires_at"],
                    1 if key_info["is_active"] else 0
                )
            )
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving API key to SQLite: {e}")
        return False

def _sqlite_delete_key(api_key: str) -> bool:
    """Delete API key from SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE api_key = ?", (api_key,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    except Exception as e:
        logger.error(f"Error deleting API key from SQLite: {e}")
        return False

# Key management interface
def get_key_info(api_key: str) -> Optional[Dict[str, Any]]:
    """Get API key information from the configured storage"""
    if KEY_STORAGE == "redis" and REDIS_AVAILABLE:
        return _redis_get_key_info(api_key)
    elif KEY_STORAGE == "sqlite":
        return _sqlite_get_key_info(api_key)
    else:
        # In-memory storage
        return api_keys.get(api_key)

def save_key(api_key: str, key_info: Dict[str, Any]) -> bool:
    """Save API key to the configured storage"""
    if KEY_STORAGE == "redis" and REDIS_AVAILABLE:
        return _redis_save_key(api_key, key_info)
    elif KEY_STORAGE == "sqlite":
        return _sqlite_save_key(api_key, key_info)
    else:
        # In-memory storage
        api_keys[api_key] = key_info
        return True

def delete_key(api_key: str) -> bool:
    """Delete API key from the configured storage"""
    if KEY_STORAGE == "redis" and REDIS_AVAILABLE:
        return _redis_delete_key(api_key)
    elif KEY_STORAGE == "sqlite":
        return _sqlite_delete_key(api_key)
    else:
        # In-memory storage
        if api_key in api_keys:
            del api_keys[api_key]
            return True
        return False

def create_api_key(user_id: str, role: str = "user", expires_in: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a new API key for a user
    
    Args:
        user_id: User identifier
        role: User role (default: "user")
        expires_in: Seconds until key expiration (default: None, never expires)
        
    Returns:
        Dict with api_key and key information
    """
    key_id = f"{user_id}_{int(time.time())}"
    api_key = generate_api_key(key_id)
    
    # Create key info
    key_info = {
        "key_id": key_id,
        "user_id": user_id,
        "role": role,
        "rate_limit": "1000/minute" if role == "admin" else "60/minute",
        "created_at": int(time.time()),
        "expires_at": int(time.time()) + expires_in if expires_in else None,
        "is_active": True
    }
    
    # Save the key
    save_key(api_key, key_info)
    
    # Return key and info
    return {
        "api_key": api_key,
        **key_info
    }

def revoke_api_key(api_key: str) -> bool:
    """Revoke (delete) an API key"""
    return delete_key(api_key)

# FastAPI dependency for API key validation
async def get_api_key(
    api_key_header: str = Depends(api_key_header),
    api_key_query: str = Depends(api_key_query),
    request: Request = None
) -> Dict[str, Any]:
    """
    Validate API key from header or query parameter
    Returns key info if valid, raises HTTPException if invalid
    
    Usage:
        @app.get("/protected")
        async def protected_route(key_info: Dict = Depends(get_api_key)):
            return {"message": f"Hello {key_info['user_id']}"}
    """
    # Get the API key from header or query param
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # Get key info
    key_info = get_key_info(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # Check if key is active
    if not key_info.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # Check expiration
    expires_at = key_info.get("expires_at")
    if expires_at and int(time.time()) > expires_at:
        # Delete expired key
        delete_key(api_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    # Store the key info in request state for use in route handlers
    if request:
        request.state.key_info = key_info
    
    return key_info

# Role-based access control dependency
def require_role(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control
    
    Usage:
        @app.get("/admin-only")
        async def admin_only(key_info: Dict = Depends(get_api_key), _: None = Depends(require_role(["admin"]))):
            return {"message": "You have admin access"}
    """
    async def role_validator(key_info: Dict = Depends(get_api_key)):
        user_role = key_info.get("role", "user")
        
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
            )
        
        return True
    
    return role_validator

# Custom rate limiter by user ID or API key
def rate_limit_by_key(rate_limit: Optional[str] = None):
    """
    Custom rate limiter that uses the user's specific rate limit from their API key
    
    Usage:
        @app.get("/custom-rate-limited")
        @limiter.limit(rate_limit_by_key())
        async def custom_rate_limited(key_info: Dict = Depends(get_api_key)):
            return {"message": "Rate limited by user"}
    """
    def limit_callable(request: Request):
        # Get key info from request state (set by get_api_key)
        key_info = getattr(request.state, "key_info", {})
        
        # Use provided rate limit, or get from key_info, or use default
        limit = rate_limit or key_info.get("rate_limit", "60/minute")
        
        # Use user_id as the key for rate limiting
        identifier = key_info.get("user_id", "anonymous")
        
        return limit, identifier
    
    return limit_callable

# Helper to extract user ID from API key authentication
def get_current_user_id(key_info: Dict = Depends(get_api_key)) -> str:
    """
    Extract user ID from API key info
    
    Usage:
        @app.get("/user-specific")
        async def user_specific(user_id: str = Depends(get_current_user_id)):
            return {"message": f"Hello {user_id}"}
    """
    return key_info.get("user_id", "anonymous")

# Initialize admin key on module load if specified
admin_key_id = os.getenv("ADMIN_KEY_ID")
admin_key_secret = os.getenv("ADMIN_KEY_SECRET")

if admin_key_id and admin_key_secret:
    api_key = generate_api_key(admin_key_id, admin_key_secret)
    
    # Add a check before saving
    key_info = get_key_info(api_key)
    if not key_info:
        key_info = {
            "key_id": admin_key_id,
            "user_id": "admin",
            "role": "admin",
            "rate_limit": "1000/minute",
            "created_at": int(time.time()),
            "expires_at": None,
            "is_active": True
        }
        save_key(api_key, key_info)
        logger.info(f"Admin API key created: {api_key}")
    else:
        logger.info(f"Admin API key already exists")