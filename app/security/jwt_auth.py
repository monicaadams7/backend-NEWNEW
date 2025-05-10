"""
JWT Authentication for FitGPT Backend
Provides dependencies for JWT-based authentication and role-based access control
"""
from typing import Tuple

import os
import time
import logging
from typing import Dict, List, Optional, Union, Callable, Any
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Configure OAuth2 for token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User storage options: in-memory, Redis, or SQLite
USER_STORAGE = os.getenv("USER_STORAGE", "sqlite").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Create a data directory in the project root for SQLite files
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(data_dir, "users.db"))

# Redis key prefix for users
REDIS_USER_PREFIX = "fitgpt:user:"

# In-memory storage as fallback
users_db: Dict[str, Dict[str, Any]] = {}

# Initialize user storage
if USER_STORAGE == "redis":
    try:
        import redis
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()  # Test connection
        logger.info("Redis initialized for user storage")
    except ImportError:
        logger.warning("Redis package not installed, falling back to SQLite")
        USER_STORAGE = "sqlite"
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}, falling back to SQLite")
        USER_STORAGE = "sqlite"

if USER_STORAGE == "sqlite":
    try:
        import sqlite3
        
        # Connect and create users table if it doesn't exist
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                hashed_password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        ''')
        
        # Create refresh tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                refresh_token TEXT UNIQUE NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create admin user if it doesn't exist
        admin_username = os.getenv("ADMIN_USERNAME")
        admin_password = os.getenv("ADMIN_PASSWORD")
        
        if admin_username and admin_password:
            admin_password_hash = pwd_context.hash(admin_password)
            
            # Check if admin user exists
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (admin_username,))
            admin_exists = cursor.fetchone() is not None
            
            if not admin_exists:
                now = int(time.time())
                cursor.execute(
                    "INSERT INTO users (user_id, username, email, hashed_password, role, is_active, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"admin_{now}",
                        admin_username,
                        os.getenv("ADMIN_EMAIL", f"{admin_username}@fitgpt.example"),
                        admin_password_hash,
                        "admin",
                        1,
                        now,
                        now
                    )
                )
                logger.info(f"Created admin user: {admin_username}")
        
        conn.commit()
        conn.close()
        logger.info("SQLite initialized for user storage")
    except Exception as e:
        logger.warning(f"SQLite initialization failed: {e}, falling back to in-memory storage")
        USER_STORAGE = "memory"
        
        # Add admin user to in-memory storage
        admin_username = os.getenv("ADMIN_USERNAME")
        admin_password = os.getenv("ADMIN_PASSWORD")
        
        if admin_username and admin_password:
            admin_password_hash = pwd_context.hash(admin_password)
            now = int(time.time())
            users_db[admin_username] = {
                "user_id": f"admin_{now}",
                "username": admin_username,
                "email": os.getenv("ADMIN_EMAIL", f"{admin_username}@fitgpt.example"),
                "hashed_password": admin_password_hash,
                "role": "admin",
                "is_active": True,
                "created_at": now,
                "updated_at": now
            }
            logger.info(f"Created admin user in memory: {admin_username}")

# Pydantic models for user data
class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None
    password: str
    role: str = "user"
    
    @validator('username')
    def username_min_length(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v
    
    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserInfo(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    created_at: int
    updated_at: int

class UserInDB(UserInfo):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: int
    
class TokenPayload(BaseModel):
    sub: str  # username
    role: str
    exp: Optional[int] = None
    user_id: str

# Helper functions for password handling
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that the plain password matches the hashed password"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password for storage"""
    return pwd_context.hash(password)

# Redis methods for user management
def _redis_get_user(username: str) -> Optional[UserInDB]:
    """Get user from Redis"""
    if not redis_client:
        return None
    
    user_data = redis_client.get(f"{REDIS_USER_PREFIX}{username}")
    if not user_data:
        return None
    
    try:
        return UserInDB.parse_raw(user_data)
    except Exception as e:
        logger.error(f"Error parsing user data: {e}")
        return None

def _redis_create_user(user: UserCreate) -> UserInDB:
    """Create a user in Redis"""
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not available")
    
    # Check if username exists
    if redis_client.get(f"{REDIS_USER_PREFIX}{user.username}"):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Generate user_id and timestamps
    now = int(time.time())
    user_id = f"user_{user.username}_{now}"
    
    # Create user document
    user_in_db = UserInDB(
        user_id=user_id,
        username=user.username,
        email=user.email,
        role=user.role,
        hashed_password=get_password_hash(user.password),
        is_active=True,
        created_at=now,
        updated_at=now
    )
    
    # Save to Redis
    redis_client.set(
        f"{REDIS_USER_PREFIX}{user.username}",
        user_in_db.json()
    )
    
    return user_in_db

def _redis_update_user(username: str, update_data: Dict[str, Any]) -> Optional[UserInDB]:
    """Update a user in Redis"""
    if not redis_client:
        return None
    
    # Get existing user
    user_data = redis_client.get(f"{REDIS_USER_PREFIX}{username}")
    if not user_data:
        return None
    
    try:
        user = UserInDB.parse_raw(user_data)
        
        # Update fields
        update_dict = {k: v for k, v in update_data.items() if k in user.__fields__}
        update_dict["updated_at"] = int(time.time())
        
        # Handle password separately
        if "password" in update_data:
            update_dict["hashed_password"] = get_password_hash(update_data["password"])
        
        # Create updated user
        updated_user = UserInDB(**{**user.dict(), **update_dict})
        
        # Save to Redis
        redis_client.set(
            f"{REDIS_USER_PREFIX}{username}",
            updated_user.json()
        )
        
        return updated_user
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return None

def _redis_delete_user(username: str) -> bool:
    """Delete a user from Redis"""
    if not redis_client:
        return False
    
    return redis_client.delete(f"{REDIS_USER_PREFIX}{username}") > 0

def _redis_save_refresh_token(user_id: str, refresh_token: str, expires_at: int) -> bool:
    """Save refresh token to Redis"""
    if not redis_client:
        return False
    
    try:
        token_id = f"refresh_{user_id}_{int(time.time())}"
        token_data = {
            "token_id": token_id,
            "user_id": user_id,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "created_at": int(time.time())
        }
        
        # Save with expiration
        ttl = max(0, expires_at - int(time.time()))
        redis_client.setex(
            f"fitgpt:refresh_token:{refresh_token}",
            ttl,
            str(token_data)
        )
        return True
    except Exception as e:
        logger.error(f"Error saving refresh token: {e}")
        return False

def _redis_get_user_by_refresh_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """Get user from refresh token in Redis"""
    if not redis_client:
        return None
    
    try:
        token_data = redis_client.get(f"fitgpt:refresh_token:{refresh_token}")
        if not token_data:
            return None
        
        # Parse token data
        token_dict = eval(token_data)
        user_id = token_dict.get("user_id")
        
        # Find the user by scanning keys
        user_keys = redis_client.keys(f"{REDIS_USER_PREFIX}*")
        
        for key in user_keys:
            user_data = redis_client.get(key)
            try:
                user = UserInDB.parse_raw(user_data)
                if user.user_id == user_id:
                    return user.dict()
            except Exception:
                continue
                
        return None
    except Exception as e:
        logger.error(f"Error getting user by refresh token: {e}")
        return None

# SQLite methods for user management
def _sqlite_get_user(username: str) -> Optional[UserInDB]:
    """Get user from SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, username, email, hashed_password, role, is_active, "
            "created_at, updated_at FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return UserInDB(
            user_id=row[0],
            username=row[1],
            email=row[2],
            hashed_password=row[3],
            role=row[4],
            is_active=bool(row[5]),
            created_at=row[6],
            updated_at=row[7]
        )
    except Exception as e:
        logger.error(f"Error getting user from SQLite: {e}")
        return None

def _sqlite_create_user(user: UserCreate) -> UserInDB:
    """Create a user in SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (user.username,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Username already registered")
        
        # Check if email exists if provided
        if user.email:
            cursor.execute("SELECT username FROM users WHERE email = ?", (user.email,))
            if cursor.fetchone():
                conn.close()
                raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate user_id and timestamps
        now = int(time.time())
        user_id = f"user_{user.username}_{now}"
        
        # Hash password
        hashed_password = get_password_hash(user.password)
        
        # Insert user
        cursor.execute(
            "INSERT INTO users (user_id, username, email, hashed_password, role, "
            "is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                user.username,
                user.email,
                hashed_password,
                user.role,
                1,
                now,
                now
            )
        )
        
        conn.commit()
        conn.close()
        
        return UserInDB(
            user_id=user_id,
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            role=user.role,
            is_active=True,
            created_at=now,
            updated_at=now
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user in SQLite: {e}")
        raise HTTPException(status_code=500, detail="Error creating user")

def _sqlite_update_user(username: str, update_data: Dict[str, Any]) -> Optional[UserInDB]:
    """Update a user in SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Get existing user
        cursor.execute(
            "SELECT user_id, username, email, hashed_password, role, is_active, "
            "created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Prepare updated values
        now = int(time.time())
        
        # Start with existing data
        user_id = row[0]
        updated_username = update_data.get("username", row[1])
        updated_email = update_data.get("email", row[2])
        updated_password = row[3]
        updated_role = update_data.get("role", row[4])
        updated_is_active = update_data.get("is_active", bool(row[5]))
        created_at = row[6]
        
        # Handle password separately
        if "password" in update_data:
            updated_password = get_password_hash(update_data["password"])
        
        # Update the user
        cursor.execute(
            "UPDATE users SET username = ?, email = ?, hashed_password = ?, role = ?, "
            "is_active = ?, updated_at = ? WHERE username = ?",
            (
                updated_username,
                updated_email,
                updated_password,
                updated_role,
                1 if updated_is_active else 0,
                now,
                username
            )
        )
        
        conn.commit()
        
        # Get updated user
        cursor.execute(
            "SELECT user_id, username, email, hashed_password, role, is_active, "
            "created_at, updated_at FROM users WHERE user_id = ?",
            (user_id,)
        )
        updated_row = cursor.fetchone()
        conn.close()
        
        if not updated_row:
            return None
            
        return UserInDB(
            user_id=updated_row[0],
            username=updated_row[1],
            email=updated_row[2],
            hashed_password=updated_row[3],
            role=updated_row[4],
            is_active=bool(updated_row[5]),
            created_at=updated_row[6],
            updated_at=updated_row[7]
        )
    except Exception as e:
        logger.error(f"Error updating user in SQLite: {e}")
        return None

def _sqlite_delete_user(username: str) -> bool:
    """Delete a user from SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Delete the user
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    except Exception as e:
        logger.error(f"Error deleting user from SQLite: {e}")
        return False

def _sqlite_save_refresh_token(user_id: str, refresh_token: str, expires_at: int) -> bool:
    """Save refresh token to SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Generate token ID
        token_id = f"refresh_{user_id}_{int(time.time())}"
        now = int(time.time())
        
        # Insert token
        cursor.execute(
            "INSERT INTO refresh_tokens (token_id, user_id, refresh_token, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token_id, user_id, refresh_token, expires_at, now)
        )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        logger.error(f"Error saving refresh token to SQLite: {e}")
        return False

def _sqlite_get_user_by_refresh_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """Get user from refresh token in SQLite"""
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        
        # Get token info
        cursor.execute(
            "SELECT user_id FROM refresh_tokens WHERE refresh_token = ? AND expires_at > ?",
            (refresh_token, int(time.time()))
        )
        token_row = cursor.fetchone()
        
        if not token_row:
            conn.close()
            return None
        
        user_id = token_row[0]
        
        # Get user info
        cursor.execute(
            "SELECT user_id, username, email, hashed_password, role, is_active, "
            "created_at, updated_at FROM users WHERE user_id = ?",
            (user_id,)
        )
        user_row = cursor.fetchone()
        conn.close()
        
        if not user_row:
            return None
            
        return {
            "user_id": user_row[0],
            "username": user_row[1],
            "email": user_row[2],
            "hashed_password": user_row[3],
            "role": user_row[4],
            "is_active": bool(user_row[5]),
            "created_at": user_row[6],
            "updated_at": user_row[7]
        }
    except Exception as e:
        logger.error(f"Error getting user by refresh token from SQLite: {e}")
        return None

# Memory methods for user management
def _memory_get_user(username: str) -> Optional[UserInDB]:
    """Get user from memory"""
    user_data = users_db.get(username)
    if not user_data:
        return None
    
    return UserInDB(**user_data)

def _memory_create_user(user: UserCreate) -> UserInDB:
    """Create a user in memory"""
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    now = int(time.time())
    user_id = f"user_{user.username}_{now}"
    
    user_data = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "role": user.role,
        "is_active": True,
        "created_at": now,
        "updated_at": now
    }
    
    users_db[user.username] = user_data
    return UserInDB(**user_data)

def _memory_update_user(username: str, update_data: Dict[str, Any]) -> Optional[UserInDB]:
    """Update a user in memory"""
    if username not in users_db:
        return None
    
    user_data = users_db[username]
    
    # Update fields
    for key, value in update_data.items():
        if key == "password":
            user_data["hashed_password"] = get_password_hash(value)
        elif key in user_data:
            user_data[key] = value
    
    user_data["updated_at"] = int(time.time())
    
    return UserInDB(**user_data)

def _memory_delete_user(username: str) -> bool:
    """Delete a user from memory"""
    if username in users_db:
        del users_db[username]
        return True
    return False

# In-memory refresh token storage
refresh_tokens_db: Dict[str, Dict[str, Any]] = {}

def _memory_save_refresh_token(user_id: str, refresh_token: str, expires_at: int) -> bool:
    """Save refresh token to memory"""
    token_id = f"refresh_{user_id}_{int(time.time())}"
    
    refresh_tokens_db[refresh_token] = {
        "token_id": token_id,
        "user_id": user_id,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "created_at": int(time.time())
    }
    
    return True

def _memory_get_user_by_refresh_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """Get user from refresh token in memory"""
    token_data = refresh_tokens_db.get(refresh_token)
    if not token_data or token_data["expires_at"] < int(time.time()):
        return None
    
    user_id = token_data["user_id"]
    
    # Find user by user_id
    for username, user_data in users_db.items():
        if user_data["user_id"] == user_id:
            return user_data
    
    return None

# User management interface
def get_user(username: str) -> Optional[UserInDB]:
    """Get a user from the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_get_user(username)
    elif USER_STORAGE == "sqlite":
        return _sqlite_get_user(username)
    else:
        return _memory_get_user(username)

def create_user(user: UserCreate) -> UserInDB:
    """Create a user in the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_create_user(user)
    elif USER_STORAGE == "sqlite":
        return _sqlite_create_user(user)
    else:
        return _memory_create_user(user)

def update_user(username: str, update_data: Dict[str, Any]) -> Optional[UserInDB]:
    """Update a user in the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_update_user(username, update_data)
    elif USER_STORAGE == "sqlite":
        return _sqlite_update_user(username, update_data)
    else:
        return _memory_update_user(username, update_data)

def delete_user(username: str) -> bool:
    """Delete a user from the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_delete_user(username)
    elif USER_STORAGE == "sqlite":
        return _sqlite_delete_user(username)
    else:
        return _memory_delete_user(username)

def save_refresh_token(user_id: str, refresh_token: str, expires_at: int) -> bool:
    """Save a refresh token to the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_save_refresh_token(user_id, refresh_token, expires_at)
    elif USER_STORAGE == "sqlite":
        return _sqlite_save_refresh_token(user_id, refresh_token, expires_at)
    else:
        return _memory_save_refresh_token(user_id, refresh_token, expires_at)

def get_user_by_refresh_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """Get a user from refresh token in the configured storage"""
    if USER_STORAGE == "redis":
        return _redis_get_user_by_refresh_token(refresh_token)
    elif USER_STORAGE == "sqlite":
        return _sqlite_get_user_by_refresh_token(refresh_token)
    else:
        return _memory_get_user_by_refresh_token(refresh_token)

# Token functions
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    to_encode.update({"exp": expire.timestamp()})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt

def create_refresh_token(user_id: str) -> Tuple[str, int]:
    """Create a refresh token and save it"""
    # Calculate expiration
    expires_delta = timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    expires_at = int((datetime.utcnow() + expires_delta).timestamp())
    
    # Create the token
    refresh_token = f"refresh_{user_id}_{int(time.time())}_{os.urandom(8).hex()}"
    
    # Save the token
    save_refresh_token(user_id, refresh_token, expires_at)
    
    return refresh_token, expires_at

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password"""
    user = get_user(username)
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user

# FastAPI dependencies
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInfo:
    """
    Verify and extract user from JWT token
    
    Usage:
        @app.get("/users/me")
        async def read_users_me(current_user: UserInfo = Depends(get_current_user)):
            return current_user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
        
        token_data = TokenPayload(**payload)
        
        # Check token expiration
        if token_data.exp and int(time.time()) > token_data.exp:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get the user
    user = get_user(username)
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    # Return user info without the password
    return UserInfo(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at
    )

# Role-based access control
def require_role(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control
    
    Usage:
        @app.get("/admin-only")
        async def admin_only(current_user: UserInfo = Depends(get_current_user), 
                            _: None = Depends(require_role(["admin"]))):
            return {"message": "You have admin access"}
    """
    async def role_validator(current_user: UserInfo = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
            )
        return True
        
    return role_validator

# Endpoint to get the current user from headers or token
async def get_current_user_or_none(token: str = Depends(oauth2_scheme)):
    """
    Get current user or None if not authenticated
    Doesn't raise exceptions for unauthorized requests
    
    Usage:
        @app.get("/public-with-user-info")
        async def public_with_user_info(current_user: Optional[UserInfo] = Depends(get_current_user_or_none)):
            return {"message": f"Hello {'anonymous' if current_user is None else current_user.username}"}
    """
    try:
        return await get_current_user(token)
    except HTTPException:
        return None