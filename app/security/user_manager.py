"""
User Manager with Background Tasks for FitGPT Backend
Provides API endpoints and background tasks for user management
"""
import os
import time
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Union, Callable
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr, validator
import jwt
from dotenv import load_dotenv

# Import from our modules
from .api_key_auth import get_api_key, require_role, create_api_key
from .jwt_auth import (
    UserCreate, UserInfo, Token, authenticate_user, 
    create_access_token, create_refresh_token,
    get_current_user, get_user_by_refresh_token,
    get_user, create_user, update_user, delete_user
)
from .request_sanitizer import SanitizedModel, validate_username, validate_email, validate_password

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Background task queue
class BackgroundTaskQueue:
    """Simple background task queue for user creation and management"""
    def __init__(self, max_workers: int = 5):
        self.tasks: List[Dict[str, Any]] = []
        self.processing: Set[str] = set()
        self.max_workers = max_workers
        self.running = True
        self.worker_task = None
        
    async def start(self):
        """Start the background task worker"""
        self.running = True
        self.worker_task = asyncio.create_task(self.worker())
        
    async def stop(self):
        """Stop the background task worker"""
        self.running = False
        if self.worker_task:
            await self.worker_task
            
    def add_task(self, task_type: str, payload: Dict[str, Any], callback: Optional[Callable] = None) -> str:
        """Add a task to the queue"""
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "status": "pending",
            "created_at": int(time.time()),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "callback": callback
        }
        self.tasks.append(task)
        logger.info(f"Added {task_type} task {task_id} to queue")
        return task_id
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        for task in self.tasks:
            if task["id"] == task_id:
                # Return a copy without the callback
                result = task.copy()
                result.pop("callback", None)
                return result
        return None
        
    async def worker(self):
        """Background worker to process tasks"""
        logger.info("Starting background task worker")
        
        while self.running:
            # Process pending tasks
            for task in self.tasks:
                # Skip tasks that are already processing or completed
                if task["status"] not in ["pending", "retry"]:
                    continue
                    
                # Skip if we've reached max workers
                if len(self.processing) >= self.max_workers:
                    break
                    
                # Mark as processing
                task_id = task["id"]
                task["status"] = "processing"
                task["started_at"] = int(time.time())
                self.processing.add(task_id)
                
                # Process the task in a separate task
                asyncio.create_task(self._process_task(task))
                
            # Sleep to avoid cpu spin
            await asyncio.sleep(0.1)
            
    async def _process_task(self, task: Dict[str, Any]):
        """Process a single task"""
        task_id = task["id"]
        task_type = task["type"]
        payload = task["payload"]
        
        try:
            logger.info(f"Processing {task_type} task {task_id}")
            
            # Process based on task type
            if task_type == "create_user":
                result = await self._create_user(payload)
            elif task_type == "update_user":
                result = await self._update_user(payload)
            elif task_type == "delete_user":
                result = await self._delete_user(payload)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            # Update task with success
            task["status"] = "completed"
            task["completed_at"] = int(time.time())
            task["result"] = result
            
            # Call callback if provided
            if task["callback"]:
                task["callback"](task_id, True, result, None)
                
            logger.info(f"Completed {task_type} task {task_id}")
            
        except Exception as e:
            logger.error(f"Error processing {task_type} task {task_id}: {str(e)}")
            
            # Update task with error
            task["status"] = "failed"
            task["completed_at"] = int(time.time())
            task["error"] = str(e)
            
            # Call callback if provided
            if task["callback"]:
                task["callback"](task_id, False, None, str(e))
                
        finally:
            # Remove from processing set
            self.processing.discard(task_id)
            
    async def _create_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a user"""
        username = payload.get("username")
        email = payload.get("email")
        password = payload.get("password")
        role = payload.get("role", "user")
        
        # Create user using the imported function
        user_create = UserCreate(
            username=username,
            email=email,
            password=password,
            role=role
        )
        
        user = create_user(user_create)
        
        # Create API key if requested
        api_key = None
        create_key = payload.get("create_api_key", False)
        api_key_expires_in = payload.get("api_key_expires_in")
        
        if create_key:
            key_data = create_api_key(
                user_id=user.user_id,
                role=role,
                expires_in=api_key_expires_in
            )
            api_key = key_data["api_key"]
        
        # Return user info
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "api_key": api_key
        }
        
    async def _update_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update a user"""
        username = payload.get("username")
        updates = payload.get("updates", {})
        
        # Update user using the imported function
        user = update_user(username, updates)
        
        if not user:
            raise ValueError(f"User not found: {username}")
        
        # Return updated user info
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active
        }
        
    async def _delete_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a user"""
        username = payload.get("username")
        
        # Delete user using the imported function
        success = delete_user(username)
        
        if not success:
            raise ValueError(f"User not found: {username}")
        
        # Return result
        return {
            "username": username,
            "deleted": success
        }

# Initialize queue
task_queue = BackgroundTaskQueue()

# Start queue on module load
@asyncio.coroutine
def start_queue():
    yield from task_queue.start()

try:
    asyncio.get_event_loop().run_until_complete(start_queue())
    logger.info("Background task queue started")
except RuntimeError:
    logger.warning("No running event loop found. Task queue will be started when the app runs.")

# Authentication router
auth_router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

class TokenRequest(BaseModel):
    """Request for token generation"""
    username: str
    password: str
    
class RefreshTokenRequest(BaseModel):
    """Request for token refresh"""
    refresh_token: str
    
@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get access token using username and password
    This endpoint is compatible with OAuth2 password flow
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.user_id, "role": user.role}
    )
    
    # Create refresh token
    refresh_token, expires_at = create_refresh_token(user.user_id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_at=expires_at
    )

@auth_router.post("/token/refresh", response_model=Token)
async def refresh_access_token(request: RefreshTokenRequest):
    """Get a new access token using a refresh token"""
    
    # Verify refresh token
    user = get_user_by_refresh_token(request.refresh_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["user_id"], "role": user["role"]}
    )
    
    # Create new refresh token
    refresh_token, expires_at = create_refresh_token(user["user_id"])
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_at=expires_at
    )

@auth_router.post("/register", response_model=UserInfo)
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        created_user = create_user(user)
        return UserInfo(
            user_id=created_user.user_id,
            username=created_user.username,
            email=created_user.email,
            role=created_user.role,
            is_active=created_user.is_active,
            created_at=created_user.created_at,
            updated_at=created_user.updated_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@auth_router.get("/me", response_model=UserInfo)
async def read_users_me(current_user: UserInfo = Depends(get_current_user)):
    """Get information about the currently authenticated user"""
    return current_user

# User management router (admin only)
user_router = APIRouter(prefix="/api/v1/users", tags=["Users"])

class BatchUserCreate(SanitizedModel):
    """Request model for batch user creation"""
    users: List[UserCreate]
    create_api_keys: bool = False
    api_key_expires_in: Optional[int] = None
    
    @validator('users')
    def validate_users(cls, v):
        if not v:
            raise ValueError("At least one user must be provided")
        if len(v) > 100:
            raise ValueError("Maximum of 100 users per batch")
        return v

class BatchUserResponse(BaseModel):
    """Response model for batch user creation"""
    task_id: str
    status: str
    user_count: int
    
class UserCreationResponse(BaseModel):
    """Response model for user creation"""
    task_id: str
    status: str
    username: str
    
class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: str
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
@user_router.post("/batch", response_model=BatchUserResponse)
async def create_users_batch(
    request: BatchUserCreate,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_role(["admin"]))
):
    """Create multiple users in a background task (admin only)"""
    user_count = len(request.users)
    
    # Add background tasks for each user
    task_ids = []
    for user_data in request.users:
        task_id = task_queue.add_task(
            task_type="create_user",
            payload={
                "username": user_data.username,
                "email": user_data.email,
                "password": user_data.password,
                "role": user_data.role,
                "create_api_key": request.create_api_keys,
                "api_key_expires_in": request.api_key_expires_in
            }
        )
        task_ids.append(task_id)
    
    # Return batch status
    return BatchUserResponse(
        task_id=task_ids[0],  # Return the first task ID as the batch ID
        status="processing",
        user_count=user_count
    )

@user_router.post("", response_model=UserCreationResponse)
async def create_user_background(
    user: UserCreate,
    create_api_key: bool = False,
    api_key_expires_in: Optional[int] = None,
    _: None = Depends(require_role(["admin"]))
):
    """Create a single user in a background task (admin only)"""
    task_id = task_queue.add_task(
        task_type="create_user",
        payload={
            "username": user.username,
            "email": user.email,
            "password": user.password,
            "role": user.role,
            "create_api_key": create_api_key,
            "api_key_expires_in": api_key_expires_in
        }
    )
    
    return UserCreationResponse(
        task_id=task_id,
        status="processing",
        username=user.username
    )

@user_router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    _: None = Depends(require_role(["admin"]))
):
    """Get the status of a background task (admin only)"""
    task = task_queue.get_task_status(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}"
        )
    
    return TaskStatusResponse(
        task_id=task["id"],
        status=task["status"],
        created_at=task["created_at"],
        started_at=task["started_at"],
        completed_at=task["completed_at"],
        result=task["result"],
        error=task["error"]
    )

@user_router.put("/{username}", response_model=UserCreationResponse)
async def update_user_background(
    username: str,
    updates: Dict[str, Any],
    _: None = Depends(require_role(["admin"]))
):
    """Update a user in a background task (admin only)"""
    # Validate allowed update fields
    allowed_fields = ["email", "password", "role", "is_active"]
    update_data = {k: v for k, v in updates.items() if k in allowed_fields}
    
    task_id = task_queue.add_task(
        task_type="update_user",
        payload={
            "username": username,
            "updates": update_data
        }
    )
    
    return UserCreationResponse(
        task_id=task_id,
        status="processing",
        username=username
    )

@user_router.delete("/{username}", response_model=UserCreationResponse)
async def delete_user_background(
    username: str,
    _: None = Depends(require_role(["admin"]))
):
    """Delete a user in a background task (admin only)"""
    task_id = task_queue.add_task(
        task_type="delete_user",
        payload={
            "username": username
        }
    )
    
    return UserCreationResponse(
        task_id=task_id,
        status="processing",
        username=username
    )

# API key router
api_key_router = APIRouter(prefix="/api/v1/api-keys", tags=["API Keys"])

class APIKeyCreateRequest(BaseModel):
    """Request for API key creation"""
    user_id: str
    role: str = "user"
    expires_in: Optional[int] = None

class APIKeyResponse(BaseModel):
    """Response model for API key operations"""
    api_key: str
    key_id: str
    user_id: str
    role: str
    created_at: int
    expires_at: Optional[int] = None

@api_key_router.post("", response_model=APIKeyResponse)
async def create_api_key_endpoint(
    request: APIKeyCreateRequest,
    _: None = Depends(require_role(["admin"]))
):
    """Create a new API key (admin only)"""
    key_data = create_api_key(
        user_id=request.user_id,
        role=request.role,
        expires_in=request.expires_in
    )
    
    return APIKeyResponse(
        api_key=key_data["api_key"],
        key_id=key_data["key_id"],
        user_id=key_data["user_id"],
        role=key_data["role"],
        created_at=key_data["created_at"],
        expires_at=key_data["expires_at"]
    )

# Function to include routers in FastAPI app
def setup_auth_routes(app):
    """
    Add authentication and user management routes to the FastAPI app
    
    Args:
        app: The FastAPI app instance
        
    Usage:
        from app.auth.user_manager import setup_auth_routes
        
        app = FastAPI()
        setup_auth_routes(app)
    """
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(api_key_router)
    
    # Ensure background task queue is started
    @app.on_event("startup")
    async def startup_event():
        await task_queue.start()
        
    @app.on_event("shutdown")
    async def shutdown_event():
        await task_queue.stop()
        
    return app