"""
Request Sanitization for FitGPT Backend
Provides dependencies and functions for input validation and sanitization
"""

# For Pydantic v2 compatibility
try:
    # Pydantic v1
    from pydantic.fields import ModelField
except ImportError:
    # Pydantic v2
    from pydantic.fields import FieldInfo as ModelField

import re
import logging
import html
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Callable
from functools import wraps

from fastapi import Depends, FastAPI, Request, Response, HTTPException
from pydantic import BaseModel, Field, validator
from starlette import status

# Configure logger
logger = logging.getLogger(__name__)

# Generic type for Pydantic models
T = TypeVar('T', bound=BaseModel)

class SanitizedModel(BaseModel):
    """
    Base model with built-in sanitization for string fields
    Automatically escapes HTML special characters in string fields
    
    Usage:
        class UserInput(SanitizedModel):
            name: str
            description: Optional[str] = None
    """
    class Config:
        validate_assignment = True
        extra = "forbid"  # Forbid extra fields
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize a string value by escaping HTML"""
        if value is None:
            return None
            
        if not isinstance(value, str):
            return value
            
        # Escape HTML special characters
        return html.escape(value)

    # For Pydantic v1 compatibility
    try:
        @classmethod
        def get_validators(cls):
            """Return the default validators."""
            yield cls.validate
        
        @classmethod
        def validate(cls, v, field: Optional[ModelField] = None):
            """Custom validation for string fields"""
            if field and field.type_ == str and v is not None:
                return cls.sanitize_string(v)
            return v
    except:
        # For Pydantic v2 compatibility
        @classmethod
        def __get_validators__(cls):
            yield cls.validate
        
        # For Pydantic v2 model_validator
        @classmethod
        def model_validator(cls, model):
            """Apply sanitization to all string fields"""
            for field_name, field_value in model.__dict__.items():
                if isinstance(field_value, str):
                    setattr(model, field_name, cls.sanitize_string(field_value))
            return model

    def dict(self, *args, **kwargs):
        """Override to ensure sanitization is applied"""
        d = super().dict(*args, **kwargs)
        
        # Apply sanitization to all string values
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = self.sanitize_string(value)
                
        return d

# Regex patterns for validation
PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "username": re.compile(r"^[a-zA-Z0-9_-]{3,32}$"),
    "password": re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$"),
    "api_key": re.compile(r"^[a-zA-Z0-9_\-\.]{8,64}$"),
    "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
    "alphanumeric": re.compile(r"^[a-zA-Z0-9_\-\.]+$"),
    "numeric": re.compile(r"^[0-9]+$"),
}

def validate_pattern(value: str, pattern_name: str) -> bool:
    """Validate a string against a predefined pattern"""
    if pattern_name not in PATTERNS:
        logger.warning(f"Unknown pattern: {pattern_name}")
        return True
        
    pattern = PATTERNS[pattern_name]
    return bool(pattern.match(value))

# Custom validator functions that can be used in Pydantic models
def validate_username(value: str) -> str:
    """Validate username format"""
    if not validate_pattern(value, "username"):
        raise ValueError("Username must be 3-32 characters and only contain letters, numbers, underscore, and hyphen")
    return value

def validate_email(value: str) -> str:
    """Validate email format"""
    if value and not validate_pattern(value, "email"):
        raise ValueError("Invalid email format")
    return value

def validate_password(value: str) -> str:
    """Validate password strength"""
    if not validate_pattern(value, "password"):
        raise ValueError("Password must be at least 8 characters and include at least one letter and one number")
    return value

def validate_api_key(value: str) -> str:
    """Validate API key format"""
    if not validate_pattern(value, "api_key"):
        raise ValueError("Invalid API key format")
    return value

# Common input validators
class PaginationParams(BaseModel):
    """Common pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    
    class Config:
        extra = "forbid"  # Forbid extra fields

class SortParams(BaseModel):
    """Common sorting parameters"""
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc or desc)")
    
    @validator("sort_order")
    def validate_sort_order(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        return v
        
    class Config:
        extra = "forbid"  # Forbid extra fields

async def get_pagination(
    page: int = 1,
    limit: int = 20
) -> PaginationParams:
    """
    FastAPI dependency for pagination parameters
    
    Usage:
        @app.get("/items")
        async def get_items(pagination: PaginationParams = Depends(get_pagination)):
            # Use pagination.page and pagination.limit
            return {"page": pagination.page, "limit": pagination.limit}
    """
    # Validate and sanitize
    if page < 1:
        page = 1
    
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100
        
    return PaginationParams(page=page, limit=limit)

async def get_sorting(
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> SortParams:
    """
    FastAPI dependency for sorting parameters
    
    Usage:
        @app.get("/items")
        async def get_items(sorting: SortParams = Depends(get_sorting)):
            # Use sorting.sort_by and sorting.sort_order
            return {"sort_by": sorting.sort_by, "sort_order": sorting.sort_order}
    """
    # Sanitize sort_order
    if sort_order not in ["asc", "desc"]:
        sort_order = "desc"
        
    # Allow only certain fields for sorting to prevent SQL injection
    allowed_sort_fields = ["created_at", "updated_at", "username", "email", "role"]
    if sort_by not in allowed_sort_fields:
        sort_by = "created_at"
        
    return SortParams(sort_by=sort_by, sort_order=sort_order)

def sanitize_input(input_data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """
    Recursively sanitize input data
    Handles nested dictionaries and lists
    
    Args:
        input_data: Input data to sanitize
        
    Returns:
        Sanitized data of the same type
    """
    if isinstance(input_data, dict):
        return {k: sanitize_input(v) for k, v in input_data.items()}
    elif isinstance(input_data, list):
        return [sanitize_input(item) for item in input_data]
    elif isinstance(input_data, str):
        return html.escape(input_data)
    else:
        return input_data

# Decorator for automatic input sanitization
def sanitize_request_body(func: Callable):
    """
    Decorator to sanitize request body before processing
    
    Usage:
        @app.post("/items")
        @sanitize_request_body
        async def create_item(item: Dict):
            # item is sanitized
            return item
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Sanitize all kwargs that might be user input
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                sanitized_kwargs[key] = sanitize_input(value)
            elif isinstance(value, BaseModel):
                # If it's a Pydantic model, only sanitize it if it's not already a SanitizedModel
                if not isinstance(value, SanitizedModel):
                    model_dict = value.dict()
                    sanitized_dict = sanitize_input(model_dict)
                    model_class = value.__class__
                    sanitized_kwargs[key] = model_class(**sanitized_dict)
                else:
                    sanitized_kwargs[key] = value
            else:
                sanitized_kwargs[key] = value
                
        return await func(*args, **sanitized_kwargs)
        
    return wrapper

# Dependency for manual validation and sanitization
async def validate_and_sanitize(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency for manual validation and sanitization
    Useful when you need to sanitize raw JSON data
    
    Usage:
        @app.post("/custom-endpoint")
        async def custom_endpoint(data: Dict[str, Any] = Depends(validate_and_sanitize)):
            # data is validated and sanitized
            return data
    """
    try:
        # Parse JSON data
        json_data = await request.json()
        
        # Sanitize the data
        sanitized_data = sanitize_input(json_data)
        
        return sanitized_data
    except Exception as e:
        logger.error(f"Error parsing request body: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request body"
        )

# Function to create a sanitized model type
def create_sanitized_model(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """
    Convert a regular Pydantic model to a sanitized model
    Useful for existing models
    
    Usage:
        class User(BaseModel):
            name: str
            email: str
            
        SanitizedUser = create_sanitized_model(User)
    """
    # Create a new model that inherits from both the original and SanitizedModel
    class CombinedModel(model_class, SanitizedModel):
        pass
        
    # Update the model name
    CombinedModel.__name__ = f"Sanitized{model_class.__name__}"
    
    return CombinedModel