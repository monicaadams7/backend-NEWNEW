"""
Input validation utilities for the FitGPT backend
"""
import re
import unicodedata
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def sanitize_string(text: str) -> str:
    """
    Sanitize a string by normalizing and removing control characters
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove control characters except whitespace
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch.isspace())
    
    return text

def is_valid_email(email: str) -> bool:
    """
    Validate email format
    """
    if not email:
        return False
    
    # Simple email validation pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Truncate at word boundary if possible
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    
    if last_space > 0.75 * max_length:
        return truncated[:last_space] + "..."
    
    return truncated + "..."

def validate_conversation_request(request: Dict[str, Any]) -> List[str]:
    """
    Validate a conversation request and return list of error messages
    Returns empty list if valid
    """
    errors = []
    
    if "messages" not in request:
        errors.append("Missing required field: messages")
        return errors
    
    if not isinstance(request["messages"], list):
        errors.append("Messages must be a list")
        return errors
    
    for i, message in enumerate(request["messages"]):
        if not isinstance(message, dict):
            errors.append(f"Message {i} must be an object")
            continue
        
        if "role" not in message:
            errors.append(f"Message {i} missing required field: role")
        elif message["role"] not in ["user", "bot"]:
            errors.append(f"Message {i} has invalid role: {message['role']}")
        
        if "text" not in message:
            errors.append(f"Message {i} missing required field: text")
        elif not isinstance(message["text"], str):
            errors.append(f"Message {i} text must be a string")
    
    return errors
