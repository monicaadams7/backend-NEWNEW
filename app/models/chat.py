"""
Chat-related data models
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class Message(BaseModel):
    """Message model for chat conversations"""
    role: str = Field(..., description="Message role: 'user' or 'bot'")
    text: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate that role is either 'user' or 'bot'"""
        if v not in ['user', 'bot']:
            raise ValueError('Role must be either "user" or "bot"')
        return v

class ConversationRequest(BaseModel):
    """Request model for chat conversations"""
    messages: List[Message] = Field(..., description="List of conversation messages")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")

class ConversationResponse(BaseModel):
    """Response model for chat conversations"""
    message: Message = Field(..., description="Response message from the bot")
    conversation_id: str = Field(..., description="Conversation ID for tracking")
