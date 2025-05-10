"""
Secured Endpoints for FitGPT Backend
Provides API-key authenticated versions of the main endpoints
"""
import logging
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Import security components
from app.security import get_api_key, require_role, sanitize_request_body

# Configure logger
logger = logging.getLogger(__name__)

# Create secured router
secured_router = APIRouter(prefix="/api/v1/secured", tags=["Secured"])

# Define models here instead of importing from main
class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'bot'")
    text: str = Field(..., description="Message content")

class ConversationRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")

class ConversationResponse(BaseModel):
    message: Message = Field(..., description="Response message from the bot")
    conversation_id: str = Field(..., description="Conversation ID for tracking")

# This function will be called from main.py to register these endpoints
def register_secured_endpoints(app, chat_function=None, storage=None, process_query_async=None):
    """
    Add secured endpoints to the FastAPI app
    
    Args:
        app: FastAPI app
        chat_function: The original chat function from main.py
        storage: The storage system from main.py
        process_query_async: The query processor from main.py
    """
    # Secured version of the chat endpoint
    @secured_router.post("/chat", response_model=ConversationResponse)
    @sanitize_request_body
    async def secured_chat(
        request: ConversationRequest, 
        key_info: Dict[str, Any] = Depends(get_api_key)
    ):
        """
        Secured version of chat endpoint that requires API key authentication
        """
        # Log the authenticated request
        user_id = key_info.get("user_id", "unknown")
        logger.info(f"Processing secured chat request from user {user_id}")
        
        if chat_function:
            # Use the provided chat function if available
            dummy_request = Request(scope={"type": "http"})
            return await chat_function(request=request, client_request=dummy_request)
        else:
            # Otherwise, replicate similar functionality
            # This is a simplified version and might not work exactly like the original
            try:
                # Get last user message
                user_messages = [msg for msg in request.messages if msg.role == "user"]
                if not user_messages:
                    return {"error": "No user message found"}
                
                user_message = user_messages[-1].text
                
                # Get conversation ID
                conversation_id = request.conversation_id or "new_conversation"
                
                # Process query
                answer = "This is a placeholder response from the secured endpoint."
                if process_query_async:
                    answer = await process_query_async(user_message)
                
                # Create response
                return ConversationResponse(
                    message=Message(role="bot", text=answer),
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.error(f"Error in secured chat: {str(e)}")
                return ConversationResponse(
                    message=Message(
                        role="bot", 
                        text="Sorry, I encountered an error processing your request."
                    ),
                    conversation_id=request.conversation_id or "error"
                )

    # Admin-only endpoint
    @secured_router.get("/admin/status")
    async def admin_status(
        _: None = Depends(require_role(["admin"]))
    ):
        """Admin-only endpoint for system status"""
        return {
            "status": "operational",
            "message": "This endpoint is only accessible to admins",
            "authenticated": True
        }

    # Add the router to the app
    app.include_router(secured_router)
    logger.info("Secured endpoints added to application")
    return app