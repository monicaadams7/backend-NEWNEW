"""
Chat API endpoints
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List, Optional, Dict
import uuid
import logging

from app.models.chat import (
    Message, 
    ConversationRequest, 
    ConversationResponse
)
from app.services.storage import ConversationStorage
from app.services.rag_wrapper import process_query

# Create logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Conversation"])

# Placeholder for async version of process_query
async def process_query_async(query: str, history: List[Dict[str, str]] = None) -> str:
    """
    Async wrapper for process_query
    """
    return process_query(query, history)

# Define endpoints
@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest,
    client_request: Request,
    storage: ConversationStorage = Depends(lambda: None)  # Placeholder for dependency injection
):
    """
    Process a chat request and return a response
    """
    # Validate input
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
        # Placeholder for conversation history retrieval
        history = []  # This would come from storage in real implementation
        
        # Process the query using RAG system
        answer = await process_query_async(user_message, history)
        
        # Create new message
        bot_message = Message(role="bot", text=answer)
        
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
