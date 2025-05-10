"""
Fallback response handling for RAG system
Provides domain-specific responses when the primary system fails
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Generic fallback responses
GENERIC_FALLBACK = ("I'm having trouble processing your fitness question right now. "
                   "Our team is working to improve the AI trainer's knowledge system. "
                   "Please try again later or rephrase your question.")

TIMEOUT_FALLBACK = ("I'm working on your fitness question, but it's taking longer than expected. "
                   "Please try a simpler question or try again later.")

SYSTEM_DOWN_FALLBACK = ("Our fitness AI system is currently undergoing maintenance. "
                       "We apologize for the inconvenience and expect to be back shortly. "
                       "Basic fitness questions may still work.")

# Domain-specific fallback categories and responses
DOMAIN_FALLBACKS = {
    "diet": {
        "keywords": ["diet", "nutrition", "food", "eat", "meal", "calories", "protein", "carbs", "fat"],
        "response": ("I'd like to help with your nutrition question, but our diet recommendation system "
                    "is temporarily unavailable. Generally, focus on whole foods, adequate protein, "
                    "and proper hydration. Please try again later for personalized advice.")
    },
    "workout": {
        "keywords": ["workout", "exercise", "training", "lift", "cardio", "run", "strength", "gym", "fitness"],
        "response": ("I understand you're asking about fitness training, but our workout recommendation "
                    "system is temporarily unavailable. For best results, combine resistance training "
                    "with cardiovascular exercise and ensure proper recovery. Please try again later.")
    },
    "recovery": {
        "keywords": ["recovery", "rest", "sleep", "injury", "pain", "sore", "stretch", "rehab"],
        "response": ("I'd like to help with your recovery question, but our system is temporarily "
                    "unavailable. Proper rest, sleep, and nutrition are fundamental to recovery. "
                    "For injuries or persistent pain, please consult a healthcare professional.")
    },
    "goals": {
        "keywords": ["goal", "progress", "track", "weight", "lose", "gain", "build", "muscle", "fat"],
        "response": ("I understand you're asking about fitness goals, but our goal-tracking system "
                    "is temporarily unavailable. Setting specific, measurable, and realistic goals "
                    "is key to success. Please try again later for personalized guidance.")
    }
}

def get_fallback_response(query: str, error_type: Optional[str] = None) -> str:
    """
    Generate appropriate fallback response based on query content and error type
    
    Args:
        query: The user's original query
        error_type: Optional error type for more specific fallbacks
        
    Returns:
        A contextual fallback response
    """
    # Handle specific error types first
    if error_type == "timeout":
        return TIMEOUT_FALLBACK
    elif error_type == "system_down":
        return SYSTEM_DOWN_FALLBACK
    
    # Check for domain-specific fallbacks
    query_lower = query.lower()
    
    for domain, data in DOMAIN_FALLBACKS.items():
        if any(kw in query_lower for kw in data["keywords"]):
            logger.info(f"Using {domain} fallback response for query: {query[:50]}...")
            return data["response"]
    
    # Default generic fallback
    return GENERIC_FALLBACK

def validate_history(history: Any) -> tuple[bool, Optional[str]]:
    """
    Validate that conversation history is in the expected format
    
    Args:
        history: The history object to validate
        
    Returns:
        (is_valid, error_message) tuple
    """
    if history is None:
        return True, None
        
    if not isinstance(history, list):
        return False, "History must be a list"
        
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            return False, f"History item {i} is not a dictionary"
            
        if "role" not in msg:
            return False, f"History item {i} is missing 'role' field"
            
        if "text" not in msg:
            return False, f"History item {i} is missing 'text' field"
            
        if msg["role"] not in ["user", "bot"]:
            return False, f"History item {i} has invalid role: {msg['role']}"
    
    return True, None