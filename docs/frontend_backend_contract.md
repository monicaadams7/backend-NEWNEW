
# Frontend-Backend Contract: FitGPT API

This document describes the API contract between the FitGPT frontend and backend RAG system.

## Endpoint: POST /api/v1/chat

### Request Format

```json
{
  "messages": [
    { "role": "user", "text": "How much protein should I eat?" },
    { "role": "bot", "text": "It depends on your goals and body weight." },
    { "role": "user", "text": "I want to build muscle." }
  ],
  "conversation_id": "optional-conversation-id-string",
  "user_id": "optional-user-identifier"
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| messages | Array | Yes | Array of message objects with role and text |
| conversation_id | String | No | Unique identifier for the conversation |
| user_id | String | No | Identifier for the user (for personalization) |

### Message Object Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| role | String | Yes | Either "user" or "bot" |
| text | String | Yes | The message content |

### Response Format

```json
{
  "message": {
    "role": "bot",
    "text": "For muscle building, aim for 1.6-2.2g of protein per kg of body weight daily. This helps with muscle repair and growth when combined with resistance training."
  },
  "conversation_id": "generated-or-echoed-conversation-id"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| message | Object | Bot response message with role and text |
| conversation_id | String | Conversation ID (either provided or newly generated) |

### Error Handling

The API will return appropriate HTTP status codes:

- 200: Success
- 400: Bad request (missing/invalid parameters)
- 429: Rate limit exceeded
- 500: Server error

In case of errors, the response will include a message field explaining the issue.

## Conversation History

- The frontend should send the full relevant conversation history in each request
- The backend will store conversation history in SQLite/Redis
- History is used for context in the RAG system's response generation
- The backend limits history to the most recent messages (configurable)
- Message format must include both "role" and "text" fields

## Performance Considerations

- Requests may take 2-10 seconds to process
- The frontend should implement appropriate loading states
- For rapidly changing UIs, consider optimistic updates

## Cache Behavior

- Identical queries with the same recent conversation history may return cached responses
- The cache TTL is configurable (default: 1 hour)
- To force a fresh response, add a unique identifier to the query

## Health Checks

The `/api/v1/health` endpoint returns system status information:

```json
{
  "status": "ok",
  "version": "1.0.0" 
}
```
```

## 10. Integration with Your Existing FastAPI app

Finally, here's how you would integrate this with your existing FastAPI app in `main.py`:

```python
# In main.py, after importing process_query

# Import additional functions from our refactored modules
from app.services.rag.monitoring import rag_status
from app.services.rag.loader import rag_loader
from app.services.rag.config import rag_config

# Add an admin endpoint to check RAG system health
@app.get("/api/v1/admin/rag-health", tags=["Admin"])
@limiter.exempt
async def rag_health_check():
    """Check RAG system health (admin only)"""
    return {
        "health_metrics": rag_status.get_health_metrics(),
        "config": {k: v for k, v in rag_config.current.items() if not k.startswith("_")},
        "rag_version": rag_loader.get_version() or "default"
    }

# Add an admin endpoint to reload the RAG system
@app.post("/api/v1/admin/reload-rag", tags=["Admin"])
@limiter.exempt
async def reload_rag_system(version: Optional[str] = None):
    """Reload the RAG system (admin only)"""
    from app.services.rag_wrapper import reload_rag_system
    
    success = reload_rag_system(version)
    return {
        "success": success,
        "message": "RAG system reloaded successfully" if success else "Failed to reload RAG system",
        "current_version": rag_loader.get_version() or "default"
    }

# Add an admin endpoint to update RAG configuration
@app.post("/api/v1/admin/configure-rag", tags=["Admin"])
@limiter.exempt
async def configure_rag(config: Dict[str, Any]):
    """Update RAG system configuration (admin only)"""
    from app.services.rag_wrapper import configure_rag_system
    
    configure_rag_system(config)
    return {
        "success": True,
        "updated_config": {k: v for k, v in rag_config.current.items() if k in config}
    }
```

## Summary of Refactoring

This comprehensive refactoring achieves all your requested goals:

1. **Clarity and Separation of Concerns**
   - Each component has its own module with clear responsibilities
   - Graph loading is encapsulated in `loader.py`
   - Fallback handling is in `fallbacks.py`
   - Health monitoring is in `monitoring.py`
   - Proper interfaces and abstractions for each component

2. **Future Extensions Support**
   - Model configuration allows swapping LLMs without touching core code
   - Extensible state input with metadata dictionary
   - Versioning support for rag.py modules
   - Component-level performance tracking
   - User-specific and conversation-specific context

3. **Error Observability**
   - Trace IDs on every query operation
   - Structured logging with optional JSON format
   - Detailed timing information for each component
   - Comprehensive error recording and health metrics

4. **Performance Optimization**
   - LRU and Redis caching implementations
   - Separate tracking of retrieval vs. generation latency
   - Configurable timeouts and rate limits
   - Circuit breaker pattern for degraded performance

5. **Frontend-Backend Contract**
   - Clear documentation of message format
   - Validation of history format
   - Detailed API contract documentation
   - Consistent error handling and feedback

The refactored system is ready for multi-user expansion, robust under failure, and easy to extend with new features. It maintains compatibility with your existing FastAPI backend while adding powerful new capabilities.