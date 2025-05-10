"""
RAG (Retrieval-Augmented Generation) wrapper for FitGPT
Production-ready service for processing user queries through the RAG system
"""
import logging
import time
import asyncio
import functools
import traceback
from concurrent.futures import TimeoutError
from typing import Dict, List, Any, Optional, Union

from .rag.loader import rag_loader
from .rag.monitoring import rag_status, structured_logger
from .rag.fallbacks import get_fallback_response, validate_history
from .rag.cache import default_cache
from .rag.config import rag_config

# Configure logger
logger = logging.getLogger(__name__)

# Timeout decorator for async functions
def async_timeout(seconds: int):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        return wrapper
    return decorator

@async_timeout(10)  # 10 second timeout for query processing
async def process_query(
    query: str, 
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> str:
    """
    Process a user query through the RAG system with production safeguards
    
    Args:
        query: The user's query text
        history: Optional conversation history, list of dicts with 'role' and 'text' keys
        user_id: Optional user identifier for personalization
        conversation_id: Optional conversation identifier for context
        
    Returns:
        The generated response text from the RAG system
    """
    start_time = time.time()
    success = False
    response = ""
    trace_id = structured_logger.get_trace_id()
    retrieval_time = 0
    generation_time = 0
    
    try:
        structured_logger.log(
            "info", 
            f"Processing query: '{query[:50]}...' (if longer)",
            trace_id=trace_id,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Check if system is available - circuit breaker pattern
        if not rag_status.is_healthy:
            structured_logger.log(
                "warning", 
                "RAG system marked unhealthy, using fallback",
                trace_id=trace_id
            )
            response = get_fallback_response(query, "system_down")
            return response
        
        # Get the graph
        graph = rag_loader.get_graph()
        
        # Check if graph is available
        if graph is None:
            structured_logger.log(
                "error", 
                "RAG graph not available, returning fallback response",
                trace_id=trace_id
            )
            response = get_fallback_response(query)
            return response
        
        # Validate history format
        if history is not None:
            is_valid, error_msg = validate_history(history)
            if not is_valid:
                structured_logger.log(
                    "warning", 
                    f"Invalid history format: {error_msg}",
                    trace_id=trace_id
                )
                # Continue with empty history rather than failing
                history = []
        
        # Check cache before processing
        if rag_config.get("enable_caching"):
            cache_key = default_cache.generate_key(query, history, user_id)
            cached_result = default_cache.get(cache_key)
            
            if cached_result:
                structured_logger.log(
                    "info", 
                    "Returning cached response",
                    trace_id=trace_id,
                    cache_hit=True
                )
                return cached_result.get("answer", "")
                
        # Format history for the graph - limit to most recent messages
        max_history = rag_config.get("max_history_length", 10)
        formatted_history = []
        if history:
            structured_logger.log(
                "info", 
                f"Processing with conversation history of {len(history)} messages",
                trace_id=trace_id
            )
            formatted_history = history[-max_history:] if len(history) > max_history else history
            
        # Initialize state for the graph - extensible format
        state = {
            "question": query,
            "context": [],
            "answer": "",
            "chat_history": formatted_history or [],
            # Add extensible fields for future features
            "metadata": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "timestamp": time.time()
            }
        }
        
        # Get model configuration
        model_config = rag_config.get_model_config()
        
        structured_logger.log(
            "info", 
            "Invoking RAG graph with initialized state",
            trace_id=trace_id,
            model=model_config.get("model")
        )
        
        # Track component time - retrieval phase
        retrieval_start = time.time()
        
        # Handle both synchronous and asynchronous graph.invoke
        if asyncio.iscoroutinefunction(getattr(graph, "invoke", None)):
            # If graph.invoke is async
            structured_logger.log("debug", "Using async graph.invoke", trace_id=trace_id)
            
            # First run vector retrieval if it's a separate step in the graph
            if hasattr(graph, "retrieval_executor") and asyncio.iscoroutinefunction(graph.retrieval_executor):
                try:
                    # Some LangGraph implementations separate retrieval and generation
                    retrieval_result = await graph.retrieval_executor.ainvoke(
                        {"question": query, "chat_history": formatted_history or []}
                    )
                    state["context"] = retrieval_result.get("context", [])
                    retrieval_time = time.time() - retrieval_start
                    
                    # Record retrieval time
                    rag_status.record_component_latency("retrieval", retrieval_time)
                    structured_logger.log(
                        "debug", 
                        f"Retrieval completed in {retrieval_time:.2f}s",
                        trace_id=trace_id,
                        document_count=len(state["context"])
                    )
                except Exception as e:
                    structured_logger.log(
                        "error", 
                        f"Error during retrieval phase: {str(e)}",
                        trace_id=trace_id
                    )
                    # Continue with empty context
            
            # Generation phase
            generation_start = time.time()
            result = await graph.invoke(
                state, 
                config={"configurable": {
                    "run_name": f"RAG-{query[:30]}",
                    **model_config  # Include model configuration
                }}
            )
            generation_time = time.time() - generation_start
        else:
            # If graph.invoke is synchronous
            structured_logger.log("debug", "Using synchronous graph.invoke", trace_id=trace_id)
            
            # Try to do retrieval separately if available
            if hasattr(graph, "retrieval_executor"):
                try:
                    retrieval_result = graph.retrieval_executor.invoke(
                        {"question": query, "chat_history": formatted_history or []}
                    )
                    state["context"] = retrieval_result.get("context", [])
                    retrieval_time = time.time() - retrieval_start
                    
                    # Record retrieval time
                    rag_status.record_component_latency("retrieval", retrieval_time)
                except Exception:
                    # Continue with empty context
                    pass
            
            # Generation phase
            generation_start = time.time()
            result = graph.invoke(
                state, 
                config={"configurable": {
                    "run_name": f"RAG-{query[:30]}",
                    **model_config
                }}
            )
            generation_time = time.time() - generation_start
        
        # Record generation time
        rag_status.record_component_latency("generation", generation_time)
        
        structured_logger.log(
            "info", 
            "RAG graph processing completed successfully", 
            trace_id=trace_id,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
        
        # Extract and return the answer
        answer = result.get("answer", "")
        if not answer:
            structured_logger.log("warning", "RAG graph returned empty answer", trace_id=trace_id)
            response = "I understand your fitness question, but I couldn't generate a proper response. Please try rephrasing or ask about a different fitness topic."
            return response
        
        # Store in cache
        if rag_config.get("enable_caching"):
            cache_key = default_cache.generate_key(query, history, user_id)
            default_cache.set(
                cache_key,
                {
                    "answer": answer,
                    "context": result.get("context", []),
                    "timestamp": time.time()
                },
                ttl=rag_config.get("cache_ttl_seconds", 3600)
            )
        
        response = answer
        structured_logger.log(
            "info", 
            f"Returning answer: '{answer[:50]}...' (if longer)",
            trace_id=trace_id
        )
        success = True
        return response
        
    except TimeoutError as e:
        structured_logger.log("error", f"Query processing timed out: {str(e)}", trace_id=trace_id)
        response = "I'm working on your fitness question, but it's taking longer than expected. Please try a simpler question or try again later."
        return response
        
    except Exception as e:
        rag_status.record_error(e)
        structured_logger.log(
            "error", 
            f"Error processing query through RAG system: {str(e)}", 
            trace_id=trace_id,
            error_trace=traceback.format_exc()
        )
        
        # If we've had too many failures, mark system as unhealthy
        if rag_status.failed_queries > 5 and rag_status.failed_queries / max(1, rag_status.total_queries) > 0.5:
            rag_status.is_healthy = False
            structured_logger.log(
                "error", 
                "Too many failures, marking RAG system as unhealthy", 
                trace_id=trace_id
            )
            
        response = get_fallback_response(query)
        return response
        
    finally:
        # Record performance metrics
        elapsed_time = time.time() - start_time
        rag_status.record_query(success, elapsed_time)
        
        # Log final performance stats
        structured_logger.log(
            "info", 
            f"Query processed in {elapsed_time:.2f}s (success={success})",
            trace_id=trace_id,
            total_time=elapsed_time,
            success=success
        )

def process_query_sync(
    query: str, 
    history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> str:
    """
    Synchronous version of process_query with error handling and metrics
    """
    try:
        # Try to create an event loop or get the existing one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            process_query(query, history, user_id, conversation_id)
        )
    except Exception as e:
        logger.error(f"Error in synchronous process_query wrapper: {str(e)}")
        return get_fallback_response(query)

def get_system_health() -> Dict[str, Any]:
    """
    Get system health metrics for monitoring dashboards and health checks
    Returns a dictionary of health metrics
    """
    return rag_status.get_health_metrics()

def reload_rag_system(version: str = None) -> bool:
    """
    Reload the RAG system, optionally with a specific version
    Useful for hot-reloading updated models without restarting the service
    
    Returns True if reload was successful, False otherwise
    """
    success, error = rag_loader.load_module(version)
    if success:
        rag_status.is_healthy = True
        rag_status.initialization_error = None
        logger.info(f"Successfully reloaded RAG system{' with version '+version if version else ''}")
        return True
    else:
        logger.error(f"Failed to reload RAG system: {error}")
        return False

def configure_rag_system(config: Dict[str, Any]) -> None:
    """
    Update RAG system configuration
    
    Args:
        config: Dictionary of configuration values to update
    """
    rag_config.update(config)
    logger.info(f"Updated RAG configuration with {len(config)} values")
    
    # Reinitialize cache if cache settings changed
    if any(k in config for k in ["cache_type", "redis_url", "lru_cache_size"]):
        global default_cache
        default_cache = create_cache(
            cache_type=rag_config.get("cache_type"),
            maxsize=rag_config.get("lru_cache_size"),
            redis_url=rag_config.get("redis_url")
        )
        logger.info(f"Reinitialized cache with type {rag_config.get('cache_type')}")

# For backward compatibility with the original module
if __name__ != "__main__":
    # Create an alias to the synchronous version for direct imports
    process_query_original = process_query
    process_query = process_query_sync