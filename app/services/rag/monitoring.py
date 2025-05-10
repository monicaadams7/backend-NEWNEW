"""
RAG monitoring system - Tracks health and performance metrics
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)

class RAGSystemStatus:
    """Tracks the health and performance of the RAG system"""
    def __init__(self):
        self.is_healthy = False
        self.initialization_error = None
        self.last_error = None
        self.last_error_time = None
        self.total_queries = 0
        self.failed_queries = 0
        self.avg_response_time = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.latency_history = {
            "total": [],        # Total request time
            "retrieval": [],    # Vector retrieval time
            "generation": []    # LLM generation time
        }
        self.max_history_size = 100  # Keep only the last 100 entries
        
    def record_query(self, success: bool, response_time: float) -> None:
        """Record query performance metrics"""
        self.total_queries += 1
        if not success:
            self.failed_queries += 1
        
        # Update moving average of response time
        if self.total_queries == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * (self.total_queries - 1) + response_time) / self.total_queries
            
        # Add to latency history for total time
        self.latency_history["total"].append(response_time)
        if len(self.latency_history["total"]) > self.max_history_size:
            self.latency_history["total"].pop(0)
    
    def record_component_latency(self, component: str, latency: float) -> None:
        """Record latency for a specific component"""
        if component in self.latency_history:
            self.latency_history[component].append(latency)
            if len(self.latency_history[component]) > self.max_history_size:
                self.latency_history[component].pop(0)
    
    def record_error(self, error: Exception) -> None:
        """Record system error"""
        self.last_error = str(error)
        self.last_error_time = time.time()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Return health metrics for monitoring"""
        uptime = time.time() - self.start_time
        error_rate = self.failed_queries / max(1, self.total_queries)
        
        # Calculate component latency statistics
        latency_stats = {}
        for component, history in self.latency_history.items():
            if history:
                latency_stats[component] = {
                    "avg": sum(history) / len(history),
                    "min": min(history),
                    "max": max(history),
                    "p95": sorted(history)[int(len(history) * 0.95)] if len(history) >= 20 else None,
                    "samples": len(history)
                }
        
        return {
            "is_healthy": self.is_healthy,
            "uptime_seconds": uptime,
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "error_rate": error_rate,
            "avg_response_time": self.avg_response_time,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "initialization_error": self.initialization_error,
            "latency_stats": latency_stats
        }
        
    def check_and_update_health(self) -> bool:
        """Check if the system should be considered healthy based on metrics"""
        # System is unhealthy if error rate is too high (> 50% over last 5 queries)
        if self.failed_queries > 5 and self.failed_queries / max(1, self.total_queries) > 0.5:
            self.is_healthy = False
            logger.warning("System marked unhealthy due to high error rate")
        
        return self.is_healthy

class StructuredLogger:
    """Provides structured logging for RAG operations"""
    
    def __init__(self, base_logger=None):
        self.logger = base_logger or logger
        self.enable_json = False  # Set to True to output JSON logs
    
    def get_trace_id(self) -> str:
        """Generate a unique trace ID for request tracking"""
        return str(uuid.uuid4())
    
    def log(self, level: str, message: str, trace_id: Optional[str] = None, **kwargs) -> None:
        """Log a message with structured data"""
        log_data = {
            "message": message,
            "timestamp": time.time(),
            **kwargs
        }
        
        if trace_id:
            log_data["trace_id"] = trace_id
            
        if self.enable_json:
            # Output as JSON for structured logging systems
            log_str = json.dumps(log_data)
        else:
            # For human-readable logs, add trace_id to message if available
            if trace_id:
                message = f"[{trace_id}] {message}"
                
            log_str = message
            
        # Log at the appropriate level
        if level == "debug":
            self.logger.debug(log_str)
        elif level == "info":
            self.logger.info(log_str)
        elif level == "warning":
            self.logger.warning(log_str)
        elif level == "error":
            self.logger.error(log_str)
        elif level == "critical":
            self.logger.critical(log_str)

# Create singleton instances
rag_status = RAGSystemStatus()
structured_logger = StructuredLogger()