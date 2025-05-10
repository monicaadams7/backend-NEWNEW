"""
Configuration management for the RAG system
Handles loading and validating configuration
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RAGConfig:
    """Configuration for the RAG system"""
    
    def __init__(self):
        # Default configuration
        self.defaults = {
            # Feature flags
            "enable_caching": True,
            "enable_monitoring": True,
            "enable_structured_logging": False,
            
            # Performance settings
            "query_timeout_seconds": 10,
            "max_history_length": 10,
            
            # Caching settings
            "cache_type": "lru",  # "lru" or "redis"
            "lru_cache_size": 100,
            "redis_url": os.getenv("REDIS_URL", None),
            "cache_ttl_seconds": 3600,  # 1 hour
            
            # Model settings
            "default_model": "gemini",  # "gemini", "openai", "claude", "mistral"
            "fallback_model": None,
            
            # Paths
            "versions_directory": os.path.join(os.path.dirname(__file__), "..", "versions")
        }
        
        # Current configuration (can be updated at runtime)
        self.current = self.defaults.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.current.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self.current[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values at once"""
        self.current.update(config_dict)
    
    def reset(self) -> None:
        """Reset to default configuration"""
        self.current = self.defaults.copy()

    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model-specific configuration
        Allows changing LLM without changing core code
        """
        model = model_name or self.get("default_model", "gemini")
        
        # Base configuration for all models
        base_config = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
            "timeout": self.get("query_timeout_seconds")
        }
        
        # Model-specific configurations
        model_configs = {
            "gemini": {
                "model": "gemini-1.0-pro",
                # Any Gemini-specific settings
            },
            "openai": {
                "model": "gpt-4",
                # Any OpenAI-specific settings
            },
            "claude": {
                "model": "claude-3-sonnet-20240229",
                # Any Claude-specific settings
            },
            "mistral": {
                "model": "mistral-large",
                # Any Mistral-specific settings
            }
        }
        
        # Merge base with model-specific config
        if model in model_configs:
            return {**base_config, **model_configs[model]}
        
        # If model not found, return base config
        logger.warning(f"Requested model '{model}' not found in configuration, using base settings")
        return base_config

# Create singleton instance
rag_config = RAGConfig()