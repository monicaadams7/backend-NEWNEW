"""
Storage service for FitGPT conversations
Provides persistent storage using SQLite or Redis
"""
import json
import os
import sqlite3
import logging
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class ConversationStorage:
    """
    Abstract base class for conversation storage
    Implementations should provide methods to save and retrieve conversations
    """
    
    @staticmethod
    def create_sqlite_storage(db_path: str) -> 'SQLiteStorage':
        """Factory method to create SQLite storage"""
        return SQLiteStorage(db_path)
    
    @staticmethod
    def create_redis_storage(redis_url: str) -> 'RedisStorage':
        """Factory method to create Redis storage"""
        return RedisStorage(redis_url)
    
    async def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history by ID"""
        raise NotImplementedError("Storage implementation must override get_conversation")
    
    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        """Save conversation history"""
        raise NotImplementedError("Storage implementation must override save_conversation")


class SQLiteStorage(ConversationStorage):
    """SQLite-based persistent storage for conversations"""
    
    def __init__(self, db_path: str):
        """Initialize SQLite storage with specified database path"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    messages TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Add index on updated_at for efficient cleanup of old conversations
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at 
                ON conversations (updated_at)
                """)
                
                conn.commit()
                logger.info(f"SQLite database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT messages FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                return []
        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding conversation {conversation_id}: {e}")
            return []
    
    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, messages, updated_at) 
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(conversation_id) 
                    DO UPDATE SET 
                        messages = excluded.messages,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (conversation_id, json.dumps(messages))
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving conversation {conversation_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving conversation {conversation_id}: {e}")
            raise


class RedisStorage(ConversationStorage):
    """Redis-based persistent storage for conversations"""
    
    def __init__(self, redis_url: str):
        """Initialize Redis storage with specified URL"""
        self.redis_url = redis_url
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            # Import redis here to make it an optional dependency
            import redis.asyncio as redis
            
            self.redis = redis.from_url(self.redis_url)
            logger.info(f"Redis connection initialized at {self.redis_url}")
        except ImportError:
            logger.error("Redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history from Redis"""
        try:
            # Key for storing the conversation
            key = f"conversation:{conversation_id}"
            
            # Get data from Redis
            data = await self.redis.get(key)
            
            if data:
                return json.loads(data)
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding conversation {conversation_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return []
    
    async def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to Redis"""
        try:
            # Key for storing the conversation
            key = f"conversation:{conversation_id}"
            
            # Set expiration time (30 days in seconds)
            expiration = 60 * 60 * 24 * 30
            
            # Save data to Redis with expiration
            await self.redis.set(
                key, 
                json.dumps(messages),
                ex=expiration
            )
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {e}")
            raise
