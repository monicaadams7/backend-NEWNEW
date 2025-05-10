"""
Service tests
"""
import pytest
import os
import json
import tempfile
from app.services.storage import ConversationStorage, SQLiteStorage
from app.services.rag_wrapper import process_query

@pytest.fixture
def sqlite_storage():
    """Create a temporary SQLite storage for testing"""
    # Create a temporary file
    fd, path = tempfile.mkstemp()
    
    try:
        # Close the file descriptor
        os.close(fd)
        
        # Create the storage instance
        storage = SQLiteStorage(path)
        yield storage
    finally:
        # Remove the temporary file
        os.unlink(path)

@pytest.mark.asyncio
async def test_sqlite_storage_save_and_retrieve():
    """Test saving and retrieving conversations from SQLite storage"""
    # Create in-memory SQLite storage
    storage = SQLiteStorage(":memory:")
    
    # Test data
    conversation_id = "test-conversation-id"
    messages = [
        {"role": "user", "text": "Hello"},
        {"role": "bot", "text": "Hi there!"}
    ]
    
    # Save the conversation
    await storage.save_conversation(conversation_id, messages)
    
    # Retrieve the conversation
    retrieved = await storage.get_conversation(conversation_id)
    
    # Verify the retrieved data
    assert retrieved == messages
    
    # Test retrieving non-existent conversation
    empty = await storage.get_conversation("non-existent-id")
    assert empty == []

def test_process_query():
    """Test the process_query function"""
    # Test with a simple query
    query = "Hello, how are you?"
    result = process_query(query)
    
    # Verify response contains the query
    assert query in result
    
    # Test with history
    history = [
        {"role": "user", "text": "Previous message"},
        {"role": "bot", "text": "Previous response"}
    ]
    
    result_with_history = process_query(query, history)
    assert query in result_with_history
