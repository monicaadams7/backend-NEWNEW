"""
PyTest configuration and fixtures
"""
import os
import pytest
import logging
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Disable logging during tests
logging.getLogger().setLevel(logging.WARNING)

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["STORAGE_TYPE"] = "sqlite"
os.environ["SQLITE_PATH"] = ":memory:"  # Use in-memory SQLite for tests

# Import the FastAPI app
from app.main import app

@pytest.fixture
def client():
    """
    Test client fixture for FastAPI app
    """
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def mock_process_query(monkeypatch):
    """
    Mock the process_query function
    """
    def _mock_process_query(query, history=None):
        return f"This is a mock response for: {query}"
    
    # Import here to avoid circular imports
    import app.services.rag_wrapper
    
    # Apply the monkeypatch
    monkeypatch.setattr(app.services.rag_wrapper, "process_query", _mock_process_query)
