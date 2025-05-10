"""
API endpoint tests
"""
import pytest
import uuid
from fastapi.testclient import TestClient

def test_root_endpoint(client):
    """Test the root endpoint returns correct metadata"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "FitGPT"
    assert data["status"] == "running"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data

def test_chat_endpoint_success(client, mock_process_query):
    """Test the chat endpoint with valid input"""
    request_data = {
        "messages": [
            {"role": "user", "text": "Hello, how are you?"}
        ]
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "conversation_id" in data
    assert data["message"]["role"] == "bot"
    assert "This is a mock response for:" in data["message"]["text"]

def test_chat_endpoint_no_messages(client):
    """Test the chat endpoint with no messages"""
    request_data = {
        "messages": []
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "No user message found" in data["detail"]

def test_chat_endpoint_invalid_role(client):
    """Test the chat endpoint with invalid role"""
    request_data = {
        "messages": [
            {"role": "invalid", "text": "This has an invalid role"}
        ]
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 422  # Validation error

def test_chat_endpoint_with_conversation_id(client, mock_process_query):
    """Test the chat endpoint with a specified conversation ID"""
    conversation_id = str(uuid.uuid4())
    request_data = {
        "messages": [
            {"role": "user", "text": "Hello with conversation ID"}
        ],
        "conversation_id": conversation_id
    }
    
    response = client.post("/api/v1/chat", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == conversation_id
