import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api import app, state

client = TestClient(app)

@pytest.fixture
def mock_session():
    session = MagicMock()
    session.device_label = "CPU"
    session.chat_history = []
    session.search_options = {}
    
    # Mock generate return value
    result = MagicMock()
    result.text = "Test response"
    result.stats = MagicMock()
    result.stats.prompt_length = 10
    result.stats.new_tokens = 5
    result.interrupted = False
    session.generate.return_value = result
    
    return session

def test_create_chat_completion_no_session():
    # Ensure state.session is None
    state.session = None
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 503

def test_create_chat_completion_success(mock_session):
    state.session = mock_session
    
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Test response"
    
    # Verify session calls
    mock_session.reset_history.assert_called_once()
    mock_session.generate.assert_called()

def test_create_chat_completion_with_history(mock_session):
    state.session = mock_session
    mock_session.chat_history = []
    
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"}
    ]
    
    response = client.post("/v1/chat/completions", json={
        "messages": messages
    })
    
    assert response.status_code == 200
    # Check history was populated (excluding last message)
    assert len(mock_session.chat_history) == 3
    assert mock_session.chat_history[0]["role"] == "system"
    assert mock_session.chat_history[2]["role"] == "assistant"
    
    # Check generate called with last message content
    # Note: generate is called with positional arg user_text
    args, _ = mock_session.generate.call_args
    assert args[0] == "Q2"

def test_create_chat_completion_params(mock_session):
    state.session = mock_session
    mock_session.search_options = {"max_length": 100, "temperature": 0.5}
    
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
        "temperature": 0.9
    })
    
    assert response.status_code == 200
    # Verify options were temporarily updated
    # Since generate is called synchronously in the endpoint (wrapped in to_thread), 
    # we can't easily check the state *during* the call unless we use a side_effect.
    # But we can check that they are restored after.
    assert mock_session.search_options["max_length"] == 100
    assert mock_session.search_options["temperature"] == 0.5

