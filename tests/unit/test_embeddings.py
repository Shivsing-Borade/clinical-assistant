import pytest
from unittest.mock import MagicMock, patch
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingProvider
from app.config import settings

@pytest.fixture
def mock_genai_client():
    with patch("google.genai.Client") as mock:
        yield mock

def test_embedding_dimension_validation(mock_genai_client):
    """Test that vectors with wrong dimensions raise a ValueError."""
    # Setup mock
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    # Return a vector of wrong size (e.g., 128 instead of 768)
    mock_response.embeddings = [MagicMock(values=[0.1] * 128)]
    mock_instance.models.embed_content.return_value = mock_response

    provider = GeminiEmbeddingProvider()

    with pytest.raises(ValueError, match="Expected vector dimension 768"):
        provider.embed_text("test text")

def test_empty_text_handling():
    """Test that empty text raises a ValueError instead of returning a zero vector."""
    with patch("app.config.settings.GEMINI_API_KEY", "fake-key"):
        provider = GeminiEmbeddingProvider()
        with pytest.raises(ValueError, match="Empty or whitespace-only text cannot be embedded"):
            provider.embed_text("   ")

def test_api_error_handling(mock_genai_client):
    """Test that API errors are raised correctly."""
    mock_instance = mock_genai_client.return_value
    mock_instance.models.embed_content.side_effect = Exception("API Failure")

    provider = GeminiEmbeddingProvider()
    with pytest.raises(Exception, match="API Failure"):
        provider.embed_text("test text")
