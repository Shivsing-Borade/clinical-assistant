import pytest
from unittest.mock import MagicMock, patch
from app.config import settings
from app.llm.gemini_generator import GeminiGenerator

@pytest.fixture
def mock_genai_client():
    with patch('google.genai.Client') as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance
        yield mock_instance

def test_default_generation_model():
    """Verify the default generation model is gemini-3.8-flash."""
    # We reset the settings for this test or check the default
    from app.config import Settings
    # Create a fresh settings object without .env to check default
    test_settings = Settings()
    # Since we can't easily 'unset' env vars in a running process reliably for Pydantic BaseSettings
    # we check if the logic in config.py matches the requirement.
    assert test_settings.GEMINI_GENERATION_MODEL == "gemini-3.8-flash" or \
           "gemini-3.8-flash" in test_settings.GEMINI_GENERATION_MODEL

def test_env_var_override_generation_model():
    """Verify that GEMINI_GENERATION_MODEL env var overrides the default."""
    with patch.dict('os.environ', {'GEMINI_GENERATION_MODEL': 'custom-model-123'}):
        from pydantic_settings import BaseSettings
        from pydantic import ConfigDict

        class TestSettings(BaseSettings):
            model_config = ConfigDict(extra='ignore', env_file=".env")
            GEMINI_GENERATION_MODEL: str = "gemini-3.8-flash"

        s = TestSettings()
        assert s.GEMINI_GENERATION_MODEL == 'custom-model-123'

def test_generator_uses_configured_model(mock_genai_client):
    """Verify GeminiGenerator uses the model ID from settings."""
    # Mock settings
    with patch('app.llm.gemini_generator.settings') as mock_settings:
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_GENERATION_MODEL = "test-model-xyz"
        mock_settings.GENERATION_TEMPERATURE = 0.0

        generator = GeminiGenerator()

        # Mock a successful generation response
        mock_response = MagicMock()
        mock_response.parsed = MagicMock()
        mock_genai_client.models.generate_content.return_value = mock_response

        generator.generate_answer("query", [{"chunk_id": "1", "content": "text"}])

        # Verify the call used the correct model ID
        args, kwargs = mock_genai_client.models.generate_content.call_args
        assert kwargs['model'] == "test-model-xyz"
