import pytest
from unittest.mock import MagicMock, patch
from app.config import settings
from app.llm.gemini_generator import GeminiGenerator
from app.rag.pipeline import ClinicalRAGPipeline

@pytest.fixture
def mock_genai_client():
    with patch('google.genai.Client') as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance
        yield mock_instance

def test_latency_measurement_structure():
    """Verify that the pipeline returns expected latency metrics."""
    with patch('app.rag.pipeline.HybridRetriever') as mock_retriever_class:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [{"chunk_id": "1", "content": "test", "metadata": {}}]
        mock_retriever_class.return_value = mock_retriever

        with patch('app.rag.pipeline.GeminiGenerator') as mock_gen_class:
            mock_gen = MagicMock()
            from app.schemas.generation import ClinicalAnswer
            mock_gen.generate_answer.return_value = ClinicalAnswer(
                answer="test", evidence_status="supported", cited_chunk_ids=["1"], clinician_review_required=False
            )
            mock_gen_class.return_value = mock_gen

            pipeline = ClinicalRAGPipeline()
            _, _, metrics, _ = pipeline.ask("test query")

            assert "retrieval_ms" in metrics
            assert "generation_ms" in metrics
            assert "total_ms" in metrics
            assert metrics["total_ms"] >= (metrics["retrieval_ms"] + metrics["generation_ms"])

def test_generation_context_limit():
    """Verify that the context selection obeys both the configured count limit and token budget."""
    from app.rag.context_manager import bound_evidence
    from app.config import settings

    # Mock chunks: each approx 11 tokens (40 chars)
    chunks = [{"chunk_id": f"c{i}", "content": "A" * 40} for i in range(10)]

    # Test Case 1: Limited by Count (TOP_K = 4, Tokens = 1000)
    with patch('app.rag.context_manager.settings.GENERATION_TOP_K_CONTEXT', 4):
        with patch('app.rag.context_manager.settings.GENERATION_EVIDENCE_MAX_TOKENS', 1000):
            result = bound_evidence(chunks, max_tokens=1000)
            assert len(result) == 4, f"Expected 4 chunks (bound by TOP_K), got {len(result)}"

    # Test Case 2: Limited by Tokens (TOP_K = 10, Tokens = 50)
    with patch('app.rag.context_manager.settings.GENERATION_TOP_K_CONTEXT', 10):
        with patch('app.rag.context_manager.settings.GENERATION_EVIDENCE_MAX_TOKENS', 50):
            # 50 tokens / 11 tokens per chunk ~= 4.5 chunks. Should be 4.
            result = bound_evidence(chunks, max_tokens=50)
            assert len(result) == 4, f"Expected 4 chunks (bound by tokens), got {len(result)}"

    # Test Case 3: No chunks should be truncated
    result = bound_evidence(chunks, max_tokens=100)
    for res_chunk in result:
        assert res_chunk in chunks


def test_model_config_propagation():
    """Verify that generator uses the correct default model for voice (flash-lite)."""
    # We check the settings object since that's where the default is defined
    assert settings.GEMINI_GENERATION_MODEL == "gemini-3.5-flash-lite" or \
           "gemini-3.5-flash-lite" in settings.GEMINI_GENERATION_MODEL
