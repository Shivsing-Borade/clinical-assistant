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
            _, _, metrics = pipeline.ask("test query")

            assert "retrieval_ms" in metrics
            assert "generation_ms" in metrics
            assert "total_ms" in metrics
            assert metrics["total_ms"] >= (metrics["retrieval_ms"] + metrics["generation_ms"])

def test_generation_context_limit():
    """Verify that only the configured top K chunks are sent to the generator."""
    with patch('app.rag.pipeline.HybridRetriever') as mock_retriever_class:
        mock_retriever = MagicMock()
        # Provide 10 results, but only 4 should be used
        mock_retriever.retrieve.return_value = [{"chunk_id": f"c{i}", "content": f"t{i}"} for i in range(10)]
        mock_retriever_class.return_value = mock_retriever

        with patch('app.rag.pipeline.GeminiGenerator') as mock_gen_class:
            mock_gen = MagicMock()
            from app.schemas.generation import ClinicalAnswer
            mock_gen.generate_answer.return_value = ClinicalAnswer(
                answer="test", evidence_status="supported", cited_chunk_ids=["c0"], clinician_review_required=False
            )
            mock_gen_class.return_value = mock_gen

            pipeline = ClinicalRAGPipeline()
            pipeline.ask("test query")

            # Check how many chunks were passed to generate_answer
            args, kwargs = mock_gen.generate_answer.call_args
            context_passed = args[1]
            assert len(context_passed) == settings.GENERATION_TOP_K_CONTEXT

def test_model_config_propagation():
    """Verify that generator uses the correct default model for voice (flash-lite)."""
    # We check the settings object since that's where the default is defined
    assert settings.GEMINI_GENERATION_MODEL == "gemini-3.5-flash-lite" or \
           "gemini-3.5-flash-lite" in settings.GEMINI_GENERATION_MODEL
