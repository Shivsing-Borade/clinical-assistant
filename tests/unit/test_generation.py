import pytest
from unittest.mock import MagicMock, patch
from app.rag.pipeline import ClinicalRAGPipeline
from app.schemas.generation import ClinicalAnswer

@pytest.fixture
def mock_retriever():
    """Returns a mock retriever that we can configure per test."""
    with patch('app.rag.pipeline.HybridRetriever') as mock_hybrid, \
         patch('app.rag.pipeline.DenseRetriever') as mock_dense:

        mock_inst = MagicMock()
        mock_hybrid.return_value = mock_inst
        mock_dense.return_value = mock_inst
        yield mock_inst

@pytest.fixture
def mock_generator():
    """Returns a mock generator."""
    with patch('app.rag.pipeline.GeminiGenerator') as mock_gen_class:
        mock_inst = MagicMock()
        mock_gen_class.return_value = mock_inst
        yield mock_inst

def test_empty_retrieval_skips_llm(mock_retriever, mock_generator):
    """Verify that if no chunks are retrieved, the LLM is not called."""
    # Setup: Retriever returns empty list
    mock_retriever.retrieve.return_value = []

    pipeline = ClinicalRAGPipeline()
    answer, results, metrics = pipeline.ask("Any question")

    # Assertions
    assert answer.evidence_status == "insufficient_evidence"
    assert results == []
    mock_generator.generate_answer.assert_not_called()
    assert metrics['generation_ms'] == 0.0

def test_grounded_answer_success(mock_retriever, mock_generator):
    """Verify successful grounded answer generation and citation preservation."""
    # Setup: Retriever returns a chunk
    mock_chunks = [{"chunk_id": "chunk_1", "content": "Text 1", "metadata": {}}]
    mock_retriever.retrieve.return_value = mock_chunks

    # Setup: Generator returns a valid ClinicalAnswer
    mock_ans = ClinicalAnswer(
        answer="The answer is Text 1.",
        evidence_status="supported",
        cited_chunk_ids=["chunk_1"],
        clinician_review_required=False
    )
    mock_generator.generate_answer.return_value = mock_ans

    pipeline = ClinicalRAGPipeline()
    answer, results, metrics = pipeline.ask("Question")

    assert answer.answer == "The answer is Text 1."
    assert answer.evidence_status == "supported"
    assert "chunk_1" in answer.cited_chunk_ids
    assert answer.clinician_review_required is False

def test_invalid_citation_detection(mock_retriever, mock_generator):
    """Verify that citing a chunk NOT in the retrieved set triggers clinician review."""
    # Setup: Retriever returns chunk_1
    mock_chunks = [{"chunk_id": "chunk_1", "content": "Text 1", "metadata": {}}]
    mock_retriever.retrieve.return_value = mock_chunks

    # Setup: Generator hallucinates chunk_99
    mock_ans = ClinicalAnswer(
        answer="The answer is from nowhere.",
        evidence_status="supported",
        cited_chunk_ids=["chunk_99"],
        clinician_review_required=False
    )
    mock_generator.generate_answer.return_value = mock_ans

    pipeline = ClinicalRAGPipeline()
    answer, results, metrics = pipeline.ask("Question")

    # Assertions
    assert answer.clinician_review_required is True
    assert "Validation Warning" in answer.caution

def test_insufficient_evidence_from_llm(mock_retriever, mock_generator):
    """Verify handling when LLM determines evidence is insufficient."""
    mock_chunks = [{"chunk_id": "chunk_1", "content": "Irrelevant text", "metadata": {}}]
    mock_retriever.retrieve.return_value = mock_chunks

    mock_ans = ClinicalAnswer(
        answer="The available indexed guidelines do not establish the answer.",
        evidence_status="insufficient_evidence",
        cited_chunk_ids=[],
        clinician_review_required=False
    )
    mock_generator.generate_answer.return_value = mock_ans

    pipeline = ClinicalRAGPipeline()
    answer, results, metrics = pipeline.ask("Question")

    assert answer.evidence_status == "insufficient_evidence"
    assert answer.clinician_review_required is False
