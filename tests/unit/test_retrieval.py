import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.dense import DenseRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.embeddings.base import EmbeddingProvider

@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_text.return_value = [0.1] * 768
    return provider

@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=QdrantVectorStore)
    return store

def test_retrieve_success(mock_embedding_provider, mock_vector_store):
    """Test successful retrieval flow."""
    # Mock vector store to return some results
    # Note: we mock the output of vector_store.search which internally calls client.query_points
    mock_vector_store.search.return_value = [
        {
            "chunk_id": "c1",
            "content": "Clinical guideline text 1",
            "score": 0.95,
            "metadata": {"source_file": "doc1.pdf", "page_start": 1}
        },
        {
            "chunk_id": "c2",
            "content": "Clinical guideline text 2",
            "score": 0.88,
            "metadata": {"source_file": "doc2.pdf", "page_start": 5}
        }
    ]


    retriever = DenseRetriever(mock_embedding_provider, mock_vector_store)
    results = retriever.retrieve("What is the dosage of X?", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] == 0.95
    mock_embedding_provider.embed_text.assert_called_once_with("What is the dosage of X?")
    mock_vector_store.search.assert_called_once()

def test_retrieve_empty_results(mock_embedding_provider, mock_vector_store):
    """Test behavior when no results are found."""
    mock_vector_store.search.return_value = []

    retriever = DenseRetriever(mock_embedding_provider, mock_vector_store)
    results = retriever.retrieve("Unknown medical term", top_k=5)

    assert results == []
    mock_embedding_provider.embed_text.assert_called_once()

def test_retrieve_error_handling(mock_embedding_provider, mock_vector_store):
    """Test that retrieval errors return empty list and don't crash."""
    mock_vector_store.search.side_effect = Exception("Qdrant Connection Error")

    retriever = DenseRetriever(mock_embedding_provider, mock_vector_store)
    results = retriever.retrieve("Test query", top_k=5)

    assert results == []
    # Verify we didn't crash and returned empty list as per app/retrieval/dense.py
