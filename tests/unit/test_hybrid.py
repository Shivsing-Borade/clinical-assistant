import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.embeddings.base import EmbeddingProvider
from app.retrieval.bm25 import LocalBM25Retriever

@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed_text.return_value = [0.1] * 768
    return provider

@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=QdrantVectorStore)
    return store

@pytest.fixture
def mock_corpus():
    return [
        {"chunk_id": "c1", "content": "The quick brown fox", "metadata": {"src": "doc1"}},
        {"chunk_id": "c2", "content": "Jumped over the lazy dog", "metadata": {"src": "doc2"}},
        {"chunk_id": "c3", "content": "Medical guidelines for hypertension", "metadata": {"src": "doc3"}},
    ]

def test_bm25_ranking(mock_corpus):
    """Test that LocalBM25Retriever ranks based on term frequency."""
    retriever = LocalBM25Retriever(mock_corpus)
    results = retriever.search("hypertension guidelines", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c3"
    assert "metadata" in results[0]

def test_bm25_empty_query(mock_corpus):
    """Test BM25 with empty or whitespace query."""
    retriever = LocalBM25Retriever(mock_corpus)
    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []

def test_hybrid_rrf_merging(mock_embedding_provider, mock_vector_store, mock_corpus):
    """Test RRF merging and deduplication of local BM25 and dense results."""
    with patch("app.retrieval.hybrid.HybridRetriever._init_bm25") as mock_init:
        # Mock the internal BM25 retriever
        bm25 = LocalBM25Retriever(mock_corpus)
        mock_init.return_value = bm25

        # Mock dense results (c1 is top)
        mock_vector_store.search.return_value = [
            {"chunk_id": "c1", "content": "The quick brown fox", "score": 0.9, "metadata": {"src": "doc1"}},
            {"chunk_id": "c2", "content": "Jumped over the lazy dog", "score": 0.8, "metadata": {"src": "doc2"}},
        ]

        # HybridRetriever's BM25 will return c3 for "hypertension"
        retriever = HybridRetriever(mock_embedding_provider, mock_vector_store)
        results = retriever.retrieve("hypertension", top_k=3)

        assert len(results) == 3
        chunk_ids = [r["chunk_id"] for r in results]
        assert "c3" in chunk_ids # From BM25
        assert "c1" in chunk_ids # From Dense
        assert "c2" in chunk_ids # From Dense

def test_dense_only_backward_compatibility(mock_embedding_provider, mock_vector_store):
    """Ensure Step 5A DenseRetriever still works independently."""
    mock_vector_store.search.return_value = [{"chunk_id": "c1", "content": "T1", "score": 0.9, "metadata": {}}]

    retriever = DenseRetriever(mock_embedding_provider, mock_vector_store)
    results = retriever.retrieve("test", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"

def test_hybrid_empty_results(mock_embedding_provider, mock_vector_store, mock_corpus):
    """Test hybrid retrieval when neither source finds anything."""
    with patch("app.retrieval.hybrid.HybridRetriever._init_bm25") as mock_init:
        # Use empty corpus for BM25
        mock_init.return_value = LocalBM25Retriever([])
        mock_vector_store.search.return_value = []

        retriever = HybridRetriever(mock_embedding_provider, mock_vector_store)
        results = retriever.retrieve("nothing", top_k=5)
        assert results == []
