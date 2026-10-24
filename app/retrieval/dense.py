import logging
from typing import List, Dict, Any, Optional
from app.embeddings.base import EmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.config import settings

logger = logging.getLogger(__name__)

class DenseRetriever:
    """Baseline retriever using only dense vector similarity."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: QdrantVectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k documents based on dense vector similarity.
        """
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        try:
            # 1. Embed the query
            query_vector = self.embedding_provider.embed_text(query)

            # 2. Search in vector store
            results = self.vector_store.search(query_vector, top_k=top_k)

            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
