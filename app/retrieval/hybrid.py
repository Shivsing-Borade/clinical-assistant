import logging
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from app.embeddings.base import EmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.bm25 import LocalBM25Retriever
from app.config import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retriever combining dense and local keyword search using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: QdrantVectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rrf_k = 60  # Standard RRF constant

        # Initialize local BM25 retriever
        self.keyword_retriever = self._init_bm25()

    def _init_bm25(self) -> LocalBM25Retriever:
        """Loads child chunks from disk and initializes local BM25 retriever."""
        processed_path = os.path.join(settings.DATA_PROCESSED_DIR, "children.json")
        try:
            with open(processed_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            logger.info(f"Initialized local BM25 with {len(corpus)} chunks.")
            return LocalBM25Retriever(corpus)
        except Exception as e:
            logger.error(f"Failed to initialize local BM25: {e}")
            # Fallback to an empty corpus if loading fails
            return LocalBM25Retriever([])

    def _rrf_merge(self, dense_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Merges two ranked lists using Reciprocal Rank Fusion.
        """
        scores = defaultdict(float)
        all_docs = {}

        # Dense rank fusion
        for rank, res in enumerate(dense_results, 1):
            chunk_id = res["chunk_id"]
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            all_docs[chunk_id] = res

        # Keyword rank fusion
        for rank, res in enumerate(keyword_results, 1):
            chunk_id = res["chunk_id"]
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            if chunk_id not in all_docs:
                all_docs[chunk_id] = res

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        # Return top_k documents with their RRF score instead of original similarity score
        final_results = []
        for cid in sorted_ids[:top_k]:
            doc = all_docs[cid].copy()
            doc["score"] = scores[cid] # RRF score
            final_results.append(doc)

        return final_results

    def retrieve(self, query: str, top_k: Optional[int] = None, candidate_limit: int = 50) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Dense search + Local BM25 search -> RRF Merge.
        """
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        try:
            # 1. Dense retrieval (via Qdrant)
            query_vector = self.embedding_provider.embed_text(query)
            dense_res = self.vector_store.search(query_vector, top_k=candidate_limit)

            # 2. Keyword retrieval (via local BM25)
            keyword_res = self.keyword_retriever.search(query, top_k=candidate_limit)

            # 3. Merge and deduplicate
            return self._rrf_merge(dense_res, keyword_res, top_k)

        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}")
            return []
