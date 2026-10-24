import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.dense import DenseRetriever
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.llm.gemini_generator import GeminiGenerator
from app.schemas.generation import ClinicalAnswer

logger = logging.getLogger(__name__)

class ClinicalRAGPipeline:
    """Orchestrates the retrieval-to-generation pipeline for clinical answers."""

    def __init__(self,
                 retriever_mode: str = "hybrid",
                 embedding_provider: Optional[Any] = None,
                 vector_store: Optional[Any] = None):
        self.generator = GeminiGenerator()

        # Setup retriever
        if embedding_provider is None:
            embedding_provider = GeminiEmbeddingProvider()
        if vector_store is None:
            vector_store = QdrantVectorStore()

        if retriever_mode == "hybrid":
            self.retriever = HybridRetriever(embedding_provider, vector_store)
        else:
            self.retriever = DenseRetriever(embedding_provider, vector_store)

    def ask(self, query: str) -> Tuple[ClinicalAnswer, List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes the full RAG pipeline: Retrieve -> Generate -> Validate.
        """
        start_total = time.perf_counter()

        # 1. Retrieval
        start_retrieval = time.perf_counter()
        results = self.retriever.retrieve(query)
        retrieval_latency = (time.perf_counter() - start_retrieval) * 1000

        # 2. Short-circuit if no evidence
        if not results:
            latency_total = (time.perf_counter() - start_total) * 1000
            # Create an insufficient evidence response without calling LLM
            ans = ClinicalAnswer(
                answer="The available indexed guidelines do not establish the answer.",
                evidence_status="insufficient_evidence",
                cited_chunk_ids=[],
                clinician_review_required=False,
                caution="No relevant evidence found in the indexed documents."
            )
            return ans, [], {"retrieval_ms": retrieval_latency, "generation_ms": 0.0, "total_ms": latency_total}

        # 3. Generation
        start_gen = time.perf_counter()
        try:
            answer_obj = self.generator.generate_answer(query, results)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        generation_latency = (time.perf_counter() - start_gen) * 1000

        # 4. Citation Validation
        actual_ids = {res['chunk_id'] for res in results}
        cited_ids = set(answer_obj.cited_chunk_ids)

        if not cited_ids.issubset(actual_ids):
            answer_obj.clinician_review_required = True
            # Update caution to include the validation warning
            existing_caution = answer_obj.caution or ""
            val_warning = "Validation Warning: Model cited evidence not present in retrieved context."
            answer_obj.caution = f"{existing_caution} {val_warning}".strip()

        latency_total = (time.perf_counter() - start_total) * 1000

        metrics = {
            "retrieval_ms": retrieval_latency,
            "generation_ms": generation_latency,
            "total_ms": latency_total
        }

        return answer_obj, results, metrics
