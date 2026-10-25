import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.dense import DenseRetriever
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.llm.gemini_generator import GeminiGenerator
from app.llm.router import IntentRouter, RoutingResponse
from app.schemas.generation import ClinicalAnswer

logger = logging.getLogger(__name__)

class OrchestratedResponse(BaseModel):
    """Consistent response object for all routes."""
    route: str
    answer: str
    evidence_status: Optional[str] = None
    cited_chunk_ids: List[str] = Field(default_factory=list)
    clinician_review_required: bool = False
    caution: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)

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

        # Optimization: Limit context to top K highest-ranked unique chunks for generation
        gen_context = results[:settings.GENERATION_TOP_K_CONTEXT]

        retrieval_latency = (time.perf_counter() - start_retrieval) * 1000

        # 2. Short-circuit if no evidence
        if not results:
            latency_total = (time.perf_counter() - start_total) * 1000
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
            answer_obj = self.generator.generate_answer(query, gen_context)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        generation_latency = (time.perf_counter() - start_gen) * 1000

        # 4. Citation Validation
        actual_ids = {res['chunk_id'] for res in results}
        cited_ids = set(answer_obj.cited_chunk_ids)

        if not cited_ids.issubset(actual_ids):
            answer_obj.clinician_review_required = True
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

class ClinicalAssistantOrchestrator:
    """Handles intent routing and dispatches to RAG or immediate response."""

    def __init__(self, retriever_mode: str = "hybrid"):
        self.router = IntentRouter()
        self.rag_pipeline = ClinicalRAGPipeline(retriever_mode=retriever_mode)

    def process_query(self, query: str) -> Tuple[OrchestratedResponse, List[Dict[str, Any]]]:
        """
        Routes the query and returns a consistent OrchestratedResponse.
        """
        start_total = time.perf_counter()

        # 1. Routing
        start_routing = time.perf_counter()
        routing_res = self.router.route_query(query)
        routing_latency = (time.perf_counter() - start_routing) * 1000

        route = routing_res.route

        # 2. Dispatch
        if route == "clinical_rag":
            # Full RAG Pipeline
            try:
                answer_obj, results, rag_metrics = self.rag_pipeline.ask(query)

                total_latency = (time.perf_counter() - start_total) * 1000

                metrics = {
                    "routing_ms": routing_latency,
                    "retrieval_ms": rag_metrics.get("retrieval_ms", 0.0),
                    "generation_ms": rag_metrics.get("generation_ms", 0.0),
                    "total_ms": total_latency
                }

                resp = OrchestratedResponse(
                    route=route,
                    answer=answer_obj.answer,
                    evidence_status=answer_obj.evidence_status,
                    cited_chunk_ids=answer_obj.cited_chunk_ids,
                    clinician_review_required=answer_obj.clinician_review_required,
                    caution=answer_obj.caution,
                    metrics=metrics
                )
                return resp, results

            except Exception as e:
                logger.error(f"RAG pipeline failed for query: {query}. Error: {e}")
                raise
        else:
            # Immediate response for non-RAG routes
            total_latency = (time.perf_counter() - start_total) * 1000

            metrics = {
                "routing_ms": routing_latency,
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": total_latency
            }

            resp = OrchestratedResponse(
                route=route,
                answer=routing_res.response or "I'm sorry, I couldn't generate a response for this request.",
                metrics=metrics
            )
            return resp, []
