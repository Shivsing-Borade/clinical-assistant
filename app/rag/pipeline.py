import logging
import time
import json
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
from app.rag.context_manager import bound_context, bound_evidence

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
    context_diagnostics: Dict[str, Any] = Field(default_factory=dict)

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

    def ask(self, query: str, history: List[Dict[str, Any]] = None, standalone_query: Optional[str] = None) -> Tuple[ClinicalAnswer, List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
        """
        Executes the full RAG pipeline: Retrieve -> Generate -> Validate.
        """
        start_total = time.perf_counter()

        # Use standalone query for retrieval if provided (for follow-ups)
        retrieval_query = standalone_query if standalone_query else query

        # 1. Retrieval
        start_retrieval = time.perf_counter()
        results = self.retriever.retrieve(retrieval_query)

        # Bounded Evidence selection based on token budget
        gen_context = bound_evidence(results)

        retrieval_latency = (time.perf_counter() - start_retrieval) * 1000
        logger.info(
            "RAG retrieval: latency_ms=%.1f chunks_retrieved=%d chunks_used=%d",
            retrieval_latency, len(results), len(gen_context),
        )

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
            return ans, [], {"retrieval_ms": retrieval_latency, "generation_ms": 0.0, "total_ms": latency_total}, {}

        # 3. Generation with Bounded History
        start_gen = time.perf_counter()
        try:
            # We pass history separately; the generator handles its prompt construction
            answer_obj = self.generator.generate_answer(query, gen_context, history=history)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        generation_latency = (time.perf_counter() - start_gen) * 1000
        logger.info("RAG generation: latency_ms=%.1f", generation_latency)

        # 4. Citation Validation
        actual_ids = {res['chunk_id'] for res in results}
        cited_ids = set(answer_obj.cited_chunk_ids)

        if not cited_ids.issubset(actual_ids):
            answer_obj.clinician_review_required = True
            existing_caution = answer_obj.caution or ""
            val_warning = "Validation Warning: Model cited evidence not present in retrieved context."
            answer_obj.caution = f"{existing_caution} {val_warning}".strip()

        latency_total = (time.perf_counter() - start_total) * 1000
        logger.info(
            "RAG pipeline total: retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f",
            retrieval_latency, generation_latency, latency_total,
        )

        metrics = {
            "retrieval_ms": retrieval_latency,
            "generation_ms": generation_latency,
            "total_ms": latency_total
        }

        # Basic diagnostics for the bounded context
        diagnostics = {
            "retrieved_chunks": len(results),
            "used_chunks": len(gen_context),
        }

        return answer_obj, results, metrics, diagnostics

class ClinicalAssistantOrchestrator:
    """Handles intent routing and dispatches to RAG or immediate response."""

    def __init__(self, retriever_mode: str = "hybrid"):
        self.router = IntentRouter()
        self.rag_pipeline = ClinicalRAGPipeline(retriever_mode=retriever_mode)

    def process_query(self, query: str, history: List[Dict[str, Any]] = None) -> Tuple[OrchestratedResponse, List[Dict[str, Any]]]:
        """
        Routes the query and returns a consistent OrchestratedResponse.
        """
        start_total = time.perf_counter()

        # Bounded history for routing and generation
        bounded_history = bound_context(history) if history else []

        # 1. Routing (intent classification + standalone-question rewrite for follow-ups)
        start_routing = time.perf_counter()
        routing_res = self.router.route_query(query, history=bounded_history)
        routing_latency = (time.perf_counter() - start_routing) * 1000
        logger.info("Orchestrator routing: route=%s latency_ms=%.1f", routing_res.route, routing_latency)

        route = routing_res.route

        # 2. Dispatch
        if route == "clinical_rag":
            # Full RAG Pipeline
            try:
                # Pass history and optional standalone query for retrieval
                answer_obj, results, rag_metrics, diagnostics = self.rag_pipeline.ask(
                    query,
                    history=bounded_history,
                    standalone_query=routing_res.standalone_question
                )

                total_latency = (time.perf_counter() - start_total) * 1000

                metrics = {
                    "routing_ms": routing_latency,
                    "retrieval_ms": rag_metrics.get("retrieval_ms", 0.0),
                    "generation_ms": rag_metrics.get("generation_ms", 0.0),
                    "total_ms": total_latency
                }

                logger.info(
                    "Orchestrator RAG complete: routing_ms=%.1f retrieval_ms=%.1f "
                    "generation_ms=%.1f total_ms=%.1f retrieved=%d used=%d",
                    routing_latency,
                    rag_metrics.get("retrieval_ms", 0.0),
                    rag_metrics.get("generation_ms", 0.0),
                    total_latency,
                    diagnostics.get("retrieved_chunks", 0),
                    diagnostics.get("used_chunks", 0),
                )

                resp = OrchestratedResponse(
                    route=route,
                    answer=answer_obj.answer,
                    evidence_status=answer_obj.evidence_status,
                    cited_chunk_ids=answer_obj.cited_chunk_ids,
                    clinician_review_required=answer_obj.clinician_review_required,
                    caution=answer_obj.caution,
                    metrics=metrics,
                    context_diagnostics={
                        "history_turns": len(bounded_history) // 2,
                        **diagnostics
                    }
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
                metrics=metrics,
                context_diagnostics={"history_turns": len(bounded_history) // 2}
            )
            return resp, []
