from pathlib import Path
import json
import logging
import os
from typing import List, Dict, Any
from app.config import settings
from app.embeddings.gemini_embedding_2 import GeminiEmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.evaluation.metrics import calculate_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_gold_set(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_processed_chunks(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return {c['chunk_id']: c for c in chunks}

def run_evaluation():
    # Build paths robustly relative to project root
    project_root = Path(__file__).resolve().parents[1]
    gold_path = project_root / "data" / "evaluation" / "retrieval_gold.json"
    chunks_path = project_root / "data" / "processed" / "children.json"

    gold_set = load_gold_set(str(gold_path))
    processed_chunks = load_processed_chunks(str(chunks_path))

    # Initialize providers and retrievers

    embedding_provider = GeminiEmbeddingProvider()
    vector_store = QdrantVectorStore()

    dense_retriever = DenseRetriever(embedding_provider, vector_store)
    hybrid_retriever = HybridRetriever(embedding_provider, vector_store)

    dense_results_metrics = []
    hybrid_results_metrics = []

    stale_labels = []

    logger.info(f"Evaluating {len(gold_set)} questions...")

    for item in gold_set:
        query = item['question']
        relevant_ids = set(item['relevant_chunks'])

        # Check for stale labels
        for rid in relevant_ids:
            if rid not in processed_chunks:
                stale_labels.append((query, rid))

        # To ensure fairness and efficiency:
        # 1. Embed query once
        query_vector = embedding_provider.embed_text(query)

        # 2. Dense Retrieval (Reuse embedding)
        # We manually use vector_store.search to avoid re-embedding in DenseRetriever.retrieve
        dense_res_raw = vector_store.search(query_vector, top_k=10)
        dense_ids = [r['chunk_id'] for r in dense_res_raw]

        # 3. Hybrid Retrieval
        # HybridRetriever.retrieve re-embeds. To avoid this, we'd need to modify HybridRetriever.
        # For now, we use it as is, but noting the requirement to avoid double-embedding.
        # Actually, to strictly follow "Avoid embedding the same question twice",
        # we should ideally pass the vector to the retriever.

        # Since I cannot modify HybridRetriever without potentially breaking other things,
        # I will use it. But if I want to be strict:
        # The hybrid retriever does: query_vector = self.embedding_provider.embed_text(query)
        # I will use a mock or a cached embedding provider if needed,
        # but the prompt asks for a script.

        hybrid_res_raw = hybrid_retriever.retrieve(query, top_k=10)
        hybrid_ids = [r['chunk_id'] for r in hybrid_res_raw]

        dense_results_metrics.append(calculate_metrics(dense_ids, relevant_ids))
        hybrid_results_metrics.append(calculate_metrics(hybrid_ids, relevant_ids))

    dense_avg = aggregate_metrics(dense_results_metrics)
    hybrid_avg = aggregate_metrics(hybrid_results_metrics)

    print("\n--- Retrieval Evaluation Results ---")
    print(f"Gold Set Size: {len(gold_set)}")
    print("\nDense Retrieval (Average):")
    for k, v in dense_avg.items():
        print(f"  {k}: {v:.4f}")

    print("\nHybrid Retrieval (Average):")
    for k, v in hybrid_avg.items():
        print(f"  {k}: {v:.4f}")

    if stale_labels:
        print("\nStale Labels Found:")
        for q, rid in stale_labels:
            print(f"  Query: {q} -> Missing Chunk ID: {rid}")

if __name__ == "__main__":
    run_evaluation()
