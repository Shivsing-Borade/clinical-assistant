import numpy as np
from typing import List, Dict, Any, Set

def recall_at_k(k: int, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculates Recall@k: (relevant retrieved items) / (total relevant items).
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = [cid for cid in top_k if cid in relevant_ids]
    return len(relevant_retrieved) / len(relevant_ids)

def mrr_at_k(k: int, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR)@k: 1 / rank of the first relevant item.
    """
    if not relevant_ids:
        return 0.0

    for rank, cid in enumerate(retrieved_ids[:k], 1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0

def calculate_metrics(retrieved_ids: List[str], relevant_ids: Set[str]) -> Dict[str, float]:
    """
    Returns a dictionary of retrieval metrics for a single query.
    """
    return {
        "recall_5": recall_at_k(5, retrieved_ids, relevant_ids),
        "recall_10": recall_at_k(10, retrieved_ids, relevant_ids),
        "mrr_10": mrr_at_k(10, retrieved_ids, relevant_ids),
    }

def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculates macro-averaged scores across all queries.
    """
    if not per_query_metrics:
        return {}

    keys = per_query_metrics[0].keys()
    averages = {}
    for key in keys:
        averages[key] = float(np.mean([m[key] for m in per_query_metrics]))

    return averages
