import pytest
from app.evaluation.metrics import recall_at_k, mrr_at_k, calculate_metrics, aggregate_metrics

def test_recall_at_k():
    relevant = {"c1", "c2", "c3"}

    # All relevant retrieved
    assert recall_at_k(5, ["c1", "c2", "c3", "c4", "c5"], relevant) == 1.0
    # Partial relevant retrieved
    assert recall_at_k(5, ["c1", "c4", "c5", "c6", "c7"], relevant) == 1/3
    # None relevant retrieved
    assert recall_at_k(5, ["c4", "c5", "c6", "c7", "c8"], relevant) == 0.0
    # No relevant IDs in gold set
    assert recall_at_k(5, ["c1", "c2"], set()) == 0.0
    # Empty retrieval
    assert recall_at_k(5, [], relevant) == 0.0

def test_mrr_at_k():
    relevant = {"c1", "c2"}

    # First item relevant
    assert mrr_at_k(10, ["c1", "c3", "c4"], relevant) == 1.0
    # Second item relevant
    assert mrr_at_k(10, ["c3", "c1", "c4"], relevant) == 0.5
    # Third item relevant
    assert mrr_at_k(10, ["c3", "c4", "c1"], relevant) == 1/3
    # None relevant in top K
    assert mrr_at_k(10, ["c3", "c4", "c5"], relevant) == 0.0
    # No relevant IDs in gold set
    assert mrr_at_k(10, ["c1", "c2"], set()) == 0.0
    # Empty retrieval
    assert mrr_at_k(10, [], relevant) == 0.0

def test_calculate_metrics():
    relevant = {"c1", "c2"}
    retrieved = ["c1", "c3", "c4", "c5", "c6", "c2", "c7", "c8", "c9", "c10"]

    metrics = calculate_metrics(retrieved, relevant)
    # Recall@5: only c1 retrieved. 1/2 = 0.5
    assert metrics["recall_5"] == 0.5
    # Recall@10: c1 and c2 retrieved. 2/2 = 1.0
    assert metrics["recall_10"] == 1.0
    # MRR@10: first relevant at rank 1. 1/1 = 1.0
    assert metrics["mrr_10"] == 1.0

def test_aggregate_metrics():
    metrics_list = [
        {"recall_5": 0.5, "recall_10": 1.0, "mrr_10": 1.0},
        {"recall_5": 0.0, "recall_10": 0.5, "mrr_10": 0.5},
    ]

    aggregated = aggregate_metrics(metrics_list)
    assert aggregated["recall_5"] == 0.25
    assert aggregated["recall_10"] == 0.75
    assert aggregated["mrr_10"] == 0.75

def test_aggregate_empty():
    assert aggregate_metrics([]) == {}
