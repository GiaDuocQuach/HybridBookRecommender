from __future__ import annotations
import numpy as np

def _dcg(rels: np.ndarray, k: int) -> float:
    rels_k = np.asarray(rels[:k], dtype=float)
    if rels_k.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels_k.size + 2))
    return float(np.sum(rels_k * discounts))

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    rel_sorted = np.asarray(y_true, dtype=float)[order]
    dcg = _dcg(rel_sorted, k)
    ideal = _dcg(np.sort(np.asarray(y_true, dtype=float))[::-1], k)
    return float(dcg / (ideal + 1e-12))

def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    positive_threshold: float = 0.5
) -> float:

    order = np.argsort(-y_score)
    topk = np.asarray(y_true, dtype=float)[order][:k]
    total_pos = np.sum(np.asarray(y_true, dtype=float) >= positive_threshold)
    if total_pos == 0:
        return 0.0
    hits = np.sum(topk >= positive_threshold)
    return float(hits / total_pos)

def hit_rate_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    positive_threshold: float = 0.5
) -> float:

    order = np.argsort(-y_score)
    topk = np.asarray(y_true, dtype=float)[order][:k]
    return float(1.0 if np.any(topk >= positive_threshold) else 0.0)

def mrr_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
    positive_threshold: float = 0.5
) -> float:

    order = np.argsort(-y_score)
    topk = np.asarray(y_true, dtype=float)[order][:k]
    for i, r in enumerate(topk, start=1):
        if r >= positive_threshold:
            return 1.0 / i
    return 0.0
