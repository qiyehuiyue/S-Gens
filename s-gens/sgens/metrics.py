from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankingMetrics:
    mrr_at_10: float
    recall_at_20: float
    hit_at_1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mrr@10": self.mrr_at_10,
            "recall@20": self.recall_at_20,
            "hit@1": self.hit_at_1,
        }


def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 20) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def hit_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 1) -> float:
    return 1.0 if any(doc_id in relevant_doc_ids for doc_id in ranked_doc_ids[:k]) else 0.0


def aggregate_metrics(results: list[tuple[list[str], set[str]]]) -> RankingMetrics:
    if not results:
        return RankingMetrics(mrr_at_10=0.0, recall_at_20=0.0, hit_at_1=0.0)
    mrr = sum(reciprocal_rank(ranked, relevant, k=10) for ranked, relevant in results) / len(results)
    recall = sum(recall_at_k(ranked, relevant, k=20) for ranked, relevant in results) / len(results)
    hit1 = sum(hit_at_k(ranked, relevant, k=1) for ranked, relevant in results) / len(results)
    return RankingMetrics(mrr_at_10=mrr, recall_at_20=recall, hit_at_1=hit1)
