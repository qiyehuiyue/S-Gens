from __future__ import annotations

import unittest
from pathlib import Path

from sgens.evaluation import evaluate_retriever
from sgens.metrics import aggregate_metrics, reciprocal_rank, recall_at_k
from sgens.rag import build_rag_prompt
from sgens.retriever import RetrievalHit


class MetricsAndRagTest(unittest.TestCase):
    def test_metrics(self) -> None:
        self.assertEqual(reciprocal_rank(["d2", "d1"], {"d1"}, k=10), 0.5)
        self.assertEqual(recall_at_k(["d2", "d1"], {"d1"}, k=2), 1.0)
        metrics = aggregate_metrics([(["d2", "d1"], {"d1"}), (["d3"], {"d3"})]).to_dict()
        self.assertAlmostEqual(metrics["hit@1"], 0.5)

    def test_evaluate_lexical_retriever(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        metrics = evaluate_retriever("lexical", base / "demo_documents.json", base / "demo_original.json")
        self.assertIn("mrr@10", metrics)
        self.assertIn("recall@20", metrics)

    def test_build_rag_prompt(self) -> None:
        prompt = build_rag_prompt(
            "Who directed Inception?",
            [RetrievalHit(document_id="d1", text="Christopher Nolan directed Inception.", score=1.0)],
        )
        self.assertIn("Who directed Inception?", prompt)
        self.assertIn("Christopher Nolan directed Inception.", prompt)


if __name__ == "__main__":
    unittest.main()
