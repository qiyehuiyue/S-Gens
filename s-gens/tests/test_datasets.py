from __future__ import annotations

import unittest
from pathlib import Path

from sgens.datasets import prepare_dataset


class DatasetPreparationTest(unittest.TestCase):
    def test_prepare_webqsp(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        prepared = prepare_dataset("webqsp", base / "raw_samples" / "webqsp_sample.json", base / "demo_kg.json")
        self.assertEqual(prepared.name, "webqsp")
        self.assertEqual(len(prepared.anchors), 1)
        self.assertEqual(len(prepared.documents), 1)
        self.assertGreaterEqual(len(prepared.triples), 1)

    def test_prepare_hotpotqa(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        prepared = prepare_dataset("hotpotqa", base / "raw_samples" / "hotpotqa_sample.json", base / "demo_kg.json")
        self.assertEqual(prepared.anchors[0].positive_doc_id, "hotpotqa-0-0")
        self.assertGreaterEqual(len(prepared.documents), 2)

    def test_prepare_nq(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        prepared = prepare_dataset("nq", base / "raw_samples" / "nq_sample.json", base / "demo_kg.json")
        self.assertEqual(len(prepared.anchors), 1)
        self.assertEqual(prepared.anchors[0].positive_doc_id, "nq-0-0")

    def test_prepare_triviaqa(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        prepared = prepare_dataset("triviaqa", base / "raw_samples" / "triviaqa_sample.json", base / "demo_kg.json")
        self.assertEqual(len(prepared.anchors), 1)
        self.assertEqual(prepared.anchors[0].positive_doc_id, "triviaqa-0-0")


if __name__ == "__main__":
    unittest.main()
