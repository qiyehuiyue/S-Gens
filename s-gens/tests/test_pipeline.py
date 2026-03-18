from __future__ import annotations

import unittest
from pathlib import Path

from sgens.pipeline import SgensConfig, SgensPipeline


class PipelineTest(unittest.TestCase):
    def test_demo_pipeline_generates_pairs_and_triplets(self) -> None:
        base = Path(__file__).resolve().parent.parent / "data"
        pipeline = SgensPipeline.from_json(base / "demo_kg.json", base / "demo_documents.json", config=SgensConfig())
        anchors = pipeline.load_original(base / "demo_original.json")
        pairs, triplets, mixed = pipeline.run(anchors)

        self.assertGreaterEqual(len(pairs), 1)
        self.assertGreaterEqual(len(triplets), 1)
        self.assertEqual(sum(1 for item in mixed if item["source"] == "original"), len(anchors))
        self.assertTrue(any(item["source"] == "synthetic" for item in mixed))


if __name__ == "__main__":
    unittest.main()
