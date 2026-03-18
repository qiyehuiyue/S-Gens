from __future__ import annotations

import json
import unittest
from pathlib import Path

from sgens.reporting import collect_experiment_result


class ReportingTest(unittest.TestCase):
    def test_collect_experiment_result(self) -> None:
        run_root = Path(__file__).resolve().parent.parent / 'outputs' / 'hotpotqa_paper'
        if not run_root.exists():
            self.skipTest('hotpotqa_paper output not available')
        result = collect_experiment_result(run_root)
        self.assertEqual(result.dataset, 'hotpotqa')
        self.assertEqual(result.preset, 'paper')
        self.assertGreaterEqual(result.num_pairs, 1)


if __name__ == '__main__':
    unittest.main()
