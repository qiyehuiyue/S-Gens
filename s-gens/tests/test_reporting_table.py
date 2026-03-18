from __future__ import annotations

import unittest
from pathlib import Path

from sgens.reporting import collect_results_table


class MultiRunReportingTest(unittest.TestCase):
    def test_collect_results_table(self) -> None:
        run_root = Path(__file__).resolve().parent.parent / 'outputs' / 'hotpotqa_paper'
        if not run_root.exists():
            self.skipTest('hotpotqa_paper output not available')
        table = collect_results_table([run_root])
        self.assertEqual(len(table.rows), 1)
        self.assertEqual(table.rows[0].dataset, 'hotpotqa')


if __name__ == '__main__':
    unittest.main()
