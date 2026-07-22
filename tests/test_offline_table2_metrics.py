from __future__ import annotations

import unittest

import numpy as np

from process.run_offline_table2_experiments import _evaluate


class OfflineTable2MetricsTest(unittest.TestCase):
    def test_auc_uses_continuous_scores(self) -> None:
        labels = np.asarray([0, 0, 0, 1])
        predictions = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = _evaluate(labels, predictions, scores)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2.0 / 3.0) / 2.0)
        self.assertEqual(metrics["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
