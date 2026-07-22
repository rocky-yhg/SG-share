from __future__ import annotations

import math
import unittest

import pandas as pd

from process.run_table3_experiments import _auc, _first_k_rows


class Table3MetricsTest(unittest.TestCase):
    def test_auc_handles_order_ties_and_single_class(self) -> None:
        self.assertEqual(_auc([0, 1], [0.1, 0.9]), 1.0)
        self.assertEqual(_auc([0, 1], [0.9, 0.1]), 0.0)
        self.assertEqual(_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertTrue(math.isnan(_auc([1, 1], [0.2, 0.8])))

    def test_first_k_metrics_are_averaged_across_complete_users(self) -> None:
        events = pd.DataFrame([
            {"event_index": 0, "user_id": "a", "label": 0, "raw_probability": 0.1},
            {"event_index": 1, "user_id": "b", "label": 1, "raw_probability": 0.8},
            {"event_index": 2, "user_id": "a", "label": 1, "raw_probability": 0.9},
            {"event_index": 3, "user_id": "b", "label": 1, "raw_probability": 0.7},
            {"event_index": 4, "user_id": "c", "label": 0, "raw_probability": 0.2},
        ])
        row = _first_k_rows(events, [2])[0]
        self.assertEqual(row["n_users"], 2)
        self.assertEqual(row["n_auc_users"], 1)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(row["precision"], 1.0)
        self.assertEqual(row["recall"], 1.0)
        self.assertEqual(row["macro_f1"], 1.0)
        self.assertEqual(row["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
