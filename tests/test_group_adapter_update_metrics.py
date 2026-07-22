from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from process.run_group_adapter_update_experiment import (
    _first_k_rows,
    _macro_f1,
    _macro_precision,
    _macro_recall,
)


class GroupAdapterUpdateMetricsTest(unittest.TestCase):
    def test_macro_f1_averages_both_classes(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        predictions = np.asarray([0, 0, 0, 1])
        self.assertAlmostEqual(_macro_f1(labels, predictions), (0.8 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(_macro_precision(labels, predictions), (2.0 / 3.0 + 1.0) / 2.0)
        self.assertAlmostEqual(_macro_recall(labels, predictions), (1.0 + 0.5) / 2.0)

    def test_first_k_uses_true_macro_f1(self) -> None:
        events = pd.DataFrame([
            {"event_index": 0, "user_id": "u", "label": 0, "raw_probability": 0.1},
            {"event_index": 1, "user_id": "u", "label": 1, "raw_probability": 0.1},
        ])
        row = _first_k_rows(events, [2])[0]
        self.assertAlmostEqual(row["macro_f1"], 1.0 / 3.0)
        self.assertAlmostEqual(row["precision"], 0.25)
        self.assertAlmostEqual(row["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
