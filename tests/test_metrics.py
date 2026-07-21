import unittest

from sgshare.metrics import binary_metrics, first_k_metrics


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        values = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(values["recall"], 0.5)
        self.assertAlmostEqual(values["specificity"], 0.5)
        self.assertAlmostEqual(values["balanced_accuracy"], 0.5)

    def test_first_k_is_user_macro(self):
        events = [
            {"user_id": "a", "label": 1, "prediction": 1},
            {"user_id": "b", "label": 1, "prediction": 0},
            {"user_id": "a", "label": 0, "prediction": 0},
        ]
        row = first_k_metrics(events, [1])[0]
        self.assertAlmostEqual(row["user_macro_f1"], 0.5)
        self.assertEqual(row["n_events"], 2)

    def test_first_k_excludes_users_without_k_events(self):
        events = [
            {"user_id": "a", "label": 1, "prediction": 1},
            {"user_id": "a", "label": 0, "prediction": 0},
            {"user_id": "b", "label": 1, "prediction": 0},
        ]
        row = first_k_metrics(events, [2])[0]
        self.assertEqual(row["n_users"], 1)
        self.assertEqual(row["n_events"], 2)


if __name__ == "__main__":
    unittest.main()
