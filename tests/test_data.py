import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ocap.data import OnlineStandardizer, load_stream, synthetic_stream


class DataTest(unittest.TestCase):
    def test_online_scaler_updates_after_transform(self):
        scaler = OnlineStandardizer(1)
        first = scaler.transform_then_update(np.array([2.0]))
        second = scaler.transform_then_update(np.array([4.0]))
        self.assertAlmostEqual(float(first[0]), 2.0)
        self.assertAlmostEqual(float(second[0]), 2.0)

    def test_synthetic_stream_is_ordered(self):
        data_triplets, names = synthetic_stream(seed=7, n_users=4, triplets_per_user=3, n_features=5)
        self.assertEqual(len(data_triplets), 12)
        self.assertEqual(len(names), 5)
        self.assertEqual([triplet.index for triplet in data_triplets], list(range(12)))

    def test_no_scaling_preserves_preprocessed_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.csv"
            pd.DataFrame({
                "user_id": ["u1", "u1"],
                "timestamp": ["2020-01-01", "2020-01-02"],
                "label": [0, 1],
                "x": [3.5, -2.0],
            }).to_csv(path, index=False)
            data_triplets, names = load_stream(path, no_scaling=True)
            self.assertEqual(names, ["x"])
            self.assertAlmostEqual(float(data_triplets[0].features[0]), 3.5)
            self.assertAlmostEqual(float(data_triplets[1].features[0]), -2.0)


if __name__ == "__main__":
    unittest.main()
