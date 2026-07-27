import unittest

from ocap.config import load_config
from ocap.data import synthetic_stream
from ocap.framework import OCAP


class SmokeTest(unittest.TestCase):
    def test_ocap_runs_prequentially(self):
        stream, feature_names = synthetic_stream(seed=3, n_users=6, triplets_per_user=8, n_features=6)
        cfg = load_config(dataset="synthetic")
        cfg.seed = 3
        ocap = OCAP(len(feature_names), cfg, "ocap", "gradient")
        result = ocap.run(stream)
        self.assertEqual(result["n_data_triplets"], len(stream))
        self.assertEqual(len(ocap.prediction_history), len(stream))
        self.assertIn("f1", result["overall"])


if __name__ == "__main__":
    unittest.main()
