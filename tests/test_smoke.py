import unittest

from sgshare.config import load_config
from sgshare.data import synthetic_stream
from sgshare.learner import SGShareLearner


class SmokeTest(unittest.TestCase):
    def test_full_final_runs_prequentially(self):
        stream, feature_names = synthetic_stream(seed=3, n_users=6, events_per_user=8, n_features=6)
        cfg = load_config(dataset="synthetic")
        cfg.seed = 3
        learner = SGShareLearner(len(feature_names), cfg, "full_final", "gradient")
        result = learner.run(stream)
        self.assertEqual(result["n_events"], len(stream))
        self.assertEqual(len(learner.events), len(stream))
        self.assertIn("f1", result["overall"])


if __name__ == "__main__":
    unittest.main()
