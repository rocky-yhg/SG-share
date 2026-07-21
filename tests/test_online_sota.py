import unittest

import numpy as np

from sgshare.config import load_config
from sgshare.data import StreamEvent, synthetic_stream
from sgshare.online_sota import OnlineSOTALearner


class OnlineSOTATest(unittest.TestCase):
    def test_all_methods_are_finite_and_strictly_prequential(self):
        stream, names = synthetic_stream(seed=17, n_users=3, events_per_user=6, n_features=5)
        for method in ("oli2ds", "obal", "hbp", "koil", "olifl", "olfl"):
            cfg = load_config(dataset="synthetic")
            learner = OnlineSOTALearner(len(names), cfg, method, "global")
            row = learner.process(stream[0])
            self.assertEqual(row["user_event_index"], 1, method)
            self.assertTrue(np.isfinite(row["probability"]), method)
            self.assertEqual(len(learner.events), 1, method)

    def test_global_and_per_user_scopes_create_expected_states(self):
        stream = [StreamEvent(index, f"u{index % 2}", str(index), index % 2,
                              np.ones(4, dtype=np.float32)) for index in range(6)]
        cfg = load_config(dataset="synthetic")
        global_result = OnlineSOTALearner(4, cfg, "oli2ds", "global").run(stream)
        personal_result = OnlineSOTALearner(4, cfg, "oli2ds", "per_user").run(stream)
        self.assertEqual(global_result["n_models"], 1)
        self.assertEqual(personal_result["n_models"], 2)
        self.assertEqual(global_result["n_users"], 2)
        self.assertEqual(personal_result["n_users"], 2)

    def test_threshold_history_follows_model_scope(self):
        cfg = load_config(dataset="synthetic")
        first = StreamEvent(0, "u0", "0", 1, np.ones(4, dtype=np.float32))
        second = StreamEvent(1, "u1", "1", 0, np.ones(4, dtype=np.float32))

        personal = OnlineSOTALearner(4, cfg, "oli2ds", "per_user")
        personal.process(first)
        self.assertEqual(list(personal.threshold_history), ["u0"])
        self.assertEqual(len(personal.threshold_history["u0"]), 1)
        personal.process(second)
        self.assertEqual(set(personal.threshold_history), {"u0", "u1"})
        self.assertEqual(len(personal.threshold_history["u0"]), 1)
        self.assertEqual(len(personal.threshold_history["u1"]), 1)

        global_learner = OnlineSOTALearner(4, cfg, "oli2ds", "global")
        global_learner.process(first)
        global_learner.process(second)
        self.assertEqual(list(global_learner.threshold_history), ["__global__"])
        self.assertEqual(len(global_learner.threshold_history["__global__"]), 2)

    def test_results_include_first_k_positive_f1(self):
        stream, names = synthetic_stream(seed=5, n_users=4, events_per_user=12, n_features=5)
        cfg = load_config(dataset="synthetic")
        result = OnlineSOTALearner(len(names), cfg, "koil", "global").run(stream)
        self.assertEqual([int(row["k"]) for row in result["cold_start"]], [5, 10])
        self.assertIn("user_macro_f1", result["cold_start"][0])
        self.assertIn("pooled_f1", result["cold_start"][0])


if __name__ == "__main__":
    unittest.main()
