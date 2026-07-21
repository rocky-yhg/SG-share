import unittest

import numpy as np
import torch

from sgshare.baselines import PortedPerUserBaseline, ReplayBuffer
from sgshare.config import load_config
from sgshare.data import StreamEvent, synthetic_stream


class PaperBaselineTest(unittest.TestCase):
    def test_all_methods_are_strictly_prequential(self):
        stream, names = synthetic_stream(
            seed=9, n_users=2, events_per_user=3, n_features=4
        )
        for method in ("pdfk", "budgeted", "supermask"):
            cfg = load_config(dataset="synthetic")
            learner = PortedPerUserBaseline(len(names), cfg, method)
            first = stream[0]
            state = learner._state(first.user_id)
            before = {
                name: parameter.detach().clone()
                for name, parameter in state.model.named_parameters()
            }
            row = learner.process(first)
            self.assertEqual(row["user_event_index"], 1)
            self.assertEqual(len(learner.events), 1)
            self.assertTrue(any(
                not torch.equal(before[name], parameter)
                for name, parameter in state.model.named_parameters()
            ))

    def test_pdfk_uses_official_core_defaults(self):
        cfg = load_config(dataset="synthetic")
        learner = PortedPerUserBaseline(4, cfg, "pdfk")
        self.assertEqual(learner.paper_cfg.pdfk_kd_lambda, 5.5)
        self.assertEqual(learner.paper_cfg.pdfk_kd_temperature, 3.0)
        self.assertEqual(learner.paper_cfg.pdfk_ema_alpha, 0.01)
        self.assertEqual(learner.paper_cfg.pdfk_perturb_lambda, 0.005)

    def test_budgeted_frequency_sampling_updates_usage(self):
        buffer = ReplayBuffer(8, np.random.default_rng(4))
        for index in range(8):
            buffer.balanced_update(np.asarray([index], dtype=np.float32), index % 2)
        selected = buffer.similarity_aware_indices(
            4, np.eye(2), frequency_scale=4.0, temperature=0.125
        )
        self.assertEqual(len(selected), 4)
        self.assertAlmostEqual(sum(buffer.frequency), 4.0)

    def test_smoke_results_expose_audit_protocol(self):
        stream = [
            StreamEvent(i, "u0", str(i), i % 2, np.ones(4, dtype=np.float32) * i)
            for i in range(4)
        ]
        cfg = load_config(dataset="synthetic")
        learner = PortedPerUserBaseline(4, cfg, "budgeted")
        result = learner.run(stream)
        self.assertEqual(result["n_events"], 4)
        self.assertTrue(result["baseline_protocol"]["first_event_included"])
        self.assertEqual(
            result["provenance"], "paper_faithful_cross_domain_port"
        )


if __name__ == "__main__":
    unittest.main()
