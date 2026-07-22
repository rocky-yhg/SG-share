from __future__ import annotations

import unittest
from unittest.mock import Mock

from sgshare.config import load_config
from sgshare.data import synthetic_stream
from sgshare.learner import SGShareLearner


class RegroupScheduleTest(unittest.TestCase):
    def _boundary_calls(self, periodic_enabled: bool) -> list[tuple[int, bool]]:
        stream, feature_names = synthetic_stream(
            seed=7, n_users=2, events_per_user=4, n_features=4,
        )
        cfg = load_config(dataset="synthetic")
        cfg.seed = 7
        cfg.grouping.global_warmup_steps = 4
        cfg.grouping.regroup_interval = 2
        cfg.grouping.periodic_regroup_enabled = periodic_enabled
        learner = SGShareLearner(len(feature_names), cfg, "full_final", "gradient")
        calls: list[tuple[int, bool]] = []
        learner._boundary = Mock(
            side_effect=lambda step, initial=False: calls.append((step, initial))
        )
        learner.run(stream)
        return calls

    def test_regroup_on_keeps_initial_and_periodic_boundaries(self) -> None:
        self.assertEqual(
            self._boundary_calls(True),
            [(4, True), (6, False), (8, False)],
        )

    def test_regroup_off_freezes_after_initial_boundary(self) -> None:
        self.assertEqual(self._boundary_calls(False), [(4, True)])


if __name__ == "__main__":
    unittest.main()
