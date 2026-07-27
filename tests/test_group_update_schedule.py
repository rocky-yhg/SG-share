from __future__ import annotations

import unittest
from unittest.mock import Mock

from ocap.config import load_config
from ocap.data import synthetic_stream
from ocap.framework import OCAP


class GroupUpdateScheduleTest(unittest.TestCase):
    def _boundary_calls(self, periodic_enabled: bool) -> list[tuple[int, bool]]:
        stream, feature_names = synthetic_stream(
            seed=7, n_users=2, triplets_per_user=4, n_features=4,
        )
        cfg = load_config(dataset="synthetic")
        cfg.seed = 7
        cfg.grouping.backbone_warmup_steps = 4
        cfg.grouping.group_update_interval = 2
        cfg.grouping.periodic_group_update_enabled = periodic_enabled
        ocap = OCAP(len(feature_names), cfg, "ocap", "gradient")
        calls: list[tuple[int, bool]] = []
        ocap._boundary = Mock(
            side_effect=lambda step, initial=False: calls.append((step, initial))
        )
        ocap.run(stream)
        return calls

    def test_update_groups_on_keeps_initial_and_periodic_boundaries(self) -> None:
        self.assertEqual(
            self._boundary_calls(True),
            [(4, True), (6, False), (8, False)],
        )

    def test_update_groups_off_freezes_after_initial_boundary(self) -> None:
        self.assertEqual(self._boundary_calls(False), [(4, True)])


if __name__ == "__main__":
    unittest.main()
