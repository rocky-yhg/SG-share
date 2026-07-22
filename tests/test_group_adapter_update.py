from __future__ import annotations

import unittest

import numpy as np
import torch

from sgshare.config import load_config
from sgshare.data import StreamEvent
from sgshare.learner import SGShareLearner


def _state(learner: SGShareLearner, key: str) -> dict[str, torch.Tensor]:
    return learner.model.clone_adapter_state(key)


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


class GroupAdapterUpdateTest(unittest.TestCase):
    def _learner(self, mode: str) -> SGShareLearner:
        cfg = load_config(dataset="synthetic")
        cfg.grouping.global_warmup_steps = 1
        cfg.refinements.per_user_bias = False
        learner = SGShareLearner(3, cfg, group_adapter_mode=mode)
        learner.groups = {"g0": {"u0", "u1"}}
        learner.assignment = {"u0": "g0", "u1": "g0"}
        sources = {uid: learner._temp_key(uid) for uid in learner.groups["g0"]}
        learner._initialize_group_adapter("g0", learner.groups["g0"], sources, created=True)
        return learner

    @staticmethod
    def _event() -> StreamEvent:
        return StreamEvent(
            index=10,
            user_id="u0",
            timestamp="10",
            label=1,
            features=np.array([0.5, -1.0, 2.0], dtype=np.float32),
        )

    def test_standalone_mode_updates_only_group_adapter(self) -> None:
        learner = self._learner("standalone")
        user_before = {uid: _state(learner, f"temp::{uid}") for uid in ("u0", "u1")}
        group_before = _state(learner, "group::g0")

        learner._train_event(self._event(), raw_probability_before_update=0.5)

        for uid in ("u0", "u1"):
            self.assertFalse(_changed(user_before[uid], _state(learner, f"temp::{uid}")))
        self.assertTrue(_changed(group_before, _state(learner, "group::g0")))

    def test_user_mean_mode_updates_every_member_adapter(self) -> None:
        learner = self._learner("user_mean")
        user_before = {uid: _state(learner, f"temp::{uid}") for uid in ("u0", "u1")}
        group_before = _state(learner, "group::g0")

        learner._train_event(self._event(), raw_probability_before_update=0.5)

        for uid in ("u0", "u1"):
            self.assertTrue(_changed(user_before[uid], _state(learner, f"temp::{uid}")))
        self.assertFalse(_changed(group_before, _state(learner, "group::g0")))

    def test_shadow_personal_updates_group_and_only_event_owner(self) -> None:
        learner = self._learner("shadow_personal")
        user_before = {uid: _state(learner, f"temp::{uid}") for uid in ("u0", "u1")}
        group_before = _state(learner, "group::g0")

        learner._train_event(self._event(), raw_probability_before_update=0.5)

        self.assertTrue(_changed(user_before["u0"], _state(learner, "temp::u0")))
        self.assertFalse(_changed(user_before["u1"], _state(learner, "temp::u1")))
        self.assertTrue(_changed(group_before, _state(learner, "group::g0")))
        self.assertEqual(learner.shadow_update_count["u0"], 1)
        self.assertEqual(learner.shadow_update_count["u1"], 0)

    def test_shadow_loss_does_not_change_hard_group_update_or_route_signature(self) -> None:
        standalone = self._learner("standalone")
        shadow = self._learner("shadow_personal")

        standalone._train_event(self._event(), raw_probability_before_update=0.5)
        shadow._train_event(self._event(), raw_probability_before_update=0.5)

        for name, value in _state(standalone, "group::g0").items():
            self.assertTrue(torch.allclose(value, _state(shadow, "group::g0")[name]))
        self.assertTrue(torch.allclose(
            torch.as_tensor(standalone.route_signature["u0"]),
            torch.as_tensor(shadow.route_signature["u0"]),
        ))

    def test_shadow_group_refresh_only_runs_for_changed_membership(self) -> None:
        learner = self._learner("shadow_personal")
        learner.cfg.grouping.shadow_group_refresh_alpha = 0.1
        learner.personal_adapter_update_count["u0"] = 1
        learner.personal_adapter_update_count["u1"] = 1
        with torch.no_grad():
            learner.model.adapter("temp::u0").B.add_(0.5)
        sources = {uid: learner._temp_key(uid) for uid in ("u0", "u1")}

        unchanged_before = _state(learner, "group::g0")
        learner._initialize_group_adapter(
            "g0", {"u0", "u1"}, sources, created=False, membership_changed=False,
        )
        self.assertFalse(_changed(unchanged_before, _state(learner, "group::g0")))

        learner._initialize_group_adapter(
            "g0", {"u0", "u1"}, sources, created=False, membership_changed=True,
        )
        self.assertTrue(_changed(unchanged_before, _state(learner, "group::g0")))
        self.assertEqual(learner._last_shadow_diagnostics["refreshes"], 1)


if __name__ == "__main__":
    unittest.main()
