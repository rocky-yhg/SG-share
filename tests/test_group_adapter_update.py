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


if __name__ == "__main__":
    unittest.main()
