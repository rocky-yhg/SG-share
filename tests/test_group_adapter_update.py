from __future__ import annotations

import unittest

import numpy as np
import torch

from ocap.config import load_config
from ocap.data import DataTriplet
from ocap.framework import OCAP


def _state(ocap: OCAP, key: str) -> dict[str, torch.Tensor]:
    return ocap.model.clone_adapter_state(key)


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


class GroupAdapterUpdateTest(unittest.TestCase):
    def _ocap(self, mode: str) -> OCAP:
        cfg = load_config(dataset="synthetic")
        cfg.grouping.backbone_warmup_steps = 1
        cfg.refinements.per_user_bias = False
        ocap = OCAP(3, cfg, group_adapter_construction=mode)
        ocap.group_set = {"g0": {"u0", "u1"}}
        ocap.group_assignment = {"u0": "g0", "u1": "g0"}
        sources = {
            uid: ocap._personal_adapter_key(uid) for uid in ocap.group_set["g0"]
        }
        ocap._initialize_group_adapter(
            "g0", ocap.group_set["g0"], sources, created=True,
        )
        return ocap

    @staticmethod
    def _event() -> DataTriplet:
        return DataTriplet(
            index=10,
            user_id="u0",
            timestamp="10",
            label=1,
            features=np.array([0.5, -1.0, 2.0], dtype=np.float32),
        )

    def test_merged_trainable_mode_updates_only_group_adapter(self) -> None:
        ocap = self._ocap("merged_trainable")
        user_before = {uid: _state(ocap, f"personal::{uid}") for uid in ("u0", "u1")}
        group_before = _state(ocap, "group::g0")

        ocap._learn_from_triplet(self._event(), raw_probability_before_update=0.5)

        for uid in ("u0", "u1"):
            self.assertFalse(_changed(user_before[uid], _state(ocap, f"personal::{uid}")))
        self.assertTrue(_changed(group_before, _state(ocap, "group::g0")))

    def test_personal_adapter_mean_mode_updates_every_member_adapter(self) -> None:
        ocap = self._ocap("personal_adapter_mean")
        user_before = {uid: _state(ocap, f"personal::{uid}") for uid in ("u0", "u1")}
        group_before = _state(ocap, "group::g0")

        ocap._learn_from_triplet(self._event(), raw_probability_before_update=0.5)

        for uid in ("u0", "u1"):
            self.assertTrue(_changed(user_before[uid], _state(ocap, f"personal::{uid}")))
        self.assertFalse(_changed(group_before, _state(ocap, "group::g0")))

    def test_shadow_personal_adapter_updates_group_and_only_event_owner(self) -> None:
        ocap = self._ocap("shadow_personal_adapter")
        user_before = {uid: _state(ocap, f"personal::{uid}") for uid in ("u0", "u1")}
        group_before = _state(ocap, "group::g0")

        ocap._learn_from_triplet(self._event(), raw_probability_before_update=0.5)

        self.assertTrue(_changed(user_before["u0"], _state(ocap, "personal::u0")))
        self.assertFalse(_changed(user_before["u1"], _state(ocap, "personal::u1")))
        self.assertTrue(_changed(group_before, _state(ocap, "group::g0")))
        self.assertEqual(ocap.shadow_update_count["u0"], 1)
        self.assertEqual(ocap.shadow_update_count["u1"], 0)

    def test_shadow_loss_does_not_change_hard_group_update_or_gradient_history(self) -> None:
        merged_trainable = self._ocap("merged_trainable")
        shadow = self._ocap("shadow_personal_adapter")

        merged_trainable._learn_from_triplet(self._event(), raw_probability_before_update=0.5)
        shadow._learn_from_triplet(self._event(), raw_probability_before_update=0.5)

        for name, value in _state(merged_trainable, "group::g0").items():
            self.assertTrue(torch.allclose(value, _state(shadow, "group::g0")[name]))
        self.assertTrue(torch.allclose(
            torch.as_tensor(merged_trainable.gradient_history["u0"]),
            torch.as_tensor(shadow.gradient_history["u0"]),
        ))

    def test_shadow_group_refresh_only_runs_for_changed_membership(self) -> None:
        ocap = self._ocap("shadow_personal_adapter")
        ocap.cfg.grouping.shadow_group_refresh_alpha = 0.1
        ocap.personal_adapter_update_count["u0"] = 1
        ocap.personal_adapter_update_count["u1"] = 1
        with torch.no_grad():
            ocap.model.adapter("personal::u0").B.add_(0.5)
        sources = {uid: ocap._personal_adapter_key(uid) for uid in ("u0", "u1")}

        unchanged_before = _state(ocap, "group::g0")
        ocap._initialize_group_adapter(
            "g0", {"u0", "u1"}, sources, created=False, membership_changed=False,
        )
        self.assertFalse(_changed(unchanged_before, _state(ocap, "group::g0")))

        ocap._initialize_group_adapter(
            "g0", {"u0", "u1"}, sources, created=False, membership_changed=True,
        )
        self.assertTrue(_changed(unchanged_before, _state(ocap, "group::g0")))
        self.assertEqual(ocap._last_shadow_diagnostics["refreshes"], 1)


if __name__ == "__main__":
    unittest.main()
