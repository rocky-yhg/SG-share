import unittest
from collections import deque

import numpy as np
import torch

from ocap.config import load_config
from ocap.data import DataTriplet
from ocap.framework import RecentDataPair, OCAP
from ocap.model import OCAPModel, weighted_focal_loss


class OCAPFrameworkTest(unittest.TestCase):
    def test_ces_and_globem_historical_configs(self):
        ces = load_config(dataset="ces")
        globem = load_config(dataset="globem")
        self.assertEqual((ces.model.hidden_dim, ces.model.depth, ces.model.num_heads), (32, 2, 4))
        self.assertEqual((globem.model.hidden_dim, globem.model.depth, globem.model.num_heads), (16, 1, 2))
        self.assertEqual(ces.model.lora_rank, 4)
        self.assertEqual(ces.model.lora_alpha, 8.0)
        self.assertEqual(ces.grouping.minimum_group_count, 4)
        self.assertEqual(ces.grouping.warmup_initial_min_samples, 5)
        self.assertTrue(ces.grouping.warmup_initial_force_accept)
        self.assertEqual(ces.grouping.warmup_initial_target_groups, 0)
        self.assertEqual(ces.grouping.warmup_initial_min_groups, 2)
        self.assertEqual(ces.grouping.warmup_initial_max_groups, 12)
        self.assertEqual(ces.refinements.per_user_bias_lr, 0.005)
        self.assertEqual(ces.refinements.split_min_users, 10)
        self.assertEqual(ces.refinements.split_min_events, 50)
        self.assertEqual(ces.refinements.reassignment_every, 10)
        self.assertEqual(ces.grouping.cfl_coherence_threshold, 0.55)
        self.assertEqual(ces.grouping.cfl_disagreement_cosine_threshold, 0.20)
        self.assertEqual(ces.refinements.split_cohesion_tolerance, 0.05)

    def test_gradient_projection_and_lora_parameterization(self):
        cfg = load_config(dataset="ces")
        model = OCAPModel(37, cfg.model)
        adapter = model.ensure_adapter("test")
        self.assertEqual(tuple(model.backbone.gradient_projection.shape), (32, 32))
        self.assertEqual(tuple(adapter.A.shape), (4, 32))
        self.assertEqual(tuple(adapter.B.shape), (32, 4))
        self.assertEqual(adapter.scale, 2.0)

    def test_positive_class_weight_is_active(self):
        negative = weighted_focal_loss(torch.zeros(1, 2), torch.tensor([0]), 0.75, 2.0, 5.0)
        positive = weighted_focal_loss(torch.zeros(1, 2), torch.tensor([1]), 0.75, 2.0, 5.0)
        self.assertGreater(float(positive), float(negative) * 10.0)

    def test_optimizer_types_match_historical_implementation(self):
        cfg = load_config(dataset="synthetic")
        ocap = OCAP(6, cfg)
        self.assertIsInstance(ocap.backbone_optimizer, torch.optim.AdamW)
        self.assertIsInstance(ocap.adapter_optimizers["global"], torch.optim.Adam)
        self.assertEqual(ocap.backbone_optimizer.param_groups[0]["weight_decay"], 1e-4)
        self.assertEqual(ocap.adapter_optimizers["global"].param_groups[0]["weight_decay"], 0.0)

    def test_bias_is_absent_during_warmup_and_uses_group_mean_afterward(self):
        cfg = load_config(dataset="synthetic")
        ocap = OCAP(6, cfg)
        ocap.group_set = {"g0": {"u0", "u1"}}
        ocap.group_assignment = {"u0": "g0", "u1": "g0"}
        ocap.user_bias["u1"] = 1.0
        logits = torch.zeros(1, 2)
        warmup_logits = ocap._apply_bias(logits, "u0", warmup=True)
        self.assertTrue(torch.equal(warmup_logits, logits))
        self.assertNotIn("u0", ocap.user_bias)
        post_logits = ocap._apply_bias(logits, "u0", warmup=False)
        self.assertAlmostEqual(ocap.user_bias["u0"], 0.5)
        self.assertAlmostEqual(float(post_logits[0, 1]), 0.5)

    def test_verified_split_requires_cfl_disagreement(self):
        cfg = load_config(dataset="synthetic")
        cfg.refinements.verified_split = True
        cfg.refinements.split_min_users = 2
        ocap = OCAP(6, cfg)
        users = {f"u{i}" for i in range(4)}
        representations = {uid: np.ones(4, dtype=np.float64) for uid in users}
        clusters, accepted = ocap._verified_splits(
            [users], representations, {uid: f"personal::{uid}" for uid in users}, step=10,
        )
        self.assertEqual(clusters, [users])
        self.assertEqual(accepted, 0)
        self.assertEqual(ocap._last_split_diagnostics["skipped_geometry"], 1)
        self.assertEqual(ocap._last_split_diagnostics["accepted"], 0)

    def test_dynamic_user_reassignment_runs_on_first_boundary(self):
        cfg = load_config(dataset="synthetic")
        cfg.refinements.reassignment_enabled = True
        cfg.refinements.reassignment_every = 10
        cfg.refinements.reassignment_min_observations = 5
        cfg.refinements.reassignment_min_buffer = 5
        ocap = OCAP(6, cfg)
        clusters = [{"u0", "u1"}, {"u2", "u3"}]
        sources = {}
        for uid in sorted(set().union(*clusters)):
            sources[uid] = ocap._personal_adapter_key(uid)
            ocap.user_record_counts[uid] = 5
            ocap.recent_data_buffers[uid] = deque(
                [RecentDataPair(np.zeros(6, dtype=np.float32), i % 2) for i in range(5)],
                maxlen=64,
            )

        def controlled_loss(uid: str, adapter_key: str) -> float:
            candidate = int(adapter_key.rsplit("::", 1)[-1])
            preferred = 1 if uid == "u0" else (0 if uid == "u1" else 1)
            return 0.0 if candidate == preferred else 1.0

        ocap._recent_loss = controlled_loss  # type: ignore[method-assign]
        refined, moves = ocap._apply_dynamic_user_reassignment(
            clusters, sorted(set().union(*clusters)), sources, step=100,
        )
        self.assertEqual(ocap.group_update_count, 0)
        self.assertGreaterEqual(moves, 1)
        self.assertIn("u0", refined[1])
        self.assertGreater(ocap._last_reassignment_diagnostics["evaluated_users"], 0)
        self.assertEqual(ocap._last_reassignment_diagnostics["moves"], moves)

    def test_gradient_history_uses_raw_persistent_ema(self):
        cfg = load_config(dataset="synthetic")
        cfg.training.gradient_history_decay = 0.8
        ocap = OCAP(3, cfg)
        first = torch.zeros_like(ocap.model.backbone.gradient_projection)
        first.reshape(-1)[:2] = torch.tensor([3.0, 4.0])
        ocap.model.backbone.gradient_projection.grad = first
        ocap._update_gradient_history("u0", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            ocap.interval_gradient_history["u0"][:2], np.array([3.0, 4.0]), atol=1e-7,
        )
        self.assertEqual(ocap.interval_gradient_count["u0"], 1)

        second = torch.zeros_like(ocap.model.backbone.gradient_projection)
        second.reshape(-1)[:2] = torch.tensor([1.0, 2.0])
        ocap.model.backbone.gradient_projection.grad = second
        ocap._update_gradient_history("u0", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            ocap.interval_gradient_history["u0"][:2], np.array([2.6, 3.6]), atol=1e-6,
        )
        self.assertEqual(ocap.interval_gradient_count["u0"], 1)
        self.assertEqual(ocap.gradient_history_count["u0"], 2)

    def test_update_groups_projection_matches_standardized_geometry(self):
        representations = {
            "u0": np.array([1.0, 2.0, 4.0]),
            "u1": np.array([2.0, 1.0, 3.0]),
            "u2": np.array([3.0, 4.0, 1.0]),
            "u3": np.array([4.0, 3.0, 2.0]),
        }
        projected = OCAP._project_user_representations(
            list(representations), representations,
        )
        ordered = sorted(representations)
        original = np.stack([representations[uid] for uid in ordered]).astype(np.float32)
        standardized = (original - original.mean(axis=0)) / original.std(axis=0)
        coordinates = np.stack([projected[uid] for uid in ordered])
        self.assertEqual(coordinates.shape, (4, 3))
        np.testing.assert_allclose(
            coordinates @ coordinates.T,
            standardized @ standardized.T,
            atol=2e-5,
        )

        rng = np.random.default_rng(7)
        wide = {f"w{i:02d}": rng.normal(size=20) for i in range(12)}
        projected_wide = OCAP._project_user_representations(list(wide), wide)
        wide_order = sorted(wide)
        wide_matrix = np.stack([wide[uid] for uid in wide_order]).astype(np.float32)
        wide_standardized = (
            wide_matrix - wide_matrix.mean(axis=0)
        ) / wide_matrix.std(axis=0)
        _, _, vh = np.linalg.svd(wide_standardized, full_matrices=False)
        historical_coordinates = wide_standardized @ vh[:8].T
        gram_coordinates = np.stack([projected_wide[uid] for uid in wide_order])
        np.testing.assert_allclose(
            gram_coordinates @ gram_coordinates.T,
            historical_coordinates @ historical_coordinates.T,
            atol=5e-4,
        )

    def test_new_group_sources_are_personal_adapters(self):
        cfg = load_config(dataset="synthetic")
        ocap = OCAP(3, cfg)
        ocap.group_set = {"g0": {"u0"}}
        ocap.group_assignment = {"u0": "g0"}
        self.assertEqual(ocap._routed_adapter_key("u0"), "group::g0")
        self.assertEqual(ocap._personal_adapter_source("u0"), "personal::u0")

    def test_gradient_histories_clear_only_after_initial_grouping(self):
        cfg = load_config(dataset="synthetic")
        ocap = OCAP(3, cfg, method="per_user_adapter")
        ocap._apply_group_gradient_update = lambda: 0  # type: ignore[method-assign]
        ocap.interval_gradient_history = {"u0": np.array([1.0, 2.0])}
        ocap.interval_gradient_count["u0"] = 1
        ocap._boundary(step=20, initial=True)
        self.assertEqual(ocap.interval_gradient_history, {})

        ocap.interval_gradient_history = {"u0": np.array([2.0, 3.0])}
        ocap.interval_gradient_count["u0"] = 7
        ocap._boundary(step=40, initial=False)
        np.testing.assert_allclose(ocap.interval_gradient_history["u0"], [2.0, 3.0])
        self.assertEqual(ocap.interval_gradient_count["u0"], 1)

    def test_first_k_starts_at_each_users_first_data_triplet(self):
        cfg = load_config(dataset="synthetic")
        cfg.grouping.backbone_warmup_steps = 2
        cfg.evaluation.first_k = [2]
        cfg.evaluation.require_complete_first_k = True
        ocap = OCAP(3, cfg, method="per_user_adapter")
        stream = [
            DataTriplet(index=0, user_id="u0", timestamp="0", label=0,
                        features=np.zeros(3, dtype=np.float32)),
            DataTriplet(index=1, user_id="u1", timestamp="1", label=1,
                        features=np.ones(3, dtype=np.float32)),
            DataTriplet(index=2, user_id="u0", timestamp="2", label=1,
                        features=np.ones(3, dtype=np.float32)),
            DataTriplet(index=3, user_id="u1", timestamp="3", label=0,
                        features=np.zeros(3, dtype=np.float32)),
        ]
        result = ocap.run(stream)
        self.assertEqual(result["low_evidence"][0]["n_users"], 2.0)
        self.assertEqual(result["low_evidence"][0]["n_data_triplets"], 4.0)


if __name__ == "__main__":
    unittest.main()
