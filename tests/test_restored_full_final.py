import unittest
from collections import deque

import numpy as np
import torch

from sgshare.config import load_config
from sgshare.data import StreamEvent
from sgshare.learner import BufferedObservation, SGShareLearner
from sgshare.model import SGModel, weighted_focal_loss


class RestoredFullFinalTest(unittest.TestCase):
    def test_ces_and_globem_historical_configs(self):
        ces = load_config(dataset="ces")
        globem = load_config(dataset="globem")
        self.assertEqual((ces.model.hidden_dim, ces.model.depth, ces.model.num_heads), (32, 2, 4))
        self.assertEqual((globem.model.hidden_dim, globem.model.depth, globem.model.num_heads), (16, 1, 2))
        self.assertEqual(ces.model.lora_rank, 4)
        self.assertEqual(ces.model.lora_alpha, 8.0)
        self.assertEqual(ces.grouping.k_min, 4)
        self.assertEqual(ces.grouping.warmup_initial_min_samples, 5)
        self.assertTrue(ces.grouping.warmup_initial_force_accept)
        self.assertEqual(ces.grouping.warmup_initial_target_groups, 0)
        self.assertEqual(ces.grouping.warmup_initial_min_groups, 2)
        self.assertEqual(ces.grouping.warmup_initial_max_groups, 12)
        self.assertEqual(ces.refinements.per_user_bias_lr, 0.005)
        self.assertEqual(ces.refinements.split_min_users, 10)
        self.assertEqual(ces.refinements.split_min_events, 50)
        self.assertEqual(ces.refinements.mature_every, 10)
        self.assertEqual(ces.grouping.cfl_coherence_threshold, 0.55)
        self.assertEqual(ces.grouping.cfl_disagreement_cosine_threshold, 0.20)
        self.assertEqual(ces.refinements.split_cohesion_tolerance, 0.05)

    def test_route_and_lora_parameterization(self):
        cfg = load_config(dataset="ces")
        model = SGModel(37, cfg.model)
        adapter = model.ensure_adapter("test")
        self.assertEqual(tuple(model.backbone.route_w.shape), (32, 32))
        self.assertEqual(tuple(adapter.A.shape), (4, 32))
        self.assertEqual(tuple(adapter.B.shape), (32, 4))
        self.assertEqual(adapter.scale, 2.0)

    def test_positive_class_weight_is_active(self):
        negative = weighted_focal_loss(torch.zeros(1, 2), torch.tensor([0]), 0.75, 2.0, 5.0)
        positive = weighted_focal_loss(torch.zeros(1, 2), torch.tensor([1]), 0.75, 2.0, 5.0)
        self.assertGreater(float(positive), float(negative) * 10.0)

    def test_optimizer_types_match_historical_implementation(self):
        cfg = load_config(dataset="synthetic")
        learner = SGShareLearner(6, cfg)
        self.assertIsInstance(learner.backbone_optimizer, torch.optim.AdamW)
        self.assertIsInstance(learner.adapter_optimizers["global"], torch.optim.Adam)
        self.assertEqual(learner.backbone_optimizer.param_groups[0]["weight_decay"], 1e-4)
        self.assertEqual(learner.adapter_optimizers["global"].param_groups[0]["weight_decay"], 0.0)

    def test_bias_is_absent_during_warmup_and_uses_group_mean_afterward(self):
        cfg = load_config(dataset="synthetic")
        learner = SGShareLearner(6, cfg)
        learner.groups = {"g0": {"u0", "u1"}}
        learner.assignment = {"u0": "g0", "u1": "g0"}
        learner.user_bias["u1"] = 1.0
        logits = torch.zeros(1, 2)
        warmup_logits = learner._apply_bias(logits, "u0", warmup=True)
        self.assertTrue(torch.equal(warmup_logits, logits))
        self.assertNotIn("u0", learner.user_bias)
        post_logits = learner._apply_bias(logits, "u0", warmup=False)
        self.assertAlmostEqual(learner.user_bias["u0"], 0.5)
        self.assertAlmostEqual(float(post_logits[0, 1]), 0.5)

    def test_verified_split_requires_cfl_disagreement(self):
        cfg = load_config(dataset="synthetic")
        cfg.refinements.verified_split = True
        cfg.refinements.split_min_users = 2
        learner = SGShareLearner(6, cfg)
        users = {f"u{i}" for i in range(4)}
        signatures = {uid: np.ones(4, dtype=np.float64) for uid in users}
        clusters, accepted = learner._verified_splits(
            [users], signatures, {uid: f"temp::{uid}" for uid in users}, step=10,
        )
        self.assertEqual(clusters, [users])
        self.assertEqual(accepted, 0)
        self.assertEqual(learner._last_split_diagnostics["skipped_geometry"], 1)
        self.assertEqual(learner._last_split_diagnostics["accepted"], 0)

    def test_mature_refinement_runs_on_first_boundary(self):
        cfg = load_config(dataset="synthetic")
        cfg.refinements.mature_refine = True
        cfg.refinements.mature_every = 10
        cfg.refinements.mature_min_observations = 5
        cfg.refinements.mature_min_buffer = 5
        learner = SGShareLearner(6, cfg)
        clusters = [{"u0", "u1"}, {"u2", "u3"}]
        sources = {}
        for uid in sorted(set().union(*clusters)):
            sources[uid] = learner._temp_key(uid)
            learner.seen[uid] = 5
            learner.recent[uid] = deque(
                [BufferedObservation(np.zeros(6, dtype=np.float32), i % 2) for i in range(5)],
                maxlen=64,
            )

        def controlled_loss(uid: str, adapter_key: str) -> float:
            candidate = int(adapter_key.rsplit("::", 1)[-1])
            preferred = 1 if uid == "u0" else (0 if uid == "u1" else 1)
            return 0.0 if candidate == preferred else 1.0

        learner._recent_loss = controlled_loss  # type: ignore[method-assign]
        refined, moves = learner._mature_refine_clusters(
            clusters, sorted(set().union(*clusters)), sources, step=100,
        )
        self.assertEqual(learner.regroup_count, 0)
        self.assertGreaterEqual(moves, 1)
        self.assertIn("u0", refined[1])
        self.assertGreater(learner._last_mature_diagnostics["evaluated_users"], 0)
        self.assertEqual(learner._last_mature_diagnostics["moves"], moves)

    def test_route_signature_uses_raw_persistent_ema(self):
        cfg = load_config(dataset="synthetic")
        cfg.training.route_grad_ema_decay = 0.8
        learner = SGShareLearner(3, cfg)
        first = torch.zeros_like(learner.model.backbone.route_w)
        first.reshape(-1)[:2] = torch.tensor([3.0, 4.0])
        learner.model.backbone.route_w.grad = first
        learner._update_signature("u0", np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            learner.window_route_sum["u0"][:2], np.array([3.0, 4.0]), atol=1e-7,
        )
        self.assertEqual(learner.window_route_count["u0"], 1)

        second = torch.zeros_like(learner.model.backbone.route_w)
        second.reshape(-1)[:2] = torch.tensor([1.0, 2.0])
        learner.model.backbone.route_w.grad = second
        learner._update_signature("u0", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            learner.window_route_sum["u0"][:2], np.array([2.6, 3.6]), atol=1e-6,
        )
        self.assertEqual(learner.window_route_count["u0"], 1)
        self.assertEqual(learner.signature_count["u0"], 2)

    def test_regroup_projection_matches_standardized_geometry(self):
        signatures = {
            "u0": np.array([1.0, 2.0, 4.0]),
            "u1": np.array([2.0, 1.0, 3.0]),
            "u2": np.array([3.0, 4.0, 1.0]),
            "u3": np.array([4.0, 3.0, 2.0]),
        }
        projected = SGShareLearner._project_regroup_signatures(
            list(signatures), signatures,
        )
        ordered = sorted(signatures)
        original = np.stack([signatures[uid] for uid in ordered]).astype(np.float32)
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
        projected_wide = SGShareLearner._project_regroup_signatures(list(wide), wide)
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
        learner = SGShareLearner(3, cfg)
        learner.groups = {"g0": {"u0"}}
        learner.assignment = {"u0": "g0"}
        self.assertEqual(learner._active_adapter("u0"), "group::g0")
        self.assertEqual(learner._source_key("u0"), "temp::u0")

    def test_signature_buffers_clear_only_after_initial_grouping(self):
        cfg = load_config(dataset="synthetic")
        learner = SGShareLearner(3, cfg, method="per_user_adapter")
        learner._apply_group_route_update = lambda: 0  # type: ignore[method-assign]
        learner.window_route_sum = {"u0": np.array([1.0, 2.0])}
        learner.window_route_count["u0"] = 1
        learner._boundary(step=20, initial=True)
        self.assertEqual(learner.window_route_sum, {})

        learner.window_route_sum = {"u0": np.array([2.0, 3.0])}
        learner.window_route_count["u0"] = 7
        learner._boundary(step=40, initial=False)
        np.testing.assert_allclose(learner.window_route_sum["u0"], [2.0, 3.0])
        self.assertEqual(learner.window_route_count["u0"], 1)

    def test_first_k_starts_at_each_users_first_stream_event(self):
        cfg = load_config(dataset="synthetic")
        cfg.grouping.global_warmup_steps = 2
        cfg.evaluation.first_k = [2]
        cfg.evaluation.require_complete_first_k = True
        learner = SGShareLearner(3, cfg, method="per_user_adapter")
        stream = [
            StreamEvent(index=0, user_id="u0", timestamp="0", label=0,
                        features=np.zeros(3, dtype=np.float32)),
            StreamEvent(index=1, user_id="u1", timestamp="1", label=1,
                        features=np.ones(3, dtype=np.float32)),
            StreamEvent(index=2, user_id="u0", timestamp="2", label=1,
                        features=np.ones(3, dtype=np.float32)),
            StreamEvent(index=3, user_id="u1", timestamp="3", label=0,
                        features=np.zeros(3, dtype=np.float32)),
        ]
        result = learner.run(stream)
        self.assertEqual(result["cold_start"][0]["n_users"], 2.0)
        self.assertEqual(result["cold_start"][0]["n_events"], 4.0)


if __name__ == "__main__":
    unittest.main()
