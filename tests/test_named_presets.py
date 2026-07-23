import unittest

from sgshare.config import load_config
from sgshare.presets import apply_named_preset


class NamedPresetTest(unittest.TestCase):
    def test_classification_tuned_parameters(self):
        expected = {
            "ces": (0.80, 20, 5),
            "globem": (0.90, 2, 5),
        }
        for dataset, values in expected.items():
            config, preset = apply_named_preset(
                load_config(dataset=dataset), "classification_tuned",
            )
            self.assertEqual(
                (
                    config.training.route_grad_ema_decay,
                    config.grouping.min_observations,
                    config.grouping.k_min,
                ),
                values,
            )
            self.assertFalse(preset.strict_initial_eligibility)
            self.assertFalse(config.refinements.verified_split)
            self.assertEqual(config.refinements.mature_min_observations, 20)
            self.assertEqual(config.refinements.mature_loss_margin, 0.01)

    def test_cold_safe_parameters(self):
        ces, ces_preset = apply_named_preset(
            load_config(dataset="ces"), "cold_safe",
        )
        self.assertEqual(ces.training.route_grad_ema_decay, 0.95)
        self.assertEqual(ces.grouping.min_observations, 20)
        self.assertEqual(ces.grouping.k_min, 3)
        self.assertFalse(ces_preset.strict_initial_eligibility)

        globem, globem_preset = apply_named_preset(
            load_config(dataset="globem"), "cold_safe",
        )
        self.assertEqual(globem.training.route_grad_ema_decay, 0.88)
        self.assertEqual(globem.grouping.min_observations, 2)
        self.assertEqual(globem.grouping.k_min, 4)
        self.assertEqual(globem.grouping.warmup_initial_min_samples, 5)
        self.assertEqual(globem.grouping.warmup_initial_min_samples_fallback, 1)
        self.assertFalse(globem_preset.strict_initial_eligibility)

    def test_globem_classification_and_cold_safe_use_distinct_alpha_and_k(self):
        classification, _ = apply_named_preset(
            load_config(dataset="globem"), "classification_tuned",
        )
        cold_safe, _ = apply_named_preset(
            load_config(dataset="globem"), "cold_safe",
        )
        self.assertEqual(classification.grouping.warmup_initial_min_samples, 5)
        self.assertEqual(classification.grouping.warmup_initial_min_samples_fallback, 1)
        self.assertEqual(classification.training.route_grad_ema_decay, 0.90)
        self.assertEqual(cold_safe.training.route_grad_ema_decay, 0.88)
        self.assertEqual(classification.grouping.k_min, 5)
        self.assertEqual(cold_safe.grouping.k_min, 4)
