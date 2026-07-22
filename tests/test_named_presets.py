import unittest

from sgshare.config import load_config
from sgshare.presets import apply_named_preset


class NamedPresetTest(unittest.TestCase):
    def test_classification_tuned_parameters(self):
        expected = {
            "ces": (0.80, 20, 4),
            "globem": (0.80, 2, 4),
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

    def test_cold_safe_parameters(self):
        ces, ces_preset = apply_named_preset(
            load_config(dataset="ces"), "cold_safe",
        )
        self.assertEqual(ces.training.route_grad_ema_decay, 0.95)
        self.assertEqual(ces.grouping.min_observations, 20)
        self.assertFalse(ces_preset.strict_initial_eligibility)

        globem, globem_preset = apply_named_preset(
            load_config(dataset="globem"), "cold_safe",
        )
        self.assertEqual(globem.training.route_grad_ema_decay, 0.80)
        self.assertEqual(globem.grouping.min_observations, 2)
        self.assertEqual(globem.grouping.warmup_initial_min_samples, 2)
        self.assertEqual(globem.grouping.warmup_initial_min_samples_fallback, 2)
        self.assertTrue(globem_preset.strict_initial_eligibility)

    def test_globem_classification_and_cold_safe_differ_only_in_initial_gate(self):
        classification, _ = apply_named_preset(
            load_config(dataset="globem"), "classification_tuned",
        )
        cold_safe, _ = apply_named_preset(
            load_config(dataset="globem"), "cold_safe",
        )
        self.assertEqual(classification.grouping.warmup_initial_min_samples, 5)
        self.assertEqual(classification.grouping.warmup_initial_min_samples_fallback, 1)
        self.assertEqual(cold_safe.grouping.warmup_initial_min_samples, 2)
        self.assertEqual(cold_safe.grouping.warmup_initial_min_samples_fallback, 2)
