import unittest

from ocap.config import load_config
from ocap.settings import apply_main_configuration


class MainConfigurationTest(unittest.TestCase):
    def test_main_parameters(self):
        expected = {
            "ces": (0.80, 20, 5),
            "globem": (0.90, 2, 5),
        }
        for dataset, values in expected.items():
            config, main_configuration = apply_main_configuration(
                load_config(dataset=dataset), "main",
            )
            self.assertEqual(
                (
                    config.training.gradient_history_decay,
                    config.grouping.min_observations,
                    config.grouping.minimum_group_count,
                ),
                values,
            )
            self.assertFalse(main_configuration.strict_initial_grouping)
            self.assertFalse(config.refinements.verified_split)
            self.assertEqual(config.refinements.reassignment_min_observations, 20)
            self.assertEqual(config.refinements.reassignment_loss_margin, 0.01)

    def test_globem_main_configuration(self):
        config, _ = apply_main_configuration(
            load_config(dataset="globem"), "main",
        )
        self.assertEqual(config.grouping.warmup_initial_min_samples, 5)
        self.assertEqual(config.grouping.warmup_initial_min_samples_fallback, 1)
        self.assertEqual(config.training.gradient_history_decay, 0.90)
        self.assertEqual(config.grouping.minimum_group_count, 5)
