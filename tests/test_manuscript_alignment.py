from __future__ import annotations

import unittest

from process.run_manuscript_ablations import _configured


class ManuscriptAlignmentTest(unittest.TestCase):
    def test_full_uses_selected_classification_configuration(self) -> None:
        ces, method, signal, changed = _configured("ces", "full", 42, "cpu")
        self.assertEqual(method, "full_final")
        self.assertEqual(signal, "gradient")
        self.assertEqual(changed, [])
        self.assertEqual(ces.training.route_grad_ema_decay, 0.80)
        self.assertEqual(ces.grouping.min_observations, 20)
        self.assertEqual(ces.grouping.k_min, 5)
        self.assertFalse(ces.refinements.verified_split)
        self.assertEqual(ces.refinements.mature_min_observations, 20)
        self.assertEqual(ces.refinements.mature_loss_margin, 0.01)

        globem, _, _, _ = _configured("globem", "full", 42, "cpu")
        self.assertEqual(globem.training.route_grad_ema_decay, 0.90)
        self.assertEqual(globem.grouping.min_observations, 2)
        self.assertEqual(globem.grouping.k_min, 5)
        self.assertFalse(globem.refinements.verified_split)

    def test_component_ablation_changes_are_isolated(self) -> None:
        no_reassign, method, signal, _ = _configured(
            "ces", "without_reassignment", 42, "cpu",
        )
        self.assertEqual(method, "full_final")
        self.assertEqual(signal, "gradient")
        self.assertFalse(no_reassign.grouping.periodic_regroup_enabled)
        self.assertFalse(no_reassign.refinements.mature_refine)
        self.assertTrue(no_reassign.refinements.per_user_bias)

        feature, method, signal, _ = _configured(
            "ces", "feature_grouping", 42, "cpu",
        )
        self.assertEqual(method, "full_final")
        self.assertEqual(signal, "feature")
        self.assertTrue(feature.grouping.periodic_regroup_enabled)
        self.assertTrue(feature.refinements.mature_refine)


if __name__ == "__main__":
    unittest.main()
