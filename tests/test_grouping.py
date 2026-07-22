import unittest

import numpy as np

from sgshare.grouping import agglomerative_groups, incremental_split_merge_groups


class GroupingTest(unittest.TestCase):
    def test_positive_pairs_merge_until_k_min(self):
        signatures = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.9, 0.1]),
            "c": np.array([0.0, 1.0]),
            "d": np.array([0.1, 0.9]),
        }
        result = agglomerative_groups(list(signatures), signatures, k_min=2)
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.stop_reason, "k_min")

    def test_negative_similarity_stops_before_target(self):
        signatures = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([-1.0, 0.0]),
        }
        result = agglomerative_groups(list(signatures), signatures, k_min=1)
        self.assertEqual(len(result.clusters), 2)
        self.assertEqual(result.stop_reason, "negative_best_gain")

    def test_incremental_grouping_reuses_partition_and_merges_new_user(self):
        signatures = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.9, 0.1]),
            "c": np.array([0.0, 1.0]),
            "d": np.array([0.1, 0.9]),
            "e": np.array([0.0, 1.0]),
        }
        result = incremental_split_merge_groups(
            list(signatures), signatures, [{"a", "b"}, {"c", "d"}],
            k_min=2, max_split_merge_swaps=0,
        )
        self.assertEqual(result.reused_users, 4)
        self.assertEqual(result.new_users, 1)
        self.assertEqual(result.merges, 1)
        self.assertIn({"c", "d", "e"}, result.clusters)

    def test_incremental_split_merge_requires_cohesion_improvement(self):
        signatures = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([-1.0, 0.0]),
            "c": np.array([0.0, 1.0]),
            "d": np.array([0.0, 0.9]),
        }
        result = incremental_split_merge_groups(
            list(signatures), signatures, [{"a", "b"}, {"c", "d"}],
            k_min=2, split_min_users=2, max_split_merge_swaps=1,
        )
        self.assertEqual(result.split_attempts, 1)
        self.assertEqual(result.split_accepts, 1)
        self.assertGreater(result.objective_after, result.objective_before)


if __name__ == "__main__":
    unittest.main()
