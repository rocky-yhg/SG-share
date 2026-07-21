import unittest

import numpy as np

from sgshare.grouping import agglomerative_groups


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


if __name__ == "__main__":
    unittest.main()
