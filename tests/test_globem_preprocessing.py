import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from scripts.align_globem_official import DEFAULT_MODALITIES, build_dataset
from scripts.prepare_globem_weekly import build_globem_weekly_arff


class GlobemHistoricalPreprocessingTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        cohort = root / "INS-W_1"
        (cohort / "SurveyData").mkdir(parents=True)
        (cohort / "ParticipantsInfoData").mkdir()
        (cohort / "FeatureData").mkdir()
        pd.DataFrame({
            "pid": ["1", "1", "2"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-01"],
            "dep": [False, True, False],
            "BDI2": [2, 10, 3],
        }).to_csv(cohort / "SurveyData" / "dep_weekly.csv", index=False)
        pd.DataFrame({
            "pid": ["1", "2"],
            "platform": ["ios", "android"],
        }).to_csv(cohort / "ParticipantsInfoData" / "platform.csv", index=False)

        for offset, modality in enumerate(DEFAULT_MODALITIES):
            pd.DataFrame({
                "pid": ["1", "1"],
                "date": ["2020-01-01", "2020-01-02"],
                f"{modality}:good": [1.0 + offset, 3.0 + offset],
                f"{modality}_dis:rank": [1, 2],
                f"{modality}:mixed": ["1", "bad"],
                f"{modality}:missing": [np.nan, np.nan],
            }).to_csv(cohort / "FeatureData" / f"{modality}.csv", index=False)

    def test_historical_alignment_and_top_k_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            self._write_fixture(root)
            stage1 = Path(directory) / "stage1"
            build_dataset(
                root=root,
                out_dir=stage1,
                modalities=list(DEFAULT_MODALITIES),
                max_missing_ratio=0.90,
                drop_discrete_rank=True,
            )

            aligned_path = stage1 / "globem_weekly_clean_all_cohorts.csv.gz"
            aligned = pd.read_csv(aligned_path)
            self.assertEqual(len(aligned), 3)
            self.assertEqual(aligned["uid"].nunique(), 2)
            self.assertEqual(set(aligned["uid"]), {"INS-W_1::1", "INS-W_1::2"})
            self.assertFalse(any("_dis:" in column for column in aligned.columns))
            self.assertFalse(any(":mixed" in column for column in aligned.columns))
            self.assertFalse(any(":missing" in column for column in aligned.columns))
            self.assertTrue(aligned.loc[aligned["pid"] == 2, "bluetooth:good"].isna().all())

            arff_path = stage1 / "globem_weekly_v5_top4.arff"
            summary = build_globem_weekly_arff(
                input_csv=aligned_path,
                output_arff=arff_path,
                top_k_features=4,
            )
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["participants"], 2)
            self.assertEqual(summary["selected_behavior_features"], 4)
            self.assertEqual(summary["selected_aux_features"], 2)
            self.assertEqual(summary["total_selected_features"], 6)

            rows, _ = arff.loadarff(arff_path)
            result = pd.DataFrame(rows)
            numeric = result.drop(columns=["uid", "class"]).astype(float)
            self.assertFalse(numeric.isna().any().any())
            np.testing.assert_allclose(numeric.mean(axis=0), 0.0, atol=1e-7)
            varying = numeric.std(axis=0, ddof=0) > 0
            np.testing.assert_allclose(
                numeric.loc[:, varying].std(axis=0, ddof=0), 1.0, atol=1e-7
            )


if __name__ == "__main__":
    unittest.main()
