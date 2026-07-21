#!/usr/bin/env python3
"""Build the CES PHQ-4 online stream from the official Kaggle release.

The College Experience Study release contains daily passive-sensing features in
``Sensing/sensing.csv`` and PHQ-4 observations in ``EMA/general_ema.csv``.
This script joins them on the documented ``(uid, day)`` keys and preserves raw
numeric features. Scaling is deliberately deferred to the prequential loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_RELEASE_ROWS = 35_289
EXPECTED_RELEASE_USERS = 218
META_COLUMNS = {"uid", "day"}


def build(
    source_root: Path,
    output_csv: Path,
    positive_threshold: float = 4.0,
    strict_release: bool = True,
) -> dict:
    ema_path = source_root / "EMA" / "general_ema.csv"
    sensing_path = source_root / "Sensing" / "sensing.csv"
    if not ema_path.is_file() or not sensing_path.is_file():
        raise FileNotFoundError(
            "Expected EMA/general_ema.csv and Sensing/sensing.csv under "
            f"{source_root}"
        )

    ema = pd.read_csv(ema_path, usecols=["uid", "day", "phq4_score"])
    ema["phq4_score"] = pd.to_numeric(ema["phq4_score"], errors="coerce")
    ema = ema.dropna(subset=["uid", "day", "phq4_score"])
    ema = ema.drop_duplicates(["uid", "day"], keep="last")

    sensing = pd.read_csv(sensing_path, low_memory=False)
    sensing = sensing.drop_duplicates(["uid", "day"], keep="last")
    joined = ema.merge(sensing, on=["uid", "day"], how="inner", validate="one_to_one")

    joined["user_id"] = joined["uid"].astype(str)
    joined["timestamp"] = pd.to_datetime(
        joined["day"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=True
    )
    joined["label"] = (joined["phq4_score"] >= float(positive_threshold)).astype(int)
    joined = joined.dropna(subset=["timestamp"])

    candidate_features = [column for column in sensing.columns if column not in META_COLUMNS]
    numeric = joined[candidate_features].apply(pd.to_numeric, errors="coerce")
    feature_columns = [column for column in candidate_features if numeric[column].notna().any()]
    output = pd.concat(
        [joined[["user_id", "timestamp", "label"]].reset_index(drop=True),
         numeric[feature_columns].reset_index(drop=True)],
        axis=1,
    )
    output = output.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)

    if strict_release:
        if len(output) != EXPECTED_RELEASE_ROWS:
            raise ValueError(
                f"Official CES release alignment produced {len(output)} rows; "
                f"expected {EXPECTED_RELEASE_ROWS}."
            )
        users = int(output["user_id"].nunique())
        if users != EXPECTED_RELEASE_USERS:
            raise ValueError(
                f"Official CES release alignment produced {users} users; "
                f"expected {EXPECTED_RELEASE_USERS}."
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    manifest = pd.DataFrame({
        "feature": feature_columns,
        "source": "Sensing/sensing.csv",
        "non_missing": [int(numeric[column].notna().sum()) for column in feature_columns],
        "missing_rate": [float(numeric[column].isna().mean()) for column in feature_columns],
    })
    manifest.to_csv(output_csv.with_suffix(".manifest.csv"), index=False)

    summary = {
        "dataset": "CES",
        "official_source": "https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset",
        "source_root": str(source_root),
        "output_csv": str(output_csv),
        "join_keys": ["uid", "day"],
        "label_source": "EMA/general_ema.csv:phq4_score",
        "label_rule": f"phq4_score >= {positive_threshold:g}",
        "rows": int(len(output)),
        "users": int(output["user_id"].nunique()),
        "features": int(len(feature_columns)),
        "positive_count": int(output["label"].sum()),
        "positive_rate": float(output["label"].mean()),
        "first_timestamp": str(output["timestamp"].min()),
        "last_timestamp": str(output["timestamp"].max()),
        "raw_phq4_rows": int(len(ema)),
        "join_coverage": float(len(output) / len(ema)),
        "normalization": "none; deferred to online experiment loader",
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the CES PHQ-4 stream from the official Kaggle release"
    )
    parser.add_argument(
        "--source-root", default="data/raw/ces/official_v5",
        help="Directory containing the extracted EMA and Sensing folders",
    )
    parser.add_argument("--output-csv", default="data/processed/ces.csv")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--no-strict-release", action="store_true")
    args = parser.parse_args()
    summary = build(
        Path(args.source_root), Path(args.output_csv), args.positive_threshold,
        strict_release=not args.no_strict_release,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
