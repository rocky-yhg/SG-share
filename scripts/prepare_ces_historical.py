#!/usr/bin/env python3
"""Rebuild the CES stream used by the historical SG-Share experiments.

The historical input was ``ces_phq4_stage1_ordered_bin_usernorm.arff``.  Its
construction is recoverable from execution logs: select the I-HOPE Stage-1
35-feature view, fill missing values with zero, sort by day and uid, and apply
per-user population z-scoring to every sensing feature while leaving ``day``
as epoch seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 35_289
EXPECTED_USERS = 218
EXPECTED_POSITIVES = 10_053

DIRECT_FEATURES = [
    "is_ios",
    "act_on_bike_ep_0",
    "act_on_foot_ep_0",
    "act_running_ep_0",
    "act_still_ep_0",
    "act_walking_ep_0",
    "audio_convo_duration_ep_0",
    "loc_food_audio_voice",
    "loc_home_audio_voice",
    "loc_social_audio_voice",
    "loc_other_dorm_audio_voice",
    "loc_self_dorm_audio_voice",
    "loc_study_audio_voice",
    "loc_food_convo_duration",
    "loc_home_convo_duration",
    "loc_other_dorm_convo_duration",
    "loc_social_convo_duration",
    "loc_study_convo_duration",
    "loc_self_dorm_convo_duration",
    "loc_home_dur",
    "loc_leisure_dur",
    "loc_other_dorm_dur",
    "loc_self_dorm_dur",
    "loc_social_dur",
    "loc_study_dur",
    "loc_workout_dur",
    "sleep_duration",
    "sleep_heathkit_dur",
]

DERIVED_FEATURES = [
    "call_rate_ep_0",
    "loc_home_unlock_rate",
    "loc_other_dorm_unlock_rate",
    "loc_self_dorm_unlock_rate",
    "loc_social_unlock_rate",
    "loc_study_unlock_rate",
    "unlock_rate_ep_0",
    "sleep_span",
]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build(source_root: Path, output_csv: Path, strict_release: bool = True) -> dict:
    ema = pd.read_csv(
        source_root / "EMA" / "general_ema.csv",
        usecols=["uid", "day", "phq4_score"],
    )
    sensing = pd.read_csv(source_root / "Sensing" / "sensing.csv", low_memory=False)

    ema["phq4_score"] = pd.to_numeric(ema["phq4_score"], errors="coerce")
    ema = ema.dropna(subset=["uid", "day", "phq4_score"])
    ema = ema.drop_duplicates(["uid", "day"], keep="last")
    sensing = sensing.drop_duplicates(["uid", "day"], keep="last")
    frame = ema.merge(sensing, on=["uid", "day"], how="inner", validate="one_to_one")

    missing = [column for column in DIRECT_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Official CES sensing table is missing historical features: {missing}")

    selected = frame[DIRECT_FEATURES].apply(pd.to_numeric, errors="coerce")
    selected["call_rate_ep_0"] = _ratio(
        pd.to_numeric(frame["call_in_num_ep_0"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["call_out_num_ep_0"], errors="coerce").fillna(0.0),
        pd.to_numeric(frame["call_in_duration_ep_0"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["call_out_duration_ep_0"], errors="coerce").fillna(0.0),
    )
    for location in ["home", "other_dorm", "self_dorm", "social", "study"]:
        selected[f"loc_{location}_unlock_rate"] = _ratio(
            frame[f"loc_{location}_unlock_num"],
            frame[f"loc_{location}_unlock_duration"],
        )
    selected["unlock_rate_ep_0"] = _ratio(
        frame["unlock_num_ep_0"], frame["unlock_duration_ep_0"]
    )
    selected["sleep_span"] = (
        pd.to_numeric(frame["sleep_end"], errors="coerce")
        - pd.to_numeric(frame["sleep_start"], errors="coerce")
    )

    feature_columns = DIRECT_FEATURES + DERIVED_FEATURES
    selected = selected[feature_columns].fillna(0.0)
    user_key = frame["uid"].astype(str)
    means = selected.groupby(user_key, sort=False).transform("mean")
    stds = selected.groupby(user_key, sort=False).transform(
        lambda values: values.std(ddof=0)
    ).replace(0.0, 1.0)
    selected = (selected - means) / stds

    timestamp = pd.to_datetime(
        frame["day"].astype("Int64").astype(str), format="%Y%m%d", utc=True
    )
    day_epoch = timestamp.astype("int64") // 10**9
    output = pd.concat(
        [
            pd.DataFrame({
                "user_id": user_key,
                "timestamp": timestamp,
                "label": (frame["phq4_score"] >= 4.0).astype(int),
                "day": day_epoch.astype(float),
            }),
            selected.reset_index(drop=True),
        ],
        axis=1,
    )
    output = output.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)

    if strict_release:
        checks = {
            "rows": (len(output), EXPECTED_ROWS),
            "users": (output["user_id"].nunique(), EXPECTED_USERS),
            "positives": (output["label"].sum(), EXPECTED_POSITIVES),
        }
        failures = {key: values for key, values in checks.items() if values[0] != values[1]}
        if failures:
            raise ValueError(f"Historical CES alignment failed: {failures}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    summary = {
        "dataset": "CES",
        "profile": "historical_stage1_ordered_bin_usernorm",
        "rows": int(len(output)),
        "users": int(output["user_id"].nunique()),
        "positive_count": int(output["label"].sum()),
        "positive_rate": float(output["label"].mean()),
        "model_features": int(1 + len(feature_columns)),
        "sensing_features": int(len(feature_columns)),
        "feature_order": ["day"] + feature_columns,
        "ordering": "day ascending, uid ascending, stable; no within-day shuffle",
        "missing_values": "coerce numeric then fill 0 before per-user normalization",
        "normalization": "per-user z-score, ddof=0; uid/day excluded",
        "label_rule": "phq4_score >= 4",
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the historical CES SG-Share stream")
    parser.add_argument("--source-root", default="data/raw/ces/official_v5")
    parser.add_argument("--output-csv", default="data/processed/ces_historical_usernorm.csv")
    parser.add_argument("--no-strict-release", action="store_true")
    args = parser.parse_args()
    summary = build(
        Path(args.source_root), Path(args.output_csv), not args.no_strict_release
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
