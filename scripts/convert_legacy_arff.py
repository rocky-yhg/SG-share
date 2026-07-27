#!/usr/bin/env python3
"""Convert historical uid/features/class ARFF streams while preserving row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.io import arff


def convert(input_arff: Path, output_csv: Path, user_column: str = "uid",
            label_column: str = "class", min_user_rows: int = 0) -> dict:
    records, _ = arff.loadarff(input_arff)
    frame = pd.DataFrame(records)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
    if user_column not in frame or label_column not in frame:
        raise ValueError(f"Expected {user_column!r} and {label_column!r} in {input_arff}")
    frame.insert(0, "timestamp", range(len(frame)))
    frame = frame.rename(columns={user_column: "user_id", label_column: "label"})
    frame["user_id"] = frame["user_id"].astype(str)
    labels = pd.to_numeric(frame["label"], errors="coerce")
    values = sorted(labels.dropna().unique())
    if len(values) != 2:
        raise ValueError(f"Expected binary class values, got {values}")
    frame["label"] = (labels == values[-1]).astype(int)
    if min_user_rows > 0:
        counts = frame.groupby("user_id").size()
        keep = counts[counts >= min_user_rows].index
        frame = frame[frame["user_id"].isin(keep)].copy()
    ordered = ["user_id", "timestamp", "label"] + [
        column for column in frame.columns if column not in {"user_id", "timestamp", "label"}
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame[ordered].to_csv(output_csv, index=False)
    summary = {
        "input_arff": str(input_arff),
        "output_csv": str(output_csv),
        "rows": len(frame),
        "users": int(frame["user_id"].nunique()),
        "features": len(ordered) - 3,
        "positive_rate": float(frame["label"].mean()),
        "min_user_rows": min_user_rows,
        "row_order_preserved": True,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-arff", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--user-column", default="uid")
    parser.add_argument("--label-column", default="class")
    parser.add_argument("--min-user-rows", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(convert(
        Path(args.input_arff), Path(args.output_csv), args.user_column,
        args.label_column, args.min_user_rows,
    ), indent=2))


if __name__ == "__main__":
    main()
