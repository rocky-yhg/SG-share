from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd
from scipy.io import arff


REQUIRED_COLUMNS = ("user_id", "timestamp", "label")


@dataclass(frozen=True)
class StreamEvent:
    index: int
    user_id: str
    timestamp: str
    label: int
    features: np.ndarray


class OnlineStandardizer:
    def __init__(self, n_features: int) -> None:
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def transform_then_update(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self.count < 2:
            transformed = x - self.mean
        else:
            variance = self.m2 / max(1, self.count - 1)
            transformed = (x - self.mean) / np.sqrt(np.maximum(variance, 1e-6))
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        return np.clip(transformed, -10.0, 10.0).astype(np.float32)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".arff":
        rows, _ = arff.loadarff(path)
        frame = pd.DataFrame(rows)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
                )
        return frame
    raise ValueError(f"Unsupported input format: {path}")


def prepare_stream(
    input_path: str | Path,
    output_path: str | Path,
    user_column: str,
    time_column: str,
    label_column: str,
    positive_values: Sequence[str] = ("1", "true", "positive", "yes"),
) -> pd.DataFrame:
    frame = _read_table(Path(input_path)).copy()
    missing = [name for name in (user_column, time_column, label_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    frame = frame.rename(columns={
        user_column: "user_id", time_column: "timestamp", label_column: "label"
    })
    labels = frame["label"]
    if not pd.api.types.is_numeric_dtype(labels):
        positive = {value.lower() for value in positive_values}
        frame["label"] = labels.astype(str).str.lower().isin(positive).astype(int)
    else:
        unique = sorted(pd.Series(labels).dropna().unique().tolist())
        if len(unique) != 2:
            raise ValueError(f"Expected a binary label, got values {unique}")
        frame["label"] = (labels == unique[-1]).astype(int)
    frame = frame.dropna(subset=["user_id", "timestamp", "label"])
    excluded = set(REQUIRED_COLUMNS)
    feature_columns = [name for name in frame.columns if name not in excluded]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    feature_columns = [name for name in feature_columns if frame[name].notna().any()]
    frame = frame[list(REQUIRED_COLUMNS) + feature_columns]
    frame["user_id"] = frame["user_id"].astype(str)
    parsed_time = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if parsed_time.notna().all():
        frame["timestamp"] = parsed_time
    frame = frame.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def load_stream(
    path: str | Path,
    legacy_global_scaling: bool = False,
    no_scaling: bool = False,
) -> tuple[List[StreamEvent], List[str]]:
    frame = _read_table(Path(path))
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Canonical stream is missing columns: {missing}")
    frame = frame.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)
    feature_columns = [name for name in frame.columns if name not in REQUIRED_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found")
    matrix = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if legacy_global_scaling and no_scaling:
        raise ValueError("legacy_global_scaling and no_scaling are mutually exclusive")
    if no_scaling:
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    elif legacy_global_scaling:
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        matrix = (np.nan_to_num(matrix, nan=means) - means) / np.maximum(stds, 1e-6)
        matrix = np.clip(matrix, -10.0, 10.0).astype(np.float32)
    else:
        scaler = OnlineStandardizer(matrix.shape[1])
        matrix = np.stack([scaler.transform_then_update(row) for row in matrix])
    events = [
        StreamEvent(
            index=i,
            user_id=str(frame.at[i, "user_id"]),
            timestamp=str(frame.at[i, "timestamp"]),
            label=int(frame.at[i, "label"]),
            features=matrix[i],
        )
        for i in range(len(frame))
    ]
    return events, feature_columns


def synthetic_stream(seed: int = 42, n_users: int = 12, events_per_user: int = 24,
                     n_features: int = 12) -> tuple[List[StreamEvent], List[str]]:
    rng = np.random.default_rng(seed)
    user_groups = rng.integers(0, 3, size=n_users)
    group_weights = rng.normal(0.0, 0.8, size=(3, n_features))
    rows = []
    for step in range(events_per_user):
        order = rng.permutation(n_users)
        for user in order:
            x = rng.normal(size=n_features) + 0.25 * user_groups[user]
            logit = float(x @ group_weights[user_groups[user]] / np.sqrt(n_features))
            logit += float(rng.normal(scale=0.4))
            y = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            rows.append(StreamEvent(
                index=len(rows), user_id=f"u{user:03d}", timestamp=str(len(rows)),
                label=y, features=x.astype(np.float32),
            ))
    return rows, [f"x{i}" for i in range(n_features)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a dataset to SG-Share canonical stream format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-column", required=True)
    parser.add_argument("--time-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-values", default="1,true,positive,yes")
    args = parser.parse_args()
    frame = prepare_stream(
        args.input, args.output, args.user_column, args.time_column, args.label_column,
        [value.strip() for value in args.positive_values.split(",") if value.strip()],
    )
    print(f"wrote {len(frame)} rows and {len(frame.columns) - 3} features to {args.output}")


if __name__ == "__main__":
    main()
