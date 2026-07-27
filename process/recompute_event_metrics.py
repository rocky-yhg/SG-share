from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_score(function, *args, **kwargs) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        frequency = float(np.mean(labels[mask]))
        error += float(np.mean(mask)) * abs(confidence - frequency)
    return error


def _metrics(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["prediction"].to_numpy(dtype=int)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    has_both_classes = np.unique(labels).size == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced_accuracy = balanced_accuracy_score(labels, predictions)
        cohen_kappa = cohen_kappa_score(labels, predictions)
    return {
        "n": int(len(frame)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_pos": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_pos": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_pos": float(f1_score(labels, predictions, zero_division=0)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "cohen_kappa": float(cohen_kappa),
        "roc_auc": (
            _safe_score(roc_auc_score, labels, probabilities)
            if has_both_classes
            else float("nan")
        ),
        "average_precision": (
            _safe_score(average_precision_score, labels, probabilities)
            if has_both_classes
            else float("nan")
        ),
        "log_loss": _safe_score(
            log_loss, labels, probabilities, labels=[0, 1]
        ),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_10": _expected_calibration_error(labels, probabilities),
    }


def _first_k(
    events: pd.DataFrame, ks: list[int], probability_column: str
) -> list[dict[str, object]]:
    sort_columns = [
        column
        for column in ("user_event_index", "event_index")
        if column in events.columns
    ]
    ordered = events.sort_values(["user_id", *sort_columns])
    rows: list[dict[str, object]] = []
    user_counts = ordered.groupby("user_id", sort=False).size()
    for k in ks:
        eligible_users = user_counts.index[user_counts >= k]
        eligible = ordered[ordered["user_id"].isin(eligible_users)]
        selected = eligible.groupby("user_id", sort=False).head(k)
        user_metrics = [
            _metrics(group, probability_column)
            for _, group in selected.groupby("user_id", sort=False)
        ]
        keys = (
            "accuracy",
            "precision_pos",
            "recall_pos",
            "specificity",
            "f1_pos",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "mcc",
            "roc_auc",
            "average_precision",
            "brier_score",
        )
        rows.append(
            {
                "k": k,
                "n_users": len(user_metrics),
                "n_events": len(selected),
                "user_macro": {
                    key: float(np.nanmean([item[key] for item in user_metrics]))
                    for key in keys
                },
                "pooled": _metrics(selected, probability_column),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute extended metrics from a retained event file."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", default="5,10")
    parser.add_argument(
        "--probability-column",
        default="probability",
        choices=("probability", "raw_probability"),
    )
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    required = {"user_id", "label", "prediction", args.probability_column}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"missing required event columns: {sorted(missing)}")
    ks = sorted({int(value) for value in args.ks.split(",") if value.strip()})
    result = {
        "source": str(args.events),
        "probability_column": args.probability_column,
        "prediction_protocol": "stored configured predictions",
        "full_stream": _metrics(events, args.probability_column),
        "first_k": _first_k(events, ks, args.probability_column),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
