from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from sgshare.metrics import binary_metrics


METRICS = (
    "f1",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def _sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = ("oli2ds", "hbp", "koil", "olifl", "olfl")

    ces_per_user = ROOT / "results/online_sota_ces_per_user_threshold_20260721/ces"
    ces_global = ROOT / "results/online_sota_ces_20260720/ces"
    ces_hbp = ROOT / "results/online_sota_ces_hbp19_20260720/ces"
    for method in methods:
        rows.append({
            "dataset": "ces", "method": method, "scope": "per_user", "seed": 42,
            "path": ces_per_user / f"{method}_per_user_seed42/events.csv",
        })
        global_root = ces_hbp if method == "hbp" else ces_global
        rows.append({
            "dataset": "ces", "method": method, "scope": "global", "seed": 42,
            "path": global_root / f"{method}_global_seed42/events.csv",
        })
    rows.append({
        "dataset": "ces", "method": "sg_share", "scope": "group_shared", "seed": 42,
        "path": ROOT / (
            "results/sg_share_parameter_confirmation_ces_20260720/ces/"
            "confirm_class_raw_a0p200_lam20_k4_legacy_seed42/events.csv"
        ),
    })

    globem_per_user = ROOT / "results/online_sota_globem_per_user_threshold_20260721/globem"
    globem_global = ROOT / "results/online_sota_globem_full_20260720/globem"
    for seed in (42, 43, 44):
        for method in methods:
            rows.append({
                "dataset": "globem", "method": method, "scope": "per_user", "seed": seed,
                "path": globem_per_user / f"{method}_per_user_seed{seed}/events.csv",
            })
            rows.append({
                "dataset": "globem", "method": method, "scope": "global", "seed": seed,
                "path": globem_global / f"{method}_global_seed{seed}/events.csv",
            })
        rows.append({
            "dataset": "globem", "method": "sg_share", "scope": "group_shared", "seed": seed,
            "path": ROOT / (
                "results/sg_share_parameter_confirmation_globem_20260720/globem/"
                f"confirm_class_raw_a0p200_lam2_k4_legacy_seed{seed}/events.csv"
            ),
        })
    return rows


def _raw_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"event_index", "user_id", "label", "raw_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.sort_values("event_index").copy()
    frame["prediction"] = (frame["raw_probability"].astype(float) >= 0.5).astype(int)
    return frame


def _metric_row(y: Sequence[int], prediction: Sequence[int]) -> Dict[str, float]:
    values = binary_metrics(y, prediction)
    return {metric: float(values[metric]) for metric in METRICS}


def _cold_rows(frame: pd.DataFrame, ks: Iterable[int]) -> list[dict[str, float]]:
    by_user = {
        str(uid): rows.sort_values("event_index")
        for uid, rows in frame.groupby("user_id", sort=False)
    }
    output: list[dict[str, float]] = []
    for k in sorted(set(int(value) for value in ks)):
        pooled_y: list[int] = []
        pooled_prediction: list[int] = []
        macro: Dict[str, list[float]] = defaultdict(list)
        included = 0
        for rows in by_user.values():
            if len(rows) < k:
                continue
            subset = rows.iloc[:k]
            y = subset["label"].astype(int).tolist()
            prediction = subset["prediction"].astype(int).tolist()
            values = _metric_row(y, prediction)
            for metric, value in values.items():
                macro[metric].append(value)
            pooled_y.extend(y)
            pooled_prediction.extend(prediction)
            included += 1
        pooled = _metric_row(pooled_y, pooled_prediction)
        row: dict[str, float] = {
            "k": float(k), "n_users": float(included), "n_events": float(len(pooled_y)),
        }
        for metric in METRICS:
            row[f"user_macro_{metric}"] = float(np.mean(macro[metric])) if macro[metric] else float("nan")
            row[f"pooled_{metric}"] = pooled[metric]
        output.append(row)
    return output


def _aggregate(frame: pd.DataFrame, keys: Sequence[str], values: Sequence[str]) -> pd.DataFrame:
    grouped = frame.groupby(list(keys), sort=True, dropna=False)
    parts = [grouped.size().rename("n_seeds")]
    for value in values:
        parts.append(grouped[value].mean().rename(f"{value}_mean"))
        parts.append(grouped[value].std(ddof=0).fillna(0.0).rename(f"{value}_std"))
    return pd.concat(parts, axis=1).reset_index()


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return frame.loc[:, list(columns)].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw-probability online baseline metrics")
    parser.add_argument(
        "--output", default="results/raw_online_sota_comparison_20260721",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    classification_rows: list[dict[str, object]] = []
    cold_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for source in _sources():
        path = Path(source["path"])
        try:
            frame = _raw_events(path)
            overall = _metric_row(
                frame["label"].astype(int).tolist(), frame["prediction"].astype(int).tolist(),
            )
            warmup = 300 if source["dataset"] == "ces" else 120
            post = frame.iloc[min(warmup, len(frame)):]
            post_values = _metric_row(
                post["label"].astype(int).tolist(), post["prediction"].astype(int).tolist(),
            )
            classification_rows.append({
                **{key: source[key] for key in ("dataset", "method", "scope", "seed")},
                **overall,
                **{f"post_{key}": value for key, value in post_values.items()},
                "n_events": len(frame),
            })
            ks = (5, 10, 20, 50) if source["dataset"] == "ces" else (5, 10)
            for row in _cold_rows(frame, ks):
                cold_rows.append({
                    **{key: source[key] for key in ("dataset", "method", "scope", "seed")},
                    **row,
                })
        except Exception as exc:
            errors.append({
                **{key: source[key] for key in ("dataset", "method", "scope", "seed")},
                "path": str(path), "error_type": type(exc).__name__, "error": str(exc),
            })

    classification = pd.DataFrame(classification_rows)
    cold = pd.DataFrame(cold_rows)
    classification_values = [*METRICS, *(f"post_{metric}" for metric in METRICS)]
    cold_values = [
        "n_users", "n_events",
        *(f"user_macro_{metric}" for metric in METRICS),
        *(f"pooled_{metric}" for metric in METRICS),
    ]
    classification_mean = _aggregate(
        classification, ("dataset", "method", "scope"), classification_values,
    )
    cold_mean = _aggregate(
        cold, ("dataset", "method", "scope", "k"), cold_values,
    )

    classification.to_csv(output / "classification_by_seed.csv", index=False)
    classification_mean.to_csv(output / "classification_mean_std.csv", index=False)
    cold.to_csv(output / "cold_start_by_seed.csv", index=False)
    cold_mean.to_csv(output / "cold_start_mean_std.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "errors.csv", index=False)

    report = [
        "# Raw-probability online comparison",
        "",
        "All predictions are recomputed as `raw_probability >= 0.5`. Probability",
        "smoothing and adaptive threshold selection are not used.",
    ]
    class_columns = [
        "method", "scope", "f1_mean", "precision_mean", "recall_mean",
        "specificity_mean", "accuracy_mean", "balanced_accuracy_mean", "post_f1_mean",
    ]
    cold_columns = [
        "method", "scope", "k", "n_users_mean",
        *(f"user_macro_{metric}_mean" for metric in METRICS),
        *(f"pooled_{metric}_mean" for metric in METRICS),
    ]
    for dataset in ("ces", "globem"):
        report.extend([
            "", f"## {dataset.upper()} classification", "",
            _markdown_table(classification_mean[classification_mean.dataset == dataset], class_columns),
            "", f"## {dataset.upper()} cold start", "",
            _markdown_table(cold_mean[cold_mean.dataset == dataset], cold_columns),
        ])
    (output / "raw_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"classification rows={len(classification)} cold rows={len(cold)} errors={len(errors)}")


if __name__ == "__main__":
    main()
