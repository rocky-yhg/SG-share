from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process.run_sg_share_parameter_search import _configured
from process.run_table3_experiments import ECAP_POINTS, TABLE3_KS, _auc
from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.metrics import binary_metrics


VARIANTS = ("standalone", "user_mean")
METRICS = ("accuracy", "precision", "recall", "macro_f1", "auc")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_positive = int(np.sum((labels != target) & (predictions == target)))
        false_negative = int(np.sum((labels == target) & (predictions != target)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _macro_precision(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_positive = int(np.sum((labels != target) & (predictions == target)))
        denominator = true_positive + false_positive
        scores.append(true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _macro_recall(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_negative = int(np.sum((labels == target) & (predictions != target)))
        denominator = true_positive + false_negative
        scores.append(true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _first_k_rows(events: pd.DataFrame, ks: Sequence[int]) -> list[dict[str, Any]]:
    users = {
        str(user_id): rows.sort_values("event_index")
        for user_id, rows in events.groupby("user_id", sort=False)
    }
    output = []
    for k in ks:
        values = {metric: [] for metric in METRICS}
        auc_users = 0
        included_users = 0
        for rows in users.values():
            if len(rows) < k:
                continue
            subset = rows.iloc[:k]
            labels = subset["label"].astype(int).to_numpy()
            probabilities = subset["raw_probability"].astype(float).to_numpy()
            predictions = (probabilities >= 0.5).astype(int)
            binary = binary_metrics(labels, predictions)
            values["accuracy"].append(binary["accuracy"])
            values["precision"].append(_macro_precision(labels, predictions))
            values["recall"].append(_macro_recall(labels, predictions))
            values["macro_f1"].append(_macro_f1(labels, predictions))
            auc = _auc(labels, probabilities)
            if np.isfinite(auc):
                values["auc"].append(auc)
                auc_users += 1
            included_users += 1
        output.append({
            "k": int(k),
            "n_users": included_users,
            "n_auc_users": auc_users,
            **{
                metric: float(np.mean(metric_values)) if metric_values else float("nan")
                for metric, metric_values in values.items()
            },
        })
    return output


def _result_rows(
    dataset: str, variant: str, seed: int, events: pd.DataFrame,
) -> list[dict[str, Any]]:
    labels = events["label"].astype(int).to_numpy()
    probabilities = events["raw_probability"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    full = binary_metrics(labels, predictions)
    rows = [{
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "scope": "full",
        "k": 0,
        "n_users": int(events["user_id"].nunique()),
        "n_auc_users": int(events["user_id"].nunique()),
        "accuracy": full["accuracy"],
        "precision": _macro_precision(labels, predictions),
        "recall": _macro_recall(labels, predictions),
        "macro_f1": _macro_f1(labels, predictions),
        "auc": _auc(labels, probabilities),
    }]
    for first_k in _first_k_rows(events, TABLE3_KS[dataset]):
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "seed": int(seed),
            "scope": "first_k",
            "k": int(first_k["k"]),
            "n_users": int(first_k["n_users"]),
            "n_auc_users": int(first_k["n_auc_users"]),
            "accuracy": first_k["accuracy"],
            "precision": first_k["precision"],
            "recall": first_k["recall"],
            "macro_f1": first_k["macro_f1"],
            "auc": first_k["auc"],
        })
    return rows


def _write_summaries(rows: list[dict[str, Any]], output: Path) -> None:
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "comparison_by_seed.csv", index=False)
    grouped = by_seed.groupby(
        ["dataset", "variant", "scope", "k"], sort=False, dropna=False,
    )
    parts = [grouped.size().rename("n_seeds")]
    for column in ("n_users", "n_auc_users", *METRICS):
        parts.append(grouped[column].mean().rename(f"{column}_mean"))
        parts.append(grouped[column].std(ddof=0).fillna(0.0).rename(f"{column}_std"))
    pd.concat(parts, axis=1).reset_index().to_csv(
        output / "comparison_mean_std.csv", index=False,
    )

    baseline = by_seed[by_seed["variant"] == "standalone"]
    changed = by_seed[by_seed["variant"] == "user_mean"]
    keys = ["dataset", "seed", "scope", "k"]
    paired = changed.merge(baseline, on=keys, suffixes=("_user_mean", "_standalone"))
    delta = paired[keys].copy()
    for metric in METRICS:
        delta[f"{metric}_standalone"] = paired[f"{metric}_standalone"]
        delta[f"{metric}_user_mean"] = paired[f"{metric}_user_mean"]
        delta[f"{metric}_delta"] = (
            paired[f"{metric}_user_mean"] - paired[f"{metric}_standalone"]
        )
    delta.to_csv(output / "paired_deltas.csv", index=False)


def run(
    dataset: str,
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    device: str,
    torch_threads: int,
    resume: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in VARIANTS:
            run_dir = output_path / f"{variant}_seed{seed}"
            events_path = run_dir / "events.csv"
            if resume and events_path.exists():
                print(f"reusing {events_path}", flush=True)
                events = pd.read_csv(events_path)
            else:
                config = _configured(load_config(dataset=dataset), ECAP_POINTS[dataset], seed)
                config.device = device
                print(
                    f"running dataset={dataset} variant={variant} seed={seed} "
                    f"events={len(stream)} device={device}",
                    flush=True,
                )
                learner = SGShareLearner(
                    len(feature_names), config, "full_final", "gradient", variant,
                )
                metrics = learner.run(stream)
                events = pd.DataFrame(learner.events)
                run_dir.mkdir(parents=True, exist_ok=True)
                events.to_csv(events_path, index=False)
                pd.DataFrame(learner.group_trace).to_csv(run_dir / "groups.csv", index=False)
                (run_dir / "metrics.json").write_text(
                    json.dumps(_json_safe(metrics), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                (run_dir / "config.json").write_text(
                    json.dumps(config.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"completed dataset={dataset} variant={variant} seed={seed}",
                    flush=True,
                )
            rows.extend(_result_rows(dataset, variant, seed, events))
            _write_summaries(rows, output_path)

    (output_path / "manifest.json").write_text(
        json.dumps({
            "dataset": dataset,
            "data": str(data),
            "n_events": len(stream),
            "n_features": len(feature_names),
            "seeds": [int(seed) for seed in seeds],
            "variants": list(VARIANTS),
            "controlled_change": (
                "standalone group adapter versus differentiable mean of current "
                "member user-adapter parameters"
            ),
            "unchanged": (
                "dataset, event order, grouping schedule, backbone update, thresholds, "
                "hyperparameters, and evaluation"
            ),
            "prediction": "raw_probability >= 0.5",
            "device": device,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare standalone and user-mean group adapters")
    parser.add_argument("--dataset", required=True, choices=sorted(TABLE3_KS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.dataset,
        args.data,
        args.output,
        [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()],
        args.device,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
