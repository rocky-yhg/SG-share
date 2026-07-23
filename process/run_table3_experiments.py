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

from process.run_sg_share_parameter_search import SearchPoint
from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.metrics import binary_metrics
from sgshare.online_sota import OnlineSOTALearner
from sgshare.presets import apply_named_preset


BASELINES = ("hbp", "koil", "olfl", "oli2ds", "olifl")
TABLE3_KS = {"ces": (5, 10, 20), "globem": (5, 10)}
# Compatibility center for older diagnostic runners. Table 3 itself resolves
# the `cold_safe` named preset below.
ECAP_POINTS = {
    "ces": SearchPoint(0.20, 20, 5, False, "classification_tuned", False),
    "globem": SearchPoint(0.10, 2, 5, False, "classification_tuned", False),
}
METRICS = ("accuracy", "precision_pos", "recall_pos", "f1_pos", "auc")


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


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = sorted_ranks
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _first_k_rows(events: pd.DataFrame, ks: Sequence[int]) -> list[dict[str, float]]:
    required = {
        "event_index", "user_id", "label", "prediction", "probability",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events are missing columns: {sorted(missing)}")

    users = {
        str(user_id): rows.sort_values("event_index")
        for user_id, rows in events.groupby("user_id", sort=False)
    }
    output: list[dict[str, float]] = []
    for k in ks:
        values = {metric: [] for metric in METRICS}
        included_users = 0
        auc_users = 0
        for rows in users.values():
            if len(rows) < k:
                continue
            subset = rows.iloc[:k]
            labels = subset["label"].astype(int).to_numpy()
            probabilities = subset["probability"].astype(float).to_numpy()
            predictions = subset["prediction"].astype(int).to_numpy()
            binary = binary_metrics(labels, predictions)
            values["accuracy"].append(float(binary["accuracy"]))
            values["precision_pos"].append(float(binary["precision"]))
            values["recall_pos"].append(float(binary["recall"]))
            values["f1_pos"].append(float(binary["f1"]))
            auc = _auc(labels, probabilities)
            if np.isfinite(auc):
                values["auc"].append(float(auc))
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


def _run_one(
    dataset: str,
    method: str,
    seed: int,
    stream,
    feature_count: int,
    output: Path,
    device: str,
    resume: bool,
) -> pd.DataFrame:
    run_dir = output / f"{method}_seed{seed}"
    events_path = run_dir / "events.csv"
    if resume and events_path.exists():
        print(f"reusing {events_path}", flush=True)
        return pd.read_csv(events_path)

    config = load_config(dataset=dataset)
    config.seed = int(seed)
    if method == "ecap":
        config, _ = apply_named_preset(config, "cold_safe")
        config.seed = int(seed)
        config.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ECAP requested CUDA, but no CUDA device is available")
        learner = SGShareLearner(feature_count, config, "full_final", "gradient")
    else:
        learner = OnlineSOTALearner(feature_count, config, method, "per_user")

    print(
        f"running dataset={dataset} method={method} seed={seed} "
        f"events={len(stream)} device={config.device}",
        flush=True,
    )
    metrics = learner.run(stream)
    events = pd.DataFrame(learner.events)
    run_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_path, index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"completed dataset={dataset} method={method} seed={seed}", flush=True)
    return events


def _write_summaries(rows: list[dict[str, Any]], output: Path) -> None:
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "table3_by_seed.csv", index=False)
    if by_seed.empty:
        return
    grouped = by_seed.groupby(["dataset", "method", "k"], sort=False, dropna=False)
    parts = [grouped.size().rename("n_seeds")]
    for column in ("n_users", "n_auc_users", *METRICS):
        parts.append(grouped[column].mean().rename(f"{column}_mean"))
        parts.append(grouped[column].std(ddof=0).fillna(0.0).rename(f"{column}_std"))
    pd.concat(parts, axis=1).reset_index().to_csv(
        output / "table3_mean_std.csv", index=False
    )


def run(
    dataset: str,
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    methods: Sequence[str],
    device: str,
    torch_threads: int,
    resume: bool,
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            events = _run_one(
                dataset, method, seed, stream, len(feature_names), output_path, device, resume
            )
            for metric_row in _first_k_rows(events, TABLE3_KS[dataset]):
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": int(seed),
                    **metric_row,
                })
            _write_summaries(rows, output_path)

    manifest = {
        "dataset": dataset,
        "data": str(data),
        "n_events": len(stream),
        "n_features": len(feature_names),
        "methods": list(methods),
        "seeds": [int(seed) for seed in seeds],
        "ks": list(TABLE3_KS[dataset]),
        "prediction": "each method's recorded configured deployment prediction",
        "aggregation": "mean of per-user positive-class first-K metrics",
        "auc": "mean across users whose first-K labels contain both classes",
        "ecap_device": device,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _csv_values(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rerun the Table 3 early-monitoring experiments")
    parser.add_argument("--dataset", required=True, choices=sorted(TABLE3_KS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--methods", default=",".join((*BASELINES, "ecap")))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    methods = _csv_values(args.methods)
    unknown = sorted(set(methods) - set((*BASELINES, "ecap")))
    if unknown:
        raise ValueError(f"unknown Table 3 methods: {unknown}")
    run(
        args.dataset,
        args.data,
        args.output,
        [int(seed) for seed in _csv_values(args.seeds)],
        methods,
        args.device,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
