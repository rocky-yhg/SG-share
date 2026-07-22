from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.online_sota import OnlineSOTALearner, SUPPORTED_ONLINE_SOTA


DEFAULT_METHODS = ("oli2ds", "obal", "hbp", "koil", "olifl", "olfl")
DEFAULT_SCOPES = ("global", "per_user")
DEFAULT_DATA = {
    "ces": "data/processed/ces_historical_usernorm.csv",
    "globem": "data/processed/globem_sample.csv",
}


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


def _csv_values(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _summary_row(dataset: str, method: str, scope: str, seed: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    overall = metrics["overall"]
    post = metrics["post_warmup"]
    return {
        "dataset": dataset, "method": method, "scope": scope, "seed": seed,
        "provenance": metrics["provenance"], "n_events": metrics["n_events"],
        "n_users": metrics["n_users"], "n_models": metrics["n_models"],
        "f1": overall["f1"], "recall": overall["recall"],
        "specificity": overall["specificity"],
        "balanced_accuracy": overall["balanced_accuracy"], "accuracy": overall["accuracy"],
        "post_f1": post["f1"], "post_recall": post["recall"],
        "post_specificity": post["specificity"],
        "post_balanced_accuracy": post["balanced_accuracy"],
    }


def _cold_rows(dataset: str, method: str, scope: str, seed: int, rows: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [{"dataset": dataset, "method": method, "scope": scope, "seed": seed, **row} for row in rows]


def _run_one(dataset: str, method: str, scope: str, seed: int, stream, feature_count: int,
             root: Path, skip_existing: bool,
             raw_fixed_05: bool) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    run_dir = root / dataset / f"{method}_{scope}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return _summary_row(dataset, method, scope, seed, metrics), _cold_rows(
            dataset, method, scope, seed, metrics["cold_start"]
        )
    config = load_config(dataset=dataset)
    config.seed = int(seed)
    if raw_fixed_05:
        config.training.smoothing_window = 1
        config.training.threshold_mode = "fixed"
        config.training.threshold_default = 0.5
        config.training.user_threshold_posrate_bias = False
    learner = OnlineSOTALearner(feature_count, config, method, scope)
    metrics = learner.run(stream)
    metrics.update({"dataset": dataset, "method": method, "scope": scope, "seed": seed})
    run_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
    pd.DataFrame(learner.events).to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(metrics["cold_start"]).to_csv(run_dir / "cold_start.csv", index=False)
    return _summary_row(dataset, method, scope, seed, metrics), _cold_rows(
        dataset, method, scope, seed, metrics["cold_start"]
    )


def run_suite(datasets: Sequence[str], methods: Sequence[str], scopes: Sequence[str],
              seeds: Sequence[int], output: str | Path,
              data_overrides: Dict[str, str] | None = None,
              max_events: int | None = None, skip_existing: bool = False,
              no_scaling: bool = False,
              raw_fixed_05: bool = False) -> Dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    data_paths = dict(DEFAULT_DATA)
    data_paths.update(data_overrides or {})
    summary_rows: list[Dict[str, Any]] = []
    cold_rows: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    for dataset in datasets:
        data_path = data_paths.get(dataset)
        if not data_path:
            raise ValueError(f"No data path configured for {dataset}")
        # Historical CES/GLOBEM streams already store normalized feature
        # values. The explicit no-scaling mode lets every method consume the
        # exact same stored representation as SG-Share.
        stream, feature_names = load_stream(
            data_path, legacy_global_scaling=False, no_scaling=no_scaling,
        )
        if max_events is not None:
            stream = stream[: int(max_events)]
        for seed in seeds:
            for method in methods:
                for scope in scopes:
                    print(f"running {dataset}/{method}/{scope}/seed{seed} n={len(stream)}", flush=True)
                    try:
                        summary, cold = _run_one(dataset, method, scope, int(seed), stream,
                                                 len(feature_names), root, skip_existing,
                                                 raw_fixed_05)
                        summary_rows.append(summary)
                        cold_rows.extend(cold)
                        print(f"done F1={summary['f1']:.4f} post={summary['post_f1']:.4f} "
                              f"recall={summary['recall']:.4f} spec={summary['specificity']:.4f}", flush=True)
                    except Exception as exc:
                        errors.append({"dataset": dataset, "method": method, "scope": scope,
                                       "seed": int(seed), "error_type": type(exc).__name__,
                                       "error": str(exc)})
                        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
    pd.DataFrame(summary_rows).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(cold_rows).to_csv(root / "cold_start.csv", index=False)
    pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)
    payload = {
        "n_runs": len(summary_rows), "n_errors": len(errors), "datasets": list(datasets),
        "methods": list(methods), "scopes": list(scopes), "seeds": [int(seed) for seed in seeds],
        "common_protocol": {
            "prediction_order": "predict_then_observe_then_update",
            "preprocessing": (
                "stored historical normalized values shared by all methods"
                if no_scaling else
                "single causal online standardizer shared by all methods"
            ),
            "primary_metric": "positive-class F1",
            "cold_start": "first-K user-macro and pooled positive-class F1",
            "prediction": (
                "raw_probability >= 0.5; no smoothing or threshold search"
                if raw_fixed_05 else "configured online prediction path"
            ),
        },
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run global and per-user online-learning baselines")
    parser.add_argument("--datasets", default="ces")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--scopes", default=",".join(DEFAULT_SCOPES))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output", default="results/online_sota_suite_20260720")
    parser.add_argument("--ces-data")
    parser.add_argument("--globem-data")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--raw-fixed-05", action="store_true",
        help="Use raw probabilities, threshold 0.5, no smoothing or threshold search.",
    )
    parser.add_argument(
        "--no-scaling", action="store_true",
        help="Use stored feature values without applying the online standardizer.",
    )
    args = parser.parse_args(argv)
    methods = _csv_values(args.methods)
    unknown = sorted(set(methods) - SUPPORTED_ONLINE_SOTA)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    scopes = _csv_values(args.scopes)
    if not set(scopes) <= {"global", "per_user"}:
        raise ValueError("Scopes must be global and/or per_user")
    overrides = {key: value for key, value in {
        "ces": args.ces_data, "globem": args.globem_data
    }.items() if value}
    payload = run_suite(
        _csv_values(args.datasets), methods, scopes,
        [int(value) for value in _csv_values(args.seeds)], args.output,
        overrides, args.max_events, args.skip_existing, args.no_scaling,
        args.raw_fixed_05,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
