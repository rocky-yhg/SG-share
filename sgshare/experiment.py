from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import yaml

from .baselines import PortedPerUserBaseline
from .config import ExperimentConfig, apply_variant, load_config
from .data import StreamEvent, load_stream, synthetic_stream
from .learner import SGShareLearner


PORTED_METHODS = {"pdfk", "budgeted", "supermask"}


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


def resolve_sg_method(config: ExperimentConfig, method: str) -> tuple[ExperimentConfig, str, str]:
    method = method.lower()
    if method in {"full", "full_final"}:
        return apply_variant(config, "full_final"), "full_final", "gradient"
    if method in {"clean", "clean_base"}:
        return apply_variant(config, "clean"), "clean", "gradient"
    if method == "clean_plus_bias":
        return apply_variant(config, method), method, "gradient"
    if method == "clean_plus_bias_plus_split":
        return apply_variant(config, method), method, "gradient"
    if method in {"minus_bias", "minus_split", "minus_mature"}:
        return apply_variant(config, method), method, "gradient"
    if method == "feature_grouping":
        return apply_variant(config, "clean"), "clean", "feature"
    if method == "random_grouping":
        return apply_variant(config, "clean"), "clean", "random"
    if method in {"global_shared", "per_user_adapter", "one_group_adapter"}:
        return apply_variant(config, "clean"), method, "gradient"
    raise ValueError(f"Unknown method: {method}")


def run_experiment(
    dataset: str,
    method: str,
    output: str | Path,
    data_path: str | Path | None = None,
    config_path: str | Path | None = None,
    seed: int | None = None,
    legacy_global_scaling: bool = False,
    no_scaling: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path, dataset)
    if seed is not None:
        config.seed = int(seed)
    if dataset == "synthetic":
        stream, feature_names = synthetic_stream(seed=config.seed)
    else:
        if data_path is None:
            raise ValueError(f"--data is required for dataset {dataset}")
        stream, feature_names = load_stream(data_path, legacy_global_scaling, no_scaling)
    if method.lower() in PORTED_METHODS:
        learner: SGShareLearner | PortedPerUserBaseline = PortedPerUserBaseline(
            len(feature_names), config, method
        )
        resolved_config = config
    else:
        resolved_config, internal_method, signal = resolve_sg_method(config, method)
        learner = SGShareLearner(len(feature_names), resolved_config, internal_method, signal)
    metrics = learner.run(stream)
    metrics.update({
        "dataset": dataset,
        "method": method,
        "seed": resolved_config.seed,
        "feature_count": len(feature_names),
    })
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
    pd.DataFrame(learner.events).to_csv(output_path / "events.csv", index=False)
    pd.DataFrame(metrics["cold_start"]).to_csv(output_path / "cold_start.csv", index=False)
    group_trace = getattr(learner, "group_trace", [])
    pd.DataFrame(group_trace).to_csv(output_path / "groups.csv", index=False)
    with (output_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config.to_dict(), handle, sort_keys=False)
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one prequential SG-Share experiment")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "ces", "globem", "studentlife"])
    parser.add_argument("--method", default="full_final")
    parser.add_argument("--data")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--legacy-global-scaling", action="store_true")
    parser.add_argument(
        "--no-scaling", action="store_true",
        help="Use features exactly as stored (for historically pre-normalized streams)",
    )
    args = parser.parse_args(argv)
    metrics = run_experiment(
        args.dataset, args.method, args.output, args.data, args.config, args.seed,
        args.legacy_global_scaling, args.no_scaling,
    )
    overall = metrics["overall"]
    post = metrics["post_warmup"]
    print(
        f"{args.dataset}/{args.method}: F1={overall['f1']:.4f}, "
        f"recall={overall['recall']:.4f}, specificity={overall['specificity']:.4f}, "
        f"post_F1={post['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
