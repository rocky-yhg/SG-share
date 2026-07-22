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

from process.run_sg_share_parameter_search import SearchPoint, _configured
from process.run_table3_experiments import _auc, _first_k_rows
from sgshare.config import ExperimentConfig, load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.metrics import binary_metrics


VARIANTS = (
    "without_ggm",
    "without_group_refinement",
    "without_group_splitting",
    "without_user_reassignment",
)
ECAP_POINTS = {
    "ces": SearchPoint(0.20, 20, 4, False, "ablation", False),
    "globem": SearchPoint(0.20, 2, 4, False, "ablation", False),
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


def _variant_config(
    dataset: str, variant: str, seed: int, device: str,
) -> tuple[ExperimentConfig, str]:
    config = _configured(load_config(dataset=dataset), ECAP_POINTS[dataset], seed)
    config.device = device
    method = "full_final"
    if variant == "without_ggm":
        method = "per_user_adapter"
        config.refinements.per_user_bias = False
        config.refinements.verified_split = False
        config.refinements.mature_refine = False
    elif variant == "without_group_refinement":
        config.refinements.verified_split = False
        config.refinements.mature_refine = False
    elif variant == "without_group_splitting":
        config.refinements.verified_split = False
    elif variant == "without_user_reassignment":
        config.refinements.mature_refine = False
    else:
        raise ValueError(f"unknown ablation variant: {variant}")
    return config, method


def _summary_row(
    dataset: str, variant: str, seed: int, events: pd.DataFrame,
) -> dict[str, Any]:
    labels = events["label"].astype(int).to_numpy()
    probabilities = events["raw_probability"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    full = binary_metrics(labels, predictions)
    first5 = _first_k_rows(events, [5])[0]
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "f1_at_5": first5["macro_f1"],
        "accuracy": full["accuracy"],
        "precision": full["precision"],
        "recall": full["recall"],
        "macro_f1": full["f1"],
        "auc": _auc(labels, probabilities),
        "n_events": len(events),
        "n_users_at_5": first5["n_users"],
    }


def _write_summaries(rows: list[dict[str, Any]], output: Path) -> None:
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "ablation_by_seed.csv", index=False)
    if by_seed.empty:
        return
    metrics = (
        "f1_at_5", "accuracy", "precision", "recall", "macro_f1", "auc",
        "n_events", "n_users_at_5",
    )
    grouped = by_seed.groupby(["dataset", "variant"], sort=False, dropna=False)
    parts = [grouped.size().rename("n_seeds")]
    for metric in metrics:
        parts.append(grouped[metric].mean().rename(f"{metric}_mean"))
        parts.append(grouped[metric].std(ddof=0).fillna(0.0).rename(f"{metric}_std"))
    pd.concat(parts, axis=1).reset_index().to_csv(
        output / "ablation_mean_std.csv", index=False
    )


def run(
    dataset: str,
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    variants: Sequence[str],
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
        for variant in variants:
            run_dir = output_path / f"{variant}_seed{seed}"
            events_path = run_dir / "events.csv"
            groups_path = run_dir / "groups.csv"
            if resume and events_path.exists():
                print(f"reusing {events_path}", flush=True)
                events = pd.read_csv(events_path)
            else:
                config, method = _variant_config(dataset, variant, seed, device)
                print(
                    f"running dataset={dataset} variant={variant} seed={seed} "
                    f"events={len(stream)} device={device}",
                    flush=True,
                )
                learner = SGShareLearner(len(feature_names), config, method, "gradient")
                metrics = learner.run(stream)
                events = pd.DataFrame(learner.events)
                run_dir.mkdir(parents=True, exist_ok=True)
                events.to_csv(events_path, index=False)
                pd.DataFrame(learner.group_trace).to_csv(groups_path, index=False)
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
            rows.append(_summary_row(dataset, variant, seed, events))
            _write_summaries(rows, output_path)

    manifest = {
        "dataset": dataset,
        "data": str(data),
        "n_events": len(stream),
        "n_features": len(feature_names),
        "seeds": [int(seed) for seed in seeds],
        "variants": list(variants),
        "prediction": "raw_probability >= 0.5",
        "reference": "complete ECAP row is produced by the concurrent Table 3 run",
        "device": device,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _csv_values(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the four Table 5 grouping ablations")
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    variants = _csv_values(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    run(
        args.dataset,
        args.data,
        args.output,
        [int(seed) for seed in _csv_values(args.seeds)],
        variants,
        args.device,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
