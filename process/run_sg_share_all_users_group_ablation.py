from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import ExperimentConfig, apply_variant, load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner


VARIANTS = (
    "full_final_reference",
    "all_users_grouped",
    "all_users_grouped_no_mature",
)


def _csv_items(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in _csv_items(value)]


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


def _configured(dataset: str, variant: str, seed: int) -> ExperimentConfig:
    cfg = apply_variant(load_config(dataset=dataset), "full_final")
    cfg.seed = int(seed)

    # Use the classification-tuned route-gradient settings confirmed for each
    # dataset. The ablation below changes admission/maturity only.
    cfg.training.route_grad_ema_decay = 0.80
    cfg.training.route_grad_normalize_before_ema = False
    cfg.grouping.k_min = 4
    cfg.grouping.min_observations = 20 if dataset == "ces" else 2

    if variant in {"all_users_grouped", "all_users_grouped_no_mature"}:
        # A user can be grouped at the first boundary after its first observed
        # label. A pre-label user has no route-gradient signature and therefore
        # cannot be assigned without leaking future feedback.
        cfg.grouping.min_observations = 1
        cfg.grouping.warmup_initial_min_samples = 1
        cfg.grouping.warmup_initial_min_samples_fallback = 1

    if variant == "all_users_grouped_no_mature":
        cfg.refinements.mature_refine = False

    return cfg


def _cold_at(metrics: dict[str, Any], k: int) -> dict[str, float]:
    for row in metrics.get("cold_start", []):
        if int(row["k"]) == int(k):
            return row
    return {}


def _summary_row(
    dataset: str, variant: str, seed: int, metrics: dict[str, Any],
) -> dict[str, Any]:
    overall = metrics["overall"]
    post = metrics["post_warmup"]
    row: dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "f1": overall["f1"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "specificity": overall["specificity"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "accuracy": overall["accuracy"],
        "post_f1": post["f1"],
        "post_recall": post["recall"],
        "post_specificity": post["specificity"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "n_groups_final": metrics["n_groups_final"],
        "mean_groups": metrics["mean_groups"],
    }
    for k in (5, 10, 20, 50):
        cold = _cold_at(metrics, k)
        row[f"cold{k}_macro_f1"] = cold.get("user_macro_f1", np.nan)
        row[f"cold{k}_macro_recall"] = cold.get("user_macro_recall", np.nan)
        row[f"cold{k}_pooled_f1"] = cold.get("pooled_f1", np.nan)
        row[f"cold{k}_pooled_recall"] = cold.get("pooled_recall", np.nan)
    return row


def _run_one(
    dataset: str,
    data: str | Path,
    output: Path,
    variant: str,
    seed: int,
    stream,
    feature_count: int,
    skip_existing: bool,
) -> dict[str, Any]:
    run_dir = output / dataset / f"{variant}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return _summary_row(dataset, variant, seed, metrics)

    cfg = _configured(dataset, variant, seed)
    learner = SGShareLearner(feature_count, cfg, "full_final", "gradient")
    metrics = learner.run(stream)
    metrics.update({
        "dataset": dataset,
        "method": "full_final",
        "variant": variant,
        "seed": int(seed),
        "data": str(data),
    })

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8",
    )
    pd.DataFrame(learner.events).to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(metrics["cold_start"]).to_csv(run_dir / "cold_start.csv", index=False)
    pd.DataFrame(learner.group_trace).to_csv(run_dir / "groups.csv", index=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8",
    )
    return _summary_row(dataset, variant, seed, metrics)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column for column in frame.columns
        if column not in {"dataset", "variant", "seed"}
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, variant), group in frame.groupby(["dataset", "variant"], sort=False):
        row: dict[str, Any] = {
            "dataset": dataset,
            "variant": variant,
            "n_seeds": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_delta(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        column for column in frame.columns
        if column not in {"dataset", "variant", "seed"}
    ]
    for dataset in frame["dataset"].unique():
        subset = frame[frame["dataset"] == dataset]
        reference = subset[
            subset["variant"] == "full_final_reference"
        ].set_index("seed")
        for variant in VARIANTS[1:]:
            candidate = subset[subset["variant"] == variant].set_index("seed")
            for seed in sorted(set(reference.index) & set(candidate.index)):
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": int(seed),
                }
                for metric in metrics:
                    row[f"delta_{metric}"] = (
                        float(candidate.loc[seed, metric])
                        - float(reference.loc[seed, metric])
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def run(
    datasets: Sequence[str],
    data_paths: dict[str, str],
    output: str | Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    skip_existing: bool,
    torch_threads: int,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for dataset in datasets:
        stream, feature_names = load_stream(
            data_paths[dataset], legacy_global_scaling=False, no_scaling=True,
        )
        for seed in seeds:
            for variant in variants:
                print(f"running {dataset}/{variant}/seed{seed}", flush=True)
                try:
                    row = _run_one(
                        dataset, data_paths[dataset], root, variant, int(seed),
                        stream, len(feature_names), skip_existing,
                    )
                    rows.append(row)
                    print(
                        f"done F1={row['f1']:.4f} post={row['post_f1']:.4f} "
                        f"spec={row['specificity']:.4f} "
                        f"cold5={row['cold5_macro_f1']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    errors.append({
                        "dataset": dataset,
                        "variant": variant,
                        "seed": int(seed),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print(
                        f"ERROR {dataset}/{variant}/seed{seed}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary_by_seed.csv", index=False)
    if not frame.empty:
        _aggregate(frame).to_csv(root / "summary_mean_std.csv", index=False)
        _paired_delta(frame).to_csv(root / "paired_delta.csv", index=False)
    pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)
    payload = {
        "datasets": list(datasets),
        "data": data_paths,
        "variants": list(variants),
        "seeds": [int(seed) for seed in seeds],
        "n_runs": len(rows),
        "n_errors": len(errors),
        "ablation": {
            "full_final_reference": "classification-tuned full_final",
            "all_users_grouped": (
                "first-label admission; mature refinement retained"
            ),
            "all_users_grouped_no_mature": (
                "first-label admission; mature refinement disabled"
            ),
        },
    }
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ablate staged admission by grouping every observed user",
    )
    parser.add_argument("--datasets", default="ces,globem")
    parser.add_argument(
        "--ces-data", default="data/processed/ces_historical_usernorm.csv",
    )
    parser.add_argument("--globem-data", default="data/processed/globem_full.csv")
    parser.add_argument(
        "--output", default="results/sg_share_all_users_group_ablation_20260721",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    datasets = _csv_items(args.datasets)
    variants = _csv_items(args.variants)
    unknown_datasets = sorted(set(datasets) - {"ces", "globem"})
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    data_paths = {"ces": args.ces_data, "globem": args.globem_data}
    payload = run(
        datasets, data_paths, args.output, variants, _csv_ints(args.seeds),
        args.skip_existing, args.torch_threads,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
