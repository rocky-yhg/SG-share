from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .experiment import run_experiment


DEFAULT_METHODS = [
    "global_shared",
    "one_group_adapter",
    "per_user_adapter",
    "feature_grouping",
    "random_grouping",
    "clean",
    "full_final",
    "pdfk",
    "budgeted",
    "supermask",
]


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    overall = metrics["overall"]
    post = metrics["post_warmup"]
    return {
        "dataset": metrics["dataset"],
        "method": metrics["method"],
        "seed": metrics["seed"],
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "specificity": overall["specificity"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "f1": overall["f1"],
        "post_f1": post["f1"],
        "mean_first_k_f1": metrics["mean_first_k_f1"],
        "n_events": metrics["n_events"],
        "n_users": metrics["n_users"],
    }


def compare_to_paper(summary: pd.DataFrame, expected_path: Path) -> pd.DataFrame:
    expected = pd.read_csv(expected_path)
    aliases = {
        "clean": "clean",
        "full_final": "full_final",
        "feature_grouping": "feature_grouping",
        "random_grouping": "random_grouping",
        "global_shared": "best_global",
        "one_group_adapter": "one_group_adapter",
        "per_user_adapter": "per_user_adapter",
    }
    rows: List[Dict[str, Any]] = []
    for _, reproduced in summary.iterrows():
        expected_method = aliases.get(str(reproduced["method"]))
        if expected_method is None:
            continue
        match = expected[
            (expected["dataset"] == reproduced["dataset"])
            & (expected["method"] == expected_method)
        ]
        if match.empty:
            continue
        reference = match.iloc[0]
        row: Dict[str, Any] = {
            "dataset": reproduced["dataset"],
            "method": reproduced["method"],
            "expected_method": expected_method,
        }
        for metric in ("accuracy", "recall", "specificity", "f1", "post_f1", "mean_first_k_f1"):
            row[f"reproduced_{metric}"] = reproduced.get(metric, np.nan)
            row[f"paper_{metric}"] = reference.get(metric, np.nan)
            row[f"delta_{metric}"] = reproduced.get(metric, np.nan) - reference.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def reproduce(
    datasets: Iterable[str], methods: Iterable[str], data_root: Path, output_root: Path,
    seeds: Iterable[int], legacy_global_scaling: bool, no_scaling: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        if dataset == "synthetic":
            data_path = None
        else:
            candidates = [data_root / f"{dataset}.parquet", data_root / f"{dataset}.csv", data_root / f"{dataset}.arff"]
            data_path = next((path for path in candidates if path.exists()), None)
            if data_path is None:
                missing.append(dataset)
                continue
        for method in methods:
            for seed in seeds:
                run_dir = output_root / dataset / method / f"seed_{seed}"
                metrics = run_experiment(
                    dataset, method, run_dir, data_path=data_path, seed=seed,
                    legacy_global_scaling=legacy_global_scaling,
                    no_scaling=no_scaling,
                )
                rows.append(flatten_metrics(metrics))
    output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    if not summary.empty:
        numeric = [
            "accuracy", "precision", "recall", "specificity", "balanced_accuracy",
            "f1", "post_f1", "mean_first_k_f1",
        ]
        aggregates = summary.groupby(["dataset", "method"])[numeric].agg(["mean", "std"]).reset_index()
        aggregates.to_csv(output_root / "summary_mean_std.csv", index=False)
        expected = Path(__file__).resolve().parents[1] / "expected" / "paper_results.csv"
        comparison = compare_to_paper(summary, expected)
        comparison.to_csv(output_root / "paper_comparison.csv", index=False)
    if missing:
        message = (
            "The following canonical streams were not found and were not run: "
            + ", ".join(sorted(set(missing)))
            + ". Place <dataset>.csv or <dataset>.parquet in the configured data root.\n"
        )
        (output_root / "MISSING_DATA.md").write_text(message, encoding="utf-8")
    manifest = {
        "datasets_requested": list(datasets),
        "methods_requested": list(methods),
        "seeds": list(seeds),
        "missing_datasets": sorted(set(missing)),
        "completed_runs": len(rows),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SG-Share reproduction matrix")
    parser.add_argument("--datasets", default="ces,globem")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output-root", default="results/reproduction")
    parser.add_argument("--legacy-global-scaling", action="store_true")
    parser.add_argument(
        "--no-scaling", action="store_true",
        help="Use features exactly as stored (required for historical CES/GLOBEM streams)",
    )
    args = parser.parse_args(argv)
    summary = reproduce(
        [value.strip() for value in args.datasets.split(",") if value.strip()],
        [value.strip() for value in args.methods.split(",") if value.strip()],
        Path(args.data_root),
        Path(args.output_root),
        [int(value) for value in args.seeds.split(",") if value.strip()],
        args.legacy_global_scaling,
        args.no_scaling,
    )
    print(f"completed {len(summary)} runs; outputs: {args.output_root}")


if __name__ == "__main__":
    main()
