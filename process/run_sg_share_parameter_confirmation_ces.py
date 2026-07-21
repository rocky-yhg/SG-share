from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import load_config
from sgshare.data import load_stream
from process.run_sg_share_parameter_search import SearchPoint, _run_one


POINTS = {
    "baseline_and_safe_cold": SearchPoint(0.05, 20, 4, False, "confirm_baseline", False),
    "classification_best": SearchPoint(0.20, 20, 4, False, "confirm_class", False),
    "cold_start_unconstrained": SearchPoint(0.05, 10, 4, False, "confirm_cold", True),
}


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "f1", "recall", "specificity", "balanced_accuracy", "accuracy", "post_f1",
        "cold5_macro_f1", "cold5_macro_recall", "cold5_pooled_f1", "cold5_pooled_recall",
        "cold10_macro_f1", "cold10_macro_recall", "cold10_pooled_f1", "cold10_pooled_recall",
        "cold20_macro_f1", "cold20_macro_recall", "cold20_pooled_f1", "cold20_pooled_recall",
        "cold50_macro_f1", "cold50_macro_recall", "cold50_pooled_f1", "cold50_pooled_recall",
    ]
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("label", sort=False):
        row: dict[str, Any] = {"label": label, "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run(data: str | Path, output: str | Path, seeds: Sequence[int], labels: Sequence[str],
        skip_existing: bool, torch_threads: int) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    base = load_config(dataset="ces")
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in seeds:
        for label in labels:
            point = POINTS[label]
            print(f"running {label}/seed{seed}", flush=True)
            try:
                row = _run_one("ces", base, point, int(seed), stream, len(feature_names), root, skip_existing)
                row["label"] = label
                rows.append(row)
                print(
                    f"done F1={row['f1']:.4f} post={row['post_f1']:.4f} "
                    f"spec={row['specificity']:.4f} cold5={row['cold5_macro_f1']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "label": label, "seed": int(seed),
                    "error_type": type(exc).__name__, "error": str(exc),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary_by_seed.csv", index=False)
    if not frame.empty:
        _aggregate(frame).to_csv(root / "summary_mean_std.csv", index=False)
    paired_rows: list[dict[str, Any]] = []
    baseline = frame[frame["label"] == "baseline_and_safe_cold"].set_index("seed")
    for label in ("classification_best", "cold_start_unconstrained"):
        candidate = frame[frame["label"] == label].set_index("seed")
        for seed in sorted(set(baseline.index) & set(candidate.index)):
            paired_rows.append({
                "label": label, "seed": int(seed),
                "delta_f1": candidate.loc[seed, "f1"] - baseline.loc[seed, "f1"],
                "delta_post_f1": candidate.loc[seed, "post_f1"] - baseline.loc[seed, "post_f1"],
                "delta_specificity": candidate.loc[seed, "specificity"] - baseline.loc[seed, "specificity"],
                "delta_cold5_macro_f1": candidate.loc[seed, "cold5_macro_f1"] - baseline.loc[seed, "cold5_macro_f1"],
                "delta_cold10_macro_f1": candidate.loc[seed, "cold10_macro_f1"] - baseline.loc[seed, "cold10_macro_f1"],
            })
    pd.DataFrame(paired_rows).to_csv(root / "paired_delta.csv", index=False)
    pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)
    payload = {
        "data": str(data), "seeds": [int(seed) for seed in seeds], "labels": list(labels),
        "n_runs": len(rows), "n_errors": len(errors), "torch_threads": int(torch_threads),
        "points": {
            "baseline_and_safe_cold": "raw alpha=0.05, lambda=20, k=4, legacy eligibility",
            "classification_best": "raw alpha=0.20, lambda=20, k=4, legacy eligibility",
            "cold_start_unconstrained": "raw alpha=0.05, lambda=10, k=4, strict eligibility",
        },
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Confirm CES SG-Share parameter optima")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/sg_share_parameter_confirmation_ces_20260720")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--labels", default=",".join(POINTS))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    unknown = sorted(set(labels) - set(POINTS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    print(json.dumps(run(
        args.data, args.output, _csv_ints(args.seeds), labels,
        args.skip_existing, args.torch_threads,
    ), indent=2))


if __name__ == "__main__":
    main()
