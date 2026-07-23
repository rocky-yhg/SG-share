from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.presets import apply_named_preset


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


def _metrics(events: pd.DataFrame) -> dict[str, float]:
    labels = events["label"].astype(int)
    predictions = events["prediction"].astype(int)
    probabilities = events["probability"].astype(float)
    return {
        "f1_pos": float(f1_score(labels, predictions, zero_division=0)),
        "recall_pos": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }


def run(
    dataset: str,
    data: Path,
    output: Path,
    lambda_values: list[int],
    delta_values: list[float],
    delta_sweep_lambda: int,
    seed: int,
    device: str,
    torch_threads: int,
    resume: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True, exist_ok=True)
    stream, feature_names = load_stream(
        data, legacy_global_scaling=False, no_scaling=True,
    )
    torch.set_num_threads(max(1, int(torch_threads)))
    base = load_config(dataset=dataset)
    base, _ = apply_named_preset(base, "classification_tuned")
    center_delta = float(base.refinements.mature_loss_margin)
    points = {(int(value), center_delta) for value in lambda_values}
    points.update(
        (int(delta_sweep_lambda), float(value)) for value in delta_values
    )
    rows: list[dict[str, Any]] = []

    for lambda_m, delta_move in sorted(points):
        label = f"lambda{lambda_m}_delta{delta_move:g}".replace(".", "p")
        run_dir = output / label
        events_path = run_dir / "events.csv"
        config = load_config(dataset=dataset)
        config, _ = apply_named_preset(config, "classification_tuned")
        config.seed = int(seed)
        config.device = device
        config.refinements.mature_refine = True
        config.refinements.mature_min_observations = int(lambda_m)
        config.refinements.mature_loss_margin = float(delta_move)

        if resume and events_path.exists():
            events = pd.read_csv(events_path)
            groups = pd.read_csv(run_dir / "groups.csv")
        else:
            learner = SGShareLearner(
                len(feature_names), config, "full_final", "gradient", "standalone",
            )
            learner.run(stream)
            events = pd.DataFrame(learner.events)
            groups = pd.DataFrame(learner.group_trace)
            run_dir.mkdir(parents=True, exist_ok=True)
            events.to_csv(events_path, index=False)
            groups.to_csv(run_dir / "groups.csv", index=False)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config.to_dict(), sort_keys=False),
                encoding="utf-8",
            )

        evaluated = int(
            groups.get(
                "reassignment_evaluated_users", pd.Series(dtype=float),
            ).fillna(0).sum()
        )
        moves = int(
            groups.get(
                "reassignment_moves", pd.Series(dtype=float),
            ).fillna(0).sum()
        )
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "lambda_m": int(lambda_m),
            "delta_move": float(delta_move),
            **_metrics(events),
            "evaluated_users": evaluated,
            "moves": moves,
            "move_rate": float(moves / evaluated) if evaluated else 0.0,
        }
        rows.append(row)
        pd.DataFrame(rows).sort_values(
            ["lambda_m", "delta_move"],
        ).to_csv(output / "sensitivity_summary.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps(_json_safe(row), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"{dataset} lambda={lambda_m} delta={delta_move:g} "
            f"F1={row['f1_pos']:.4f} moves={moves}/{evaluated}",
            flush=True,
        )


def _ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reassignment sensitivity reported in main.pdf",
    )
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lambda-values", required=True)
    parser.add_argument("--delta-values", default="0,0.001,0.005,0.01,0.02")
    parser.add_argument("--delta-sweep-lambda", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", default=8, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.data,
        args.output,
        _ints(args.lambda_values),
        _floats(args.delta_values),
        args.delta_sweep_lambda,
        args.seed,
        args.device,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
