from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.metrics import binary_metrics, first_k_metrics


DEFAULTS = {
    "ces": {
        "data": ROOT / "data/processed/ces_historical_usernorm.csv",
        "config": ROOT / "results/real_ces_full_final_causal_scaled_seed42/config.yaml",
        "no_scaling": True,
        "parameter_label": "alpha=0.05, lambda=20, k_min=4, legacy eligibility",
    },
    "globem": {
        "data": ROOT / "data/processed/globem_full.csv",
        "config": ROOT / (
            "results/sg_share_parameter_confirmation_globem_20260720/globem/"
            "confirm_cold_raw_a0p200_lam2_k4_strict_seed42/config.yaml"
        ),
        "no_scaling": True,
        "parameter_label": "alpha=0.20, lambda=2, k_min=4, strict eligibility",
    },
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _current_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except (ImportError, OSError):
        return float("nan")


class MemoryTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.observed_mps_peak_mb = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "accelerator": self.device.type,
            "gpu_memory_current_mb": float("nan"),
            "gpu_memory_reserved_mb": float("nan"),
            "gpu_memory_peak_mb": float("nan"),
            "gpu_memory_peak_kind": "unavailable",
            "process_rss_mb": _current_rss_mb(),
            "process_peak_rss_mb": _peak_rss_mb(),
        }
        scale = 1024.0 * 1024.0
        if self.device.type == "cuda":
            row.update({
                "gpu_memory_current_mb": torch.cuda.memory_allocated(self.device) / scale,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved(self.device) / scale,
                "gpu_memory_peak_mb": torch.cuda.max_memory_allocated(self.device) / scale,
                "gpu_memory_peak_kind": "allocator_peak",
            })
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            current = float(torch.mps.current_allocated_memory()) / scale
            driver = float(torch.mps.driver_allocated_memory()) / scale
            self.observed_mps_peak_mb = max(self.observed_mps_peak_mb, current)
            row.update({
                "gpu_memory_current_mb": current,
                "gpu_memory_reserved_mb": driver,
                "gpu_memory_peak_mb": self.observed_mps_peak_mb,
                "gpu_memory_peak_kind": "checkpoint_observed_peak",
            })
        return row


def _metric_columns(prefix: str, rows: Sequence[dict[str, object]]) -> dict[str, float]:
    values = binary_metrics(
        [int(row["label"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )
    return {
        f"{prefix}_accuracy": float(values["accuracy"]),
        f"{prefix}_precision_pos": float(values["precision"]),
        f"{prefix}_recall_pos": float(values["recall"]),
        f"{prefix}_specificity": float(values["specificity"]),
        f"{prefix}_balanced_accuracy": float(values["balanced_accuracy"]),
        f"{prefix}_f1_pos": float(values["f1"]),
    }


def _boundaries(n_events: int) -> list[int]:
    if n_events < 10:
        raise ValueError(f"At least 10 events are required, got {n_events}")
    return [int(math.ceil(n_events * decile / 10.0)) for decile in range(1, 11)]


def run_dataset(
    dataset: str,
    data_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    seed: int,
    torch_threads: int,
    no_scaling: bool,
    max_events: int | None = None,
) -> dict[str, Any]:
    dataset = dataset.lower()
    output = Path(output_root) / dataset
    output.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    stream, feature_names = load_stream(data_path, no_scaling=no_scaling)
    if max_events is not None:
        stream = stream[: max(10, int(max_events))]
    data_load_seconds = time.perf_counter() - load_started

    config = load_config(config_path, dataset)
    config.seed = int(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    init_started = time.perf_counter()
    learner = SGShareLearner(len(feature_names), config, "full_final", "gradient")
    model_init_seconds = time.perf_counter() - init_started
    memory = MemoryTracker(learner.device)

    checkpoints: list[dict[str, Any]] = []
    boundaries = _boundaries(len(stream))
    previous = 0
    cumulative_train_seconds = 0.0
    end_to_end_started = time.perf_counter()
    for decile, boundary in enumerate(boundaries, start=1):
        _sync(learner.device)
        chunk_started = time.perf_counter()
        for event in stream[previous:boundary]:
            learner.process(event)
        _sync(learner.device)
        chunk_seconds = time.perf_counter() - chunk_started
        cumulative_train_seconds += chunk_seconds

        cumulative_rows = learner.events[:boundary]
        decile_rows = learner.events[previous:boundary]
        row: dict[str, Any] = {
            "dataset": dataset,
            "decile": decile,
            "fraction": decile / 10.0,
            "start_event_1based": previous + 1,
            "end_event_1based": boundary,
            "events_in_decile": boundary - previous,
            "events_processed": boundary,
            "total_events": len(stream),
            "decile_train_seconds": chunk_seconds,
            "cumulative_train_seconds": cumulative_train_seconds,
            "decile_throughput_events_per_second": (
                (boundary - previous) / chunk_seconds if chunk_seconds > 0 else float("nan")
            ),
            "cumulative_throughput_events_per_second": (
                boundary / cumulative_train_seconds
                if cumulative_train_seconds > 0 else float("nan")
            ),
            "wall_elapsed_including_checkpoints_seconds": time.perf_counter() - end_to_end_started,
            "n_users_seen": len(learner.seen),
            "n_groups": len(learner.groups),
        }
        row.update(_metric_columns("test_cumulative", cumulative_rows))
        row.update(_metric_columns("test_decile", decile_rows))
        row.update(memory.sample())
        checkpoints.append(row)
        print(
            f"{dataset} {decile * 10:3d}% n={boundary}/{len(stream)} "
            f"time={cumulative_train_seconds:.2f}s "
            f"throughput={row['cumulative_throughput_events_per_second']:.2f}/s "
            f"F1+={row['test_cumulative_f1_pos']:.4f}",
            flush=True,
        )
        previous = boundary

    profile = pd.DataFrame(checkpoints)
    profile.to_csv(output / "decile_profile.csv", index=False)
    pd.DataFrame(learner.events).to_csv(output / "events.csv", index=False)
    pd.DataFrame(learner.group_trace).to_csv(output / "groups.csv", index=False)
    cold = first_k_metrics(
        learner.events,
        config.evaluation.first_k,
        config.evaluation.require_complete_first_k,
    )
    pd.DataFrame(cold).to_csv(output / "cold_start.csv", index=False)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    payload = {
        "dataset": dataset,
        "seed": int(seed),
        "data": str(data_path),
        "config": str(config_path),
        "device": str(learner.device),
        "torch_threads": int(torch_threads),
        "feature_count": len(feature_names),
        "n_events": len(stream),
        "n_users": len(learner.seen),
        "data_load_seconds": data_load_seconds,
        "model_init_seconds": model_init_seconds,
        "full_train_seconds": cumulative_train_seconds,
        "full_throughput_events_per_second": (
            len(stream) / cumulative_train_seconds
            if cumulative_train_seconds > 0 else float("nan")
        ),
        "final_test_metrics": {
            key.removeprefix("test_cumulative_"): value
            for key, value in checkpoints[-1].items()
            if key.startswith("test_cumulative_")
        },
        "memory": {
            key: checkpoints[-1][key]
            for key in (
                "accelerator", "gpu_memory_current_mb", "gpu_memory_reserved_mb",
                "gpu_memory_peak_mb", "gpu_memory_peak_kind", "process_rss_mb",
                "process_peak_rss_mb",
            )
        },
        "metric_protocol": "cumulative prequential predictions at each event-count decile",
        "timing_protocol": "predict+observe+update only; excludes data loading, model init, and metric aggregation",
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile SG-Share runtime, memory, and prequential metrics at ten stream deciles"
    )
    parser.add_argument("--datasets", default="ces,globem")
    parser.add_argument(
        "--output", default="results/sg_share_cold_safe_decile_profile_20260721"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args(argv)

    datasets = [part.strip().lower() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(datasets) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    errors = []
    for dataset in datasets:
        spec = DEFAULTS[dataset]
        try:
            payload = run_dataset(
                dataset=dataset,
                data_path=spec["data"],
                config_path=spec["config"],
                output_root=output,
                seed=args.seed,
                torch_threads=args.torch_threads,
                no_scaling=bool(spec["no_scaling"]),
                max_events=args.max_events,
            )
            payload["parameter_label"] = spec["parameter_label"]
            summaries.append(payload)
        except Exception as exc:
            errors.append({
                "dataset": dataset,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"ERROR {dataset}: {type(exc).__name__}: {exc}", flush=True)
    pd.DataFrame(errors).to_csv(output / "errors.csv", index=False)
    summary_path = output / "summary.json"
    prior_runs: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            prior_runs = list(json.loads(summary_path.read_text(encoding="utf-8")).get("runs", []))
        except (json.JSONDecodeError, OSError, TypeError):
            prior_runs = []
    merged_runs = {str(run.get("dataset")): run for run in prior_runs}
    merged_runs.update({str(run.get("dataset")): run for run in summaries})
    ordered_runs = [merged_runs[key] for key in sorted(merged_runs)]
    summary_path.write_text(
        json.dumps(_json_safe({
            "runs": ordered_runs,
            "n_runs": len(ordered_runs),
            "n_errors": len(errors),
            "host": platform.platform(),
            "torch_version": torch.__version__,
        }), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(f"{len(errors)} dataset run(s) failed")


if __name__ == "__main__":
    main()
