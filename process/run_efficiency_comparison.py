from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import ExperimentConfig, load_config
from sgshare.data import StreamEvent, load_stream
from sgshare.learner import SGShareLearner
from sgshare.model import TinyTFTBackbone


DATASETS = {
    "ces": {
        "data": ROOT / "data/processed/ces_historical_usernorm.csv",
        "config": ROOT / "configs/ces.yaml",
    },
    "globem": {
        "data": ROOT / "data/processed/globem_full.csv",
        "config": ROOT / "configs/globem.yaml",
    },
}


class OfflineTransformer(nn.Module):
    """The shared Transformer backbone and classification head without adapters."""

    def __init__(self, input_dim: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.backbone = TinyTFTBackbone(input_dim, config.model)
        self.backbone.route_enabled = False
        self.head = nn.Linear(config.model.hidden_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.backbone(features)
        return self.head(hidden)


def _boundaries(n_events: int) -> list[int]:
    if n_events < 10:
        raise ValueError(f"At least 10 events are required, got {n_events}")
    return [int(math.ceil(n_events * decile / 10.0)) for decile in range(1, 11)]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stream_tensors(stream: Sequence[StreamEvent]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(np.stack([event.features for event in stream])).float()
    labels = torch.tensor([event.label for event in stream], dtype=torch.long)
    return features, labels


def _measure_online_intervals(
    stream: Sequence[StreamEvent],
    input_dim: int,
    config: ExperimentConfig,
    boundaries: Sequence[int],
    device: torch.device,
) -> list[float]:
    _seed_everything(config.seed)
    warmup_learner = SGShareLearner(input_dim, config, "full_final", "gradient")
    warmup_learner.process(stream[0])
    _sync(device)
    del warmup_learner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _seed_everything(config.seed)
    learner = SGShareLearner(input_dim, config, "full_final", "gradient")
    times: list[float] = []
    previous = 0
    for decile, boundary in enumerate(boundaries, start=1):
        _sync(device)
        started = time.perf_counter()
        for event in stream[previous:boundary]:
            learner.process(event)
        _sync(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(
            f"online {config.dataset} {decile * 10:3d}% "
            f"events={boundary - previous} seconds={elapsed:.4f}",
            flush=True,
        )
        previous = boundary
    return times


def _measure_offline_retraining(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: ExperimentConfig,
    boundaries: Sequence[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list[float]:
    _seed_everything(config.seed)
    warmup_model = OfflineTransformer(features.shape[1], config).to(device)
    warmup_optimizer = torch.optim.AdamW(
        warmup_model.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    warmup_features = features[: min(batch_size, len(features))].to(device)
    warmup_labels = labels[: min(batch_size, len(labels))].to(device)
    warmup_optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(
        warmup_model(warmup_features), warmup_labels
    ).backward()
    warmup_optimizer.step()
    _sync(device)
    del warmup_features, warmup_labels, warmup_optimizer, warmup_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    times: list[float] = []
    for decile, boundary in enumerate(boundaries, start=1):
        _seed_everything(config.seed)
        model = OfflineTransformer(features.shape[1], config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.training.weight_decay,
        )
        loader_generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            TensorDataset(features[:boundary], labels[:boundary]),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model.train()
        _sync(device)
        started = time.perf_counter()
        for _ in range(epochs):
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(
            f"offline-transformer {config.dataset} {decile * 10:3d}% "
            f"prefix_events={boundary} epochs={epochs} seconds={elapsed:.4f}",
            flush=True,
        )
        del loader, optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return times


def run_dataset(
    dataset: str,
    output: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    offline_hidden_dim: int,
    offline_depth: int,
    offline_heads: int,
    offline_learning_rate: float,
    torch_threads: int,
    max_events: int | None,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    stream, feature_names = load_stream(spec["data"], no_scaling=True)
    if max_events is not None:
        stream = stream[: max(10, int(max_events))]
    boundaries = _boundaries(len(stream))

    config = load_config(spec["config"], dataset)
    config.seed = seed
    config.device = str(device)
    torch.set_num_threads(max(1, torch_threads))

    features, labels = _stream_tensors(stream)
    online_times = _measure_online_intervals(
        stream, len(feature_names), config, boundaries, device
    )
    offline_config = copy.deepcopy(config)
    offline_config.model.hidden_dim = offline_hidden_dim
    offline_config.model.depth = offline_depth
    offline_config.model.num_heads = offline_heads
    offline_times = _measure_offline_retraining(
        features,
        labels,
        offline_config,
        boundaries,
        device,
        epochs,
        batch_size,
        offline_learning_rate,
    )

    rows = []
    previous = 0
    for decile, (boundary, online_seconds, offline_seconds) in enumerate(
        zip(boundaries, online_times, offline_times), start=1
    ):
        rows.append({
            "dataset": dataset.upper(),
            "decile": decile,
            "stream_progress_percent": decile * 10,
            "new_interval_events": boundary - previous,
            "prefix_events": boundary,
            "total_events": len(stream),
            "online_new_interval_seconds": online_seconds,
            "offline_transformer_retrain_seconds": offline_seconds,
        })
        previous = boundary

    frame = pd.DataFrame(rows)
    dataset_output = output / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset_output / "efficiency_profile.csv", index=False)
    summary = {
        "dataset": dataset,
        "data": str(spec["data"]),
        "config": str(spec["config"]),
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "n_events": len(stream),
        "n_features": len(feature_names),
        "offline_model": "Transformer backbone and classification head without adapters",
        "offline_epochs_per_checkpoint": epochs,
        "offline_batch_size": batch_size,
        "offline_hidden_dim": offline_hidden_dim,
        "offline_depth": offline_depth,
        "offline_heads": offline_heads,
        "offline_learning_rate": offline_learning_rate,
        "online_total_seconds": float(sum(online_times)),
        "offline_total_seconds": float(sum(offline_times)),
        "offline_over_online_total_ratio": float(sum(offline_times) / sum(online_times)),
        "timing_protocol": {
            "online": "predict, observe label, and update on only the newly arrived 10% interval",
            "offline": "initialize from scratch and train for the fixed epoch count on the complete prefix",
            "excluded": "data loading, model initialization, metric computation, and file output",
        },
    }
    (dataset_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"rows": rows, "summary": summary}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare online 10% interval updates with offline Transformer prefix retraining"
    )
    parser.add_argument("--datasets", default="ces,globem")
    parser.add_argument("--output", default="results/efficiency_transformer_v100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline-epochs", type=int, default=50)
    parser.add_argument("--offline-batch-size", type=int, default=128)
    parser.add_argument("--offline-hidden-dim", type=int, default=256)
    parser.add_argument("--offline-depth", type=int, default=1)
    parser.add_argument("--offline-heads", type=int, default=4)
    parser.add_argument("--offline-learning-rate", type=float, default=1e-2)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args(argv)

    datasets = [part.strip().lower() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    positive_settings = (
        args.offline_epochs,
        args.offline_batch_size,
        args.offline_hidden_dim,
        args.offline_depth,
        args.offline_heads,
    )
    if any(value < 1 for value in positive_settings) or args.offline_learning_rate <= 0:
        raise ValueError("offline Transformer settings must be positive")
    if args.offline_hidden_dim % args.offline_heads != 0:
        raise ValueError("offline hidden dimension must be divisible by the head count")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark must run on a V100 CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_name = torch.cuda.get_device_name(device)
    if "V100" not in gpu_name.upper():
        raise RuntimeError(f"Expected a V100 GPU, found {gpu_name}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        run_dataset(
            dataset=dataset,
            output=output,
            device=device,
            seed=args.seed,
            epochs=args.offline_epochs,
            batch_size=args.offline_batch_size,
            offline_hidden_dim=args.offline_hidden_dim,
            offline_depth=args.offline_depth,
            offline_heads=args.offline_heads,
            offline_learning_rate=args.offline_learning_rate,
            torch_threads=args.torch_threads,
            max_events=args.max_events,
        )
        for dataset in datasets
    ]
    pd.DataFrame(row for run in runs for row in run["rows"]).to_csv(
        output / "efficiency_profile.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps({"runs": [run["summary"] for run in runs]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
