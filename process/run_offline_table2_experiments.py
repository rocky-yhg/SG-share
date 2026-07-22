from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process.run_efficiency_comparison import OfflineTransformer
from sgshare.config import load_config
from sgshare.data import load_stream


METHODS = ("transformer", "tcn", "mlp", "xgboost", "svm", "lr")


class OfflineMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class OfflineTCN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.unsqueeze(1))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(
            labels, predictions, average="macro", zero_division=0,
        )),
        "recall": float(recall_score(
            labels, predictions, average="macro", zero_division=0,
        )),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0,
        )),
        "auc": float(roc_auc_score(labels, scores)),
    }


def _class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    weights = len(labels) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_torch(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> np.ndarray:
    _seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss(weight=_class_weights(train_y, device))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_y).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss_function(model(features), labels).backward()
            optimizer.step()

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test_x), batch_size):
            features = torch.from_numpy(test_x[start:start + batch_size]).float().to(device)
            probabilities = torch.softmax(model(features), dim=1)[:, 1]
            outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs)


def _run_model(
    dataset: str,
    method: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> np.ndarray:
    if method == "lr":
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=8,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1]
    if method == "svm":
        model = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        model.fit(train_x, train_y)
        return model.decision_function(test_x)
    if method == "xgboost":
        negatives = max(1, int(np.sum(train_y == 0)))
        positives = max(1, int(np.sum(train_y == 1)))
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=8,
            random_state=seed,
            scale_pos_weight=negatives / positives,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1]
    if method == "mlp":
        model: nn.Module = OfflineMLP(train_x.shape[1])
    elif method == "tcn":
        model = OfflineTCN(train_x.shape[1])
    elif method == "transformer":
        config = copy.deepcopy(load_config(dataset=dataset))
        model = OfflineTransformer(train_x.shape[1], config)
    else:
        raise ValueError(f"unknown offline method: {method}")
    return _train_torch(
        model,
        train_x,
        train_y,
        test_x,
        device,
        seed,
        epochs,
        batch_size,
        learning_rate,
        1e-4,
    )


def _write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "offline_by_seed.csv", index=False)
    if frame.empty:
        return
    metrics = ("accuracy", "precision", "recall", "macro_f1", "auc")
    grouped = frame.groupby(["dataset", "method"], sort=False, dropna=False)
    parts = [grouped.size().rename("n_seeds")]
    for metric in metrics:
        parts.append(grouped[metric].mean().rename(f"{metric}_mean"))
        parts.append(grouped[metric].std(ddof=0).fillna(0.0).rename(f"{metric}_std"))
    pd.concat(parts, axis=1).reset_index().to_csv(
        output / "offline_mean_std.csv", index=False
    )


def run(
    datasets: Sequence[str],
    data_paths: dict[str, str],
    output: str | Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    train_fraction: float,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    torch_threads: int,
    resume: bool,
) -> None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("offline experiments require an allocated CUDA device")
    torch.set_num_threads(max(1, int(torch_threads)))
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for dataset in datasets:
        stream, feature_names = load_stream(
            data_paths[dataset], legacy_global_scaling=False, no_scaling=True,
        )
        features = np.stack([event.features for event in stream]).astype(np.float32)
        labels = np.asarray([event.label for event in stream], dtype=np.int64)
        boundary = int(len(stream) * train_fraction)
        if boundary <= 0 or boundary >= len(stream):
            raise ValueError(f"invalid train boundary {boundary} for {dataset}")
        train_x, test_x = features[:boundary], features[boundary:]
        train_y, test_y = labels[:boundary], labels[boundary:]
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            raise ValueError(f"chronological split for {dataset} lacks both classes")
        manifests.append({
            "dataset": dataset,
            "data": data_paths[dataset],
            "n_events": len(stream),
            "n_features": len(feature_names),
            "train_events": len(train_y),
            "test_events": len(test_y),
            "train_positive_rate": float(np.mean(train_y)),
            "test_positive_rate": float(np.mean(test_y)),
        })
        for seed in seeds:
            for method in methods:
                run_dir = output_path / dataset / f"{method}_seed{seed}"
                predictions_path = run_dir / "predictions.csv"
                if resume and predictions_path.exists():
                    prediction_frame = pd.read_csv(predictions_path)
                    scores = prediction_frame["score"].astype(float).to_numpy()
                    predictions = prediction_frame["prediction"].astype(int).to_numpy()
                else:
                    print(
                        f"running dataset={dataset} method={method} seed={seed} "
                        f"train={len(train_y)} test={len(test_y)}",
                        flush=True,
                    )
                    scores = _run_model(
                        dataset, method, seed, train_x, train_y, test_x,
                        torch_device, epochs, batch_size, learning_rate,
                    )
                    threshold = 0.0 if method == "svm" else 0.5
                    predictions = (scores >= threshold).astype(int)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "label": test_y,
                        "prediction": predictions,
                        "score": scores,
                    }).to_csv(predictions_path, index=False)
                    print(f"completed dataset={dataset} method={method} seed={seed}", flush=True)
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": int(seed),
                    **_evaluate(test_y, predictions, scores),
                })
                _write_summary(rows, output_path)

    manifest = {
        "protocol": "chronological first 80% train, final 20% frozen evaluation",
        "train_fraction": train_fraction,
        "methods": list(methods),
        "seeds": [int(seed) for seed in seeds],
        "metrics": [
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "roc_auc_from_continuous_scores",
        ],
        "deep_model_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "datasets": manifests,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _csv_values(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rerun the Table 2 offline models")
    parser.add_argument("--datasets", default="ces,globem")
    parser.add_argument("--ces-data", required=True)
    parser.add_argument("--globem-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    methods = _csv_values(args.methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown offline methods: {unknown}")
    run(
        _csv_values(args.datasets),
        {"ces": args.ces_data, "globem": args.globem_data},
        args.output,
        methods,
        [int(seed) for seed in _csv_values(args.seeds)],
        args.train_fraction,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
