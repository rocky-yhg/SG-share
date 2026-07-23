from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

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


VARIANTS = (
    "full",
    "without_personalization",
    "without_grouping",
    "without_reassignment",
    "feature_grouping",
    "random_grouping",
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured(dataset: str, variant: str, seed: int, device: str):
    config = load_config(dataset=dataset)
    config.seed = int(seed)
    config.device = device
    config, _ = apply_named_preset(config, "classification_tuned")
    method = "full_final"
    signal = "gradient"
    changed: list[str] = []

    if variant == "full":
        pass
    elif variant == "without_personalization":
        method = "global_shared"
        config.refinements.per_user_bias = False
        config.training.user_threshold_posrate_bias = False
        config.refinements.mature_refine = False
        changed = [
            "method=global_shared",
            "refinements.per_user_bias=false",
            "training.user_threshold_posrate_bias=false",
            "refinements.mature_refine=false",
        ]
    elif variant == "without_grouping":
        method = "per_user_adapter"
        config.refinements.mature_refine = False
        changed = ["method=per_user_adapter", "refinements.mature_refine=false"]
    elif variant == "without_reassignment":
        config.grouping.periodic_regroup_enabled = False
        config.refinements.mature_refine = False
        changed = [
            "grouping.periodic_regroup_enabled=false",
            "refinements.mature_refine=false",
        ]
    elif variant == "feature_grouping":
        signal = "feature"
        changed = ["grouping_signal=feature"]
    elif variant == "random_grouping":
        signal = "random"
        changed = ["grouping_signal=random"]
    else:
        raise ValueError(f"unknown variant: {variant}")
    return config, method, signal, changed


def _first_k(events: pd.DataFrame, k: int) -> dict[str, float]:
    f1_pos_values: list[float] = []
    recall_pos_values: list[float] = []
    f1_macro_values: list[float] = []
    recall_macro_values: list[float] = []
    for _, rows in events.groupby("user_id", sort=False):
        rows = rows.sort_values("event_index")
        if len(rows) < k:
            continue
        subset = rows.iloc[:k]
        labels = subset["label"].astype(int)
        predictions = subset["prediction"].astype(int)
        f1_pos_values.append(f1_score(labels, predictions, zero_division=0))
        recall_pos_values.append(recall_score(labels, predictions, zero_division=0))
        f1_macro_values.append(
            f1_score(labels, predictions, average="macro", zero_division=0)
        )
        recall_macro_values.append(
            recall_score(labels, predictions, average="macro", zero_division=0)
        )
    return {
        "first_k": int(k),
        "first_k_users": len(f1_pos_values),
        "first_k_user_macro_f1_pos": float(np.mean(f1_pos_values)),
        "first_k_user_macro_recall_pos": float(np.mean(recall_pos_values)),
        "first_k_user_macro_class_macro_f1": float(np.mean(f1_macro_values)),
        "first_k_user_macro_class_macro_recall": float(
            np.mean(recall_macro_values)
        ),
    }


def _summary(
    dataset: str,
    variant: str,
    seed: int,
    events: pd.DataFrame,
    changed: Sequence[str],
) -> dict[str, Any]:
    labels = events["label"].astype(int)
    predictions = events["prediction"].astype(int)
    probabilities = events["probability"].astype(float)
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "f1_pos": float(f1_score(labels, predictions, zero_division=0)),
        "recall_pos": float(recall_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        **_first_k(events, 5),
        "changed_fields": "|".join(changed),
        "n_events": len(events),
    }


def run(
    dataset: str,
    data: Path,
    output: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
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
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for variant in variants:
            run_dir = output / f"{variant}_seed{seed}"
            events_path = run_dir / "events.csv"
            config, method, signal, changed = _configured(
                dataset, variant, seed, device,
            )
            if resume and events_path.exists():
                events = pd.read_csv(events_path)
            else:
                learner = SGShareLearner(
                    len(feature_names), config, method, signal, "standalone",
                )
                metrics = learner.run(stream)
                events = pd.DataFrame(learner.events)
                run_dir.mkdir(parents=True, exist_ok=True)
                events.to_csv(events_path, index=False)
                pd.DataFrame(learner.group_trace).to_csv(
                    run_dir / "groups.csv", index=False,
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps(_json_safe(metrics), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(config.to_dict(), sort_keys=False),
                    encoding="utf-8",
                )
            rows.append(_summary(dataset, variant, seed, events, changed))
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "data": str(data),
                "data_sha256": _sha256(data),
                "variants": list(variants),
                "seeds": list(seeds),
                "device": device,
                "base_preset": "classification_tuned",
                "prediction": "configured deployment prediction",
                "first_k": (
                    "positive-class and class-macro metrics per user, then "
                    "user mean"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the component and grouping-signal ablations in main.pdf",
    )
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    run(
        args.dataset,
        args.data,
        args.output,
        variants,
        [int(seed) for seed in _csv(args.seeds)],
        args.device,
        args.torch_threads,
        args.resume,
    )


if __name__ == "__main__":
    main()
