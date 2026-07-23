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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.presets import PRESETS, apply_named_preset


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run(
    dataset: str,
    preset_name: str,
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    device: str,
    torch_threads: int,
) -> None:
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "preset": preset_name,
        "data": str(data),
        "data_sha256": _sha256(data),
        "n_events": len(stream),
        "n_features": len(feature_names),
        "seeds": [int(seed) for seed in seeds],
        "device": device,
        "group_adapter_mode": "standalone",
    }
    for seed in seeds:
        config = load_config(dataset=dataset)
        config.seed = int(seed)
        config.device = device
        config, preset = apply_named_preset(config, preset_name)
        learner = SGShareLearner(
            len(feature_names), config, "full_final", "gradient", "standalone",
        )
        metrics = learner.run(stream)
        run_dir = root / f"{dataset}_{preset.name}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(_json_safe(metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pd.DataFrame(learner.events).to_csv(run_dir / "events.csv", index=False)
        pd.DataFrame(metrics["cold_start"]).to_csv(
            run_dir / "cold_start.csv", index=False,
        )
        pd.DataFrame(learner.group_trace).to_csv(
            run_dir / "groups.csv", index=False,
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8",
        )
        (run_dir / "preset.json").write_text(
            json.dumps(preset.to_dict(), indent=2, sort_keys=True), encoding="utf-8",
        )
        overall = metrics["overall"]
        print(
            f"{dataset}/{preset.name}/seed{seed}: "
            f"F1={overall['f1']:.4f}, recall={overall['recall']:.4f}, "
            f"specificity={overall['specificity']:.4f}",
            flush=True,
        )
        manifest["resolved_preset"] = preset.to_dict()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a named SG-Share classification-tuned or cold-safe preset",
    )
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument(
        "--preset",
        required=True,
        choices=tuple(sorted({name for _, name in PRESETS})),
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    run(
        args.dataset, args.preset, args.data, args.output,
        _csv_ints(args.seeds), args.device, args.torch_threads,
    )


if __name__ == "__main__":
    main()
