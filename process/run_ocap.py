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

from ocap.config import load_config
from ocap.data import load_stream
from ocap.framework import OCAP
from ocap.settings import apply_main_configuration


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
    data: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    device: str,
    torch_threads: int,
) -> None:
    configuration_name = "main"
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "configuration": configuration_name,
        "data": str(data),
        "data_sha256": _sha256(data),
        "n_data_triplets": len(stream),
        "n_features": len(feature_names),
        "seeds": [int(seed) for seed in seeds],
        "device": device,
        "group_adapter_construction": "merged_trainable",
    }
    for seed in seeds:
        config = load_config(dataset=dataset)
        config.seed = int(seed)
        config.device = device
        config, main_configuration = apply_main_configuration(config, configuration_name)
        ocap = OCAP(
            len(feature_names), config, "ocap", "gradient", "merged_trainable",
        )
        metrics = ocap.run(stream)
        run_dir = root / f"{dataset}_{main_configuration.name}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(_json_safe(metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pd.DataFrame(ocap.prediction_history).to_csv(run_dir / "prediction_history.csv", index=False)
        pd.DataFrame(metrics["low_evidence"]).to_csv(
            run_dir / "low_evidence.csv", index=False,
        )
        pd.DataFrame(ocap.group_trace).to_csv(
            run_dir / "groups.csv", index=False,
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8",
        )
        (run_dir / "configuration.json").write_text(
            json.dumps(main_configuration.to_dict(), indent=2, sort_keys=True), encoding="utf-8",
        )
        overall = metrics["overall"]
        print(
            f"{dataset}/{main_configuration.name}/seed{seed}: "
            f"F1={overall['f1']:.4f}, recall={overall['recall']:.4f}, "
            f"specificity={overall['specificity']:.4f}",
            flush=True,
        )
        manifest["resolved_configuration"] = main_configuration.to_dict()
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed OCAP main-experiment configuration",
    )
    parser.add_argument("--dataset", required=True, choices=("ces", "globem"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    run(
        args.dataset, args.data, args.output,
        _csv_ints(args.seeds), args.device, args.torch_threads,
    )


if __name__ == "__main__":
    main()
