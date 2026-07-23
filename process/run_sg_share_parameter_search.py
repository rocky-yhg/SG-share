from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

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
from sgshare.presets import PRESETS, apply_named_preset


@dataclass(frozen=True)
class SearchPoint:
    alpha: float
    eligibility_lambda: int
    k_min: int
    normalize_gradient: bool
    stage: str
    strict_eligibility: bool = False

    @property
    def name(self) -> str:
        mode = "norm" if self.normalize_gradient else "raw"
        alpha = f"{self.alpha:.3f}".replace(".", "p")
        eligibility = "strict" if self.strict_eligibility else "legacy"
        return (
            f"{self.stage}_{mode}_a{alpha}_lam{self.eligibility_lambda}_"
            f"k{self.k_min}_{eligibility}"
        )


def _csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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


def _configured(base: ExperimentConfig, point: SearchPoint, seed: int) -> ExperimentConfig:
    cfg = apply_variant(base, "full_final")
    cfg.seed = int(seed)
    cfg.training.route_grad_ema_decay = float(1.0 - point.alpha)
    cfg.training.route_grad_normalize_before_ema = bool(point.normalize_gradient)

    lam = int(point.eligibility_lambda)
    cfg.grouping.min_observations = lam
    if point.strict_eligibility:
        # For the lambda experiment, one threshold controls both initial
        # admission and later regrouping, without a lower warmup fallback.
        cfg.grouping.warmup_initial_min_samples = lam
        cfg.grouping.warmup_initial_min_samples_fallback = lam
    cfg.grouping.k_min = int(point.k_min)
    return cfg


def _cold_at(metrics: dict[str, Any], k: int) -> dict[str, float]:
    for row in metrics.get("cold_start", []):
        if int(row["k"]) == int(k):
            return row
    return {}


def _row(dataset: str, point: SearchPoint, seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    post = metrics["post_warmup"]
    cold5 = _cold_at(metrics, 5)
    cold10 = _cold_at(metrics, 10)
    cold_f1 = np.nanmean([
        float(row.get("user_macro_f1", np.nan)) for row in (cold5, cold10)
    ])
    cold_recall = np.nanmean([
        float(row.get("user_macro_recall", np.nan)) for row in (cold5, cold10)
    ])
    row = {
        "dataset": dataset,
        "variant": point.name,
        "stage": point.stage,
        "seed": int(seed),
        "alpha": point.alpha,
        "route_grad_ema_decay": 1.0 - point.alpha,
        "normalize_gradient": point.normalize_gradient,
        "eligibility_lambda": point.eligibility_lambda,
        "k_min": point.k_min,
        "strict_eligibility": point.strict_eligibility,
        "f1": overall["f1"],
        "recall": overall["recall"],
        "specificity": overall["specificity"],
        "balanced_accuracy": overall["balanced_accuracy"],
        "accuracy": overall["accuracy"],
        "post_f1": post["f1"],
        "post_recall": post["recall"],
        "post_specificity": post["specificity"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "cold5_macro_f1": cold5.get("user_macro_f1", np.nan),
        "cold5_macro_recall": cold5.get("user_macro_recall", np.nan),
        "cold5_pooled_f1": cold5.get("pooled_f1", np.nan),
        "cold5_pooled_recall": cold5.get("pooled_recall", np.nan),
        "cold10_macro_f1": cold10.get("user_macro_f1", np.nan),
        "cold10_macro_recall": cold10.get("user_macro_recall", np.nan),
        "cold10_pooled_f1": cold10.get("pooled_f1", np.nan),
        "cold10_pooled_recall": cold10.get("pooled_recall", np.nan),
        "cold_score": float(cold_f1 + 0.25 * cold_recall),
        "n_groups_final": metrics["n_groups_final"],
        "mean_groups": metrics["mean_groups"],
    }
    for k in (20, 50):
        cold = _cold_at(metrics, k)
        row[f"cold{k}_macro_f1"] = cold.get("user_macro_f1", np.nan)
        row[f"cold{k}_macro_recall"] = cold.get("user_macro_recall", np.nan)
        row[f"cold{k}_pooled_f1"] = cold.get("pooled_f1", np.nan)
        row[f"cold{k}_pooled_recall"] = cold.get("pooled_recall", np.nan)
    return row


def _run_one(dataset: str, base: ExperimentConfig, point: SearchPoint, seed: int,
             stream, feature_count: int, output: Path, skip_existing: bool) -> dict[str, Any]:
    run_dir = output / dataset / f"{point.name}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return _row(dataset, point, seed, metrics)
    cfg = _configured(base, point, seed)
    learner = SGShareLearner(feature_count, cfg, "full_final", "gradient")
    metrics = learner.run(stream)
    metrics.update({
        "dataset": dataset, "method": "full_final", "variant": point.name,
        "seed": seed,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(learner.events).to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(metrics["cold_start"]).to_csv(run_dir / "cold_start.csv", index=False)
    pd.DataFrame(learner.group_trace).to_csv(run_dir / "groups.csv", index=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8",
    )
    return _row(dataset, point, seed, metrics)


def _passes_constraints(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        float(row["f1"]) >= float(baseline["f1"]) - 0.003
        and float(row["specificity"]) >= float(baseline["specificity"]) - 0.01
    )


def _enriched(rows: Iterable[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    enriched_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["passes_constraints"] = _passes_constraints(row, baseline)
        enriched["passes_classification_safety"] = (
            float(row["specificity"]) >= float(baseline["specificity"]) - 0.01
        )
        enriched["delta_f1"] = float(row["f1"]) - float(baseline["f1"])
        enriched["delta_specificity"] = float(row["specificity"]) - float(baseline["specificity"])
        enriched["delta_cold5_macro_f1"] = (
            float(row["cold5_macro_f1"]) - float(baseline["cold5_macro_f1"])
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _rank_cold(rows: Iterable[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        _enriched(rows, baseline),
        key=lambda row: (
            bool(row["passes_constraints"]), float(row["cold_score"]),
            float(row["post_f1"]), float(row["f1"]),
        ),
        reverse=True,
    )


def _rank_classification(
    rows: Iterable[dict[str, Any]], baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    return sorted(
        _enriched(rows, baseline),
        key=lambda row: (
            bool(row["passes_classification_safety"]), float(row["post_f1"]),
            float(row["f1"]), float(row["balanced_accuracy"]),
        ),
        reverse=True,
    )


def _rank_cold_unconstrained(
    rows: Iterable[dict[str, Any]], baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    return sorted(
        _enriched(rows, baseline),
        key=lambda row: (
            float(row["cold_score"]), float(row["cold5_macro_f1"]),
            float(row["cold10_macro_f1"]),
        ),
        reverse=True,
    )


def _top_unique(
    rankings: Sequence[Sequence[dict[str, Any]]], count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[float, int, int, bool, bool]] = set()
    for ranking in rankings:
        for row in ranking:
            key = (
                float(row["alpha"]), int(row["eligibility_lambda"]),
                int(row["k_min"]), bool(row["normalize_gradient"]),
                bool(row["strict_eligibility"]),
            )
            if key not in seen:
                seen.add(key)
                output.append(row)
            if len(output) >= count:
                return output
    return output


def _dedupe(points: Iterable[SearchPoint]) -> list[SearchPoint]:
    output: list[SearchPoint] = []
    seen: set[tuple[float, int, int, bool, bool]] = set()
    for point in points:
        key = (
            point.alpha, point.eligibility_lambda, point.k_min,
            point.normalize_gradient, point.strict_eligibility,
        )
        if key not in seen:
            seen.add(key)
            output.append(point)
    return output


def run_search(dataset: str, data: str | Path, output: str | Path, seed: int,
               alphas: Sequence[float], lambdas: Sequence[int], ks: Sequence[int],
               skip_existing: bool, gradient_modes: Sequence[str] = ("raw", "norm"),
               eligibility_modes: Sequence[str] = ("legacy", "strict"),
               max_events: int | None = None, search_strategy: str = "sequential",
               torch_threads: int = 1, base_preset: str | None = None,
               device: str = "cpu") -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    base = load_config(dataset=dataset)
    if base_preset:
        base, _ = apply_named_preset(base, base_preset)
    base.device = device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    if max_events is not None and max_events > 0:
        stream = stream[:max_events]
    torch.set_num_threads(max(1, int(torch_threads)))
    default_alpha = 1.0 - base.training.route_grad_ema_decay
    default_lambda = base.grouping.min_observations
    default_k = base.grouping.k_min
    baseline_point = SearchPoint(
        default_alpha, default_lambda, default_k, False, "baseline", False,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def execute(points: Sequence[SearchPoint]) -> list[dict[str, Any]]:
        stage_rows: list[dict[str, Any]] = []
        for point in _dedupe(points):
            print(f"running {dataset}/{point.name}/seed{seed}", flush=True)
            try:
                row = _run_one(dataset, base, point, seed, stream, len(feature_names), root, skip_existing)
                rows.append(row)
                stage_rows.append(row)
                print(
                    f"done F1={row['f1']:.4f} post={row['post_f1']:.4f} "
                    f"spec={row['specificity']:.4f} cold5={row['cold5_macro_f1']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "dataset": dataset, "variant": point.name, "seed": seed,
                    "error_type": type(exc).__name__, "error": str(exc),
                }
                errors.append(error)
                print(f"ERROR {point.name}: {type(exc).__name__}: {exc}", flush=True)
        return stage_rows

    baseline = execute([baseline_point])[0]

    normalization_modes = [mode == "norm" for mode in gradient_modes]
    strict_modes = [mode == "strict" for mode in eligibility_modes]
    alpha_points = [
        SearchPoint(alpha, default_lambda, default_k, normalize, "alpha", False)
        for normalize in normalization_modes for alpha in alphas
        if normalize or not np.isclose(alpha, default_alpha)
    ]
    alpha_rows = execute(alpha_points)
    if search_strategy == "one_factor":
        lambda_points = [
            SearchPoint(default_alpha, lam, default_k, False, "lambda", strict)
            for strict in strict_modes for lam in lambdas
        ]
        lambda_rows = execute(lambda_points)
        k_points = [
            SearchPoint(default_alpha, default_lambda, k, False, "k", False)
            for k in ks
            if int(k) != int(default_k)
        ]
        k_rows = execute(k_points)
        alpha_candidates = []
        lambda_candidates = []
    else:
        alpha_candidates = _top_unique([
            _rank_cold(alpha_rows + [baseline], baseline)[:1],
            _rank_classification(alpha_rows + [baseline], baseline)[:1],
        ], 2)

        lambda_points = [
            SearchPoint(
                float(candidate["alpha"]), lam, default_k,
                bool(candidate["normalize_gradient"]), "lambda", strict,
            )
            for candidate in alpha_candidates for strict in strict_modes for lam in lambdas
        ]
        lambda_rows = execute(lambda_points)
        lambda_candidates = _top_unique([
            _rank_cold(lambda_rows + alpha_candidates, baseline)[:1],
            _rank_classification(lambda_rows + alpha_candidates, baseline)[:1],
        ], 2)

        k_points = [
            SearchPoint(
                float(candidate["alpha"]), int(candidate["eligibility_lambda"]), k,
                bool(candidate["normalize_gradient"]), "k",
                bool(candidate["strict_eligibility"]),
            )
            for candidate in lambda_candidates for k in ks
        ]
        k_rows = execute(k_points)

    cold_ranked = _rank_cold(rows, baseline)
    cold_unconstrained_ranked = _rank_cold_unconstrained(rows, baseline)
    classification_ranked = _rank_classification(rows, baseline)
    pd.DataFrame(rows).drop_duplicates(subset=["variant", "seed"], keep="last").to_csv(
        root / "summary.csv", index=False,
    )
    pd.DataFrame(cold_ranked).drop_duplicates(
        subset=[
            "alpha", "eligibility_lambda", "k_min", "normalize_gradient",
            "strict_eligibility", "seed",
        ],
        keep="first",
    ).to_csv(root / "ranking_cold_start.csv", index=False)
    pd.DataFrame(cold_unconstrained_ranked).drop_duplicates(
        subset=[
            "alpha", "eligibility_lambda", "k_min", "normalize_gradient",
            "strict_eligibility", "seed",
        ],
        keep="first",
    ).to_csv(root / "ranking_cold_start_unconstrained.csv", index=False)
    pd.DataFrame(classification_ranked).drop_duplicates(
        subset=[
            "alpha", "eligibility_lambda", "k_min", "normalize_gradient",
            "strict_eligibility", "seed",
        ],
        keep="first",
    ).to_csv(root / "ranking_classification.csv", index=False)
    pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)
    best_cold = cold_ranked[0]
    best_cold_unconstrained = cold_unconstrained_ranked[0]
    best_classification = classification_ranked[0]
    payload = {
        "dataset": dataset,
        "data": str(data),
        "seed": seed,
        "n_events": len(stream),
        "n_features": len(feature_names),
        "max_events": max_events,
        "search_strategy": search_strategy,
        "torch_threads": torch_threads,
        "base_preset": base_preset,
        "device": device,
        "selection_rules": {
            "cold_start": {
                "constraints": "overall F1 >= baseline-0.003 and specificity >= baseline-0.01",
                "objective": "maximize mean(K5,K10 user-macro F1) + 0.25*mean user-macro recall",
            },
            "classification": {
                "constraint": "specificity >= baseline-0.01",
                "objective": "maximize post F1, then overall F1 and balanced accuracy",
            },
        },
        "baseline": _json_safe(baseline),
        "best_cold_start_safe": _json_safe(best_cold),
        "best_cold_start_unconstrained": _json_safe(best_cold_unconstrained),
        "best_classification": _json_safe(best_classification),
        "n_runs": len(rows),
        "n_errors": len(errors),
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search SG-Share alpha, eligibility lambda, and k_min")
    parser.add_argument("--dataset", required=True, choices=["ces", "globem"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/sg_share_parameter_search_20260720")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--lambdas", default="2,5,10,15,20")
    parser.add_argument("--ks", default="2,3,4,5,6")
    parser.add_argument("--gradient-modes", default="raw,norm")
    parser.add_argument("--eligibility-modes", default="legacy,strict")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--search-strategy", choices=["sequential", "one_factor"], default="sequential")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--base-preset",
        choices=tuple(sorted({name for _, name in PRESETS})),
        help="Apply a named SG-Share preset before varying search parameters.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    payload = run_search(
        args.dataset, args.data, args.output, args.seed,
        _csv_floats(args.alphas), _csv_ints(args.lambdas), _csv_ints(args.ks),
        args.skip_existing,
        [part.strip().lower() for part in args.gradient_modes.split(",") if part.strip()],
        [part.strip().lower() for part in args.eligibility_modes.split(",") if part.strip()],
        args.max_events, args.search_strategy, args.torch_threads,
        args.base_preset, args.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
