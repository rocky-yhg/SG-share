from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process.run_sg_share_parameter_search import _configured
from process.run_table3_experiments import ECAP_POINTS, TABLE3_KS, _auc
from sgshare.config import load_config
from sgshare.data import load_stream
from sgshare.learner import SGShareLearner
from sgshare.metrics import binary_metrics


@dataclass(frozen=True)
class VariantConfig:
    adapter_mode: str
    refresh_alpha: float = 0.0
    split_profile: str = "historical"
    regroup_mode: str = "full"
    disable_mature_refine: bool = False
    periodic_regroup_enabled: bool = True


VARIANTS: Mapping[str, VariantConfig] = {
    "standalone": VariantConfig("standalone"),
    "regroup_on": VariantConfig("standalone"),
    "regroup_off": VariantConfig(
        "standalone", periodic_regroup_enabled=False,
    ),
    "user_mean": VariantConfig("user_mean"),
    "shadow_no_refresh": VariantConfig("shadow_personal"),
    "shadow_refresh_a10": VariantConfig("shadow_personal", 0.10),
    "shadow_refresh_a30": VariantConfig("shadow_personal", 0.30),
    "split_relaxed_standalone": VariantConfig("standalone", split_profile="relaxed"),
    "split_relaxed_shadow": VariantConfig("shadow_personal", split_profile="relaxed"),
    "full_regroup_no_post": VariantConfig(
        "standalone", split_profile="off", disable_mature_refine=True,
    ),
    "incremental_split_merge": VariantConfig(
        "standalone",
        split_profile="off",
        regroup_mode="incremental_split_merge",
        disable_mature_refine=True,
    ),
}


def _apply_split_profile(config, profile: str) -> None:
    if profile == "historical":
        return
    if profile == "off":
        config.refinements.verified_split = False
        return
    if profile != "relaxed":
        raise ValueError(f"unknown split profile: {profile}")
    grouping = config.grouping
    refinements = config.refinements
    refinements.verified_split = True
    grouping.cfl_coherence_threshold = 1.0
    grouping.cfl_disagreement_cosine_threshold = 1.0
    refinements.split_min_users = 5
    refinements.split_min_events = 10
    refinements.split_holdout_min_users = 2
    refinements.split_max_groups = 8
    refinements.split_loss_margin = -0.01
    refinements.split_pred_posrate_delta_tol = 1.0
    refinements.split_cohesion_tolerance = 1.0


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


def _macro_precision(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_positive = int(np.sum((labels != target) & (predictions == target)))
        denominator = true_positive + false_positive
        values.append(true_positive / denominator if denominator else 0.0)
    return float(np.mean(values))


def _macro_recall(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_negative = int(np.sum((labels == target) & (predictions != target)))
        denominator = true_positive + false_negative
        values.append(true_positive / denominator if denominator else 0.0)
    return float(np.mean(values))


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_positive = int(np.sum((labels != target) & (predictions == target)))
        false_negative = int(np.sum((labels == target) & (predictions != target)))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(values))


def _raw_metrics(events: pd.DataFrame) -> dict[str, float]:
    labels = events["label"].astype(int).to_numpy()
    probabilities = events["raw_probability"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    positive = binary_metrics(labels, predictions)
    return {
        "raw_accuracy": positive["accuracy"],
        "raw_positive_f1": positive["f1"],
        "raw_positive_precision": positive["precision"],
        "raw_positive_recall": positive["recall"],
        "raw_specificity": positive["specificity"],
        "raw_balanced_accuracy": positive["balanced_accuracy"],
        "raw_macro_f1": _macro_f1(labels, predictions),
        "raw_macro_precision": _macro_precision(labels, predictions),
        "raw_macro_recall": _macro_recall(labels, predictions),
        "raw_auc": _auc(labels, probabilities),
    }


def _cold_raw_metrics(events: pd.DataFrame, ks: Sequence[int]) -> list[dict[str, Any]]:
    users = {
        str(uid): rows.sort_values("event_index")
        for uid, rows in events.groupby("user_id", sort=False)
    }
    output = []
    for k in ks:
        macro_f1 = []
        macro_recall = []
        pooled_labels = []
        pooled_probabilities = []
        for rows in users.values():
            if len(rows) < k:
                continue
            subset = rows.iloc[:k]
            labels = subset["label"].astype(int).to_numpy()
            probabilities = subset["raw_probability"].astype(float).to_numpy()
            predictions = (probabilities >= 0.5).astype(int)
            macro_f1.append(_macro_f1(labels, predictions))
            macro_recall.append(_macro_recall(labels, predictions))
            pooled_labels.extend(labels.tolist())
            pooled_probabilities.extend(probabilities.tolist())
        labels = np.asarray(pooled_labels, dtype=np.int64)
        probabilities = np.asarray(pooled_probabilities, dtype=np.float64)
        predictions = (probabilities >= 0.5).astype(int)
        output.append({
            "k": int(k),
            "n_users": len(macro_f1),
            "raw_user_macro_f1": float(np.mean(macro_f1)) if macro_f1 else float("nan"),
            "raw_user_macro_recall": (
                float(np.mean(macro_recall)) if macro_recall else float("nan")
            ),
            "raw_pooled_macro_f1": (
                _macro_f1(labels, predictions) if len(labels) else float("nan")
            ),
            "raw_pooled_macro_recall": (
                _macro_recall(labels, predictions) if len(labels) else float("nan")
            ),
            "raw_pooled_auc": _auc(labels, probabilities) if len(labels) else float("nan"),
        })
    return output


def _trace_diagnostics(learner: SGShareLearner) -> dict[str, Any]:
    trace = pd.DataFrame(learner.group_trace)
    if trace.empty:
        return {
            "regroup_boundaries": 0,
            "regroup_churn_boundaries": 0,
            "split_parent_checks": 0,
            "split_accepted": 0,
            "reassignment_evaluated_users": 0,
            "reassignment_moves": 0,
            "shadow_changed_groups": 0,
            "shadow_refreshes": 0,
        }

    def total(column: str) -> float:
        return float(trace[column].fillna(0).sum()) if column in trace else 0.0

    refreshes = total("shadow_refreshes")
    output = {
        "regroup_boundaries": int(len(trace)),
        "regroup_churn_boundaries": int((trace.get("churn", 0) > 0).sum()),
        "mean_group_churn": float(trace.get("churn", pd.Series([0.0])).mean()),
        "split_parent_checks": int(total("split_clusters")),
        "split_accepted": int(total("split_accepted")),
        "split_rejected_validation": int(total("split_rejected_validation")),
        "split_skipped_group_limit": int(total("split_skipped_group_limit")),
        "split_skipped_user_support": int(total("split_skipped_user_support")),
        "split_skipped_geometry": int(total("split_skipped_geometry")),
        "split_skipped_child_users": int(total("split_skipped_child_users")),
        "split_skipped_child_events": int(total("split_skipped_child_events")),
        "reassignment_evaluated_users": int(total("reassignment_evaluated_users")),
        "reassignment_moves": int(total("reassignment_moves")),
        "shadow_new_group_inits": int(total("shadow_new_group_inits")),
        "shadow_changed_groups": int(total("shadow_changed_groups")),
        "shadow_refreshes": int(refreshes),
        "shadow_refresh_donors": int(total("shadow_refresh_donors")),
        "shadow_mean_refresh_relative_l2": (
            total("shadow_refresh_relative_l2_sum") / refreshes if refreshes else 0.0
        ),
        "shadow_mean_refresh_cosine_distance": (
            total("shadow_refresh_cosine_distance_sum") / refreshes if refreshes else 0.0
        ),
        "incremental_boundaries": int(total("incremental_enabled")),
        "incremental_split_attempts": int(total("incremental_split_attempts")),
        "incremental_split_accepts": int(total("incremental_split_accepts")),
        "incremental_reused_users": int(total("incremental_reused_users")),
        "incremental_new_users": int(total("incremental_new_users")),
        "incremental_similarity_evaluations": int(total("incremental_similarity_evaluations")),
        "incremental_inherited_adapters": int(total("incremental_inherited_adapters")),
    }
    return output


def _checkpoint(events: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(events)
    labels = frame["label"].astype(int).to_numpy()
    configured_predictions = frame["prediction"].astype(int).to_numpy()
    configured = binary_metrics(labels, configured_predictions)
    raw = _raw_metrics(frame)
    return {
        "configured_f1": configured["f1"],
        "raw_macro_f1": raw["raw_macro_f1"],
    }


def _run_one(
    dataset: str,
    stream,
    feature_count: int,
    output: Path,
    variant: str,
    seed: int,
    device: str,
    max_events: int,
    baseline_checkpoints: Mapping[int, Mapping[str, float]],
    early_stop: bool,
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    variant_config = VARIANTS[variant]
    mode = variant_config.adapter_mode
    refresh_alpha = variant_config.refresh_alpha
    config = _configured(load_config(dataset=dataset), ECAP_POINTS[dataset], seed)
    config.device = device
    config.grouping.shadow_group_refresh_alpha = refresh_alpha
    config.grouping.regroup_mode = variant_config.regroup_mode
    config.grouping.periodic_regroup_enabled = variant_config.periodic_regroup_enabled
    _apply_split_profile(config, variant_config.split_profile)
    if variant_config.disable_mature_refine:
        config.refinements.mature_refine = False
    learner = SGShareLearner(
        feature_count, config, "full_final", "gradient", mode,
    )
    selected_stream = stream[:max_events] if max_events > 0 else stream
    checkpoints = sorted({max(1, len(selected_stream) // 4), max(1, len(selected_stream) // 2), len(selected_stream)})
    observed_checkpoints: dict[int, dict[str, float]] = {}
    stopped_early = False
    stop_reason = ""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    for count, event in enumerate(selected_stream, start=1):
        learner.process(event)
        if count not in checkpoints:
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        values = _checkpoint(learner.events)
        observed_checkpoints[count] = values
        baseline = baseline_checkpoints.get(count)
        if (
            early_stop
            and variant not in {"standalone", "regroup_on"}
            and baseline
            and count < len(selected_stream)
        ):
            margin = 0.03 if count == checkpoints[0] else 0.02
            if (
                values["configured_f1"] < baseline["configured_f1"] - margin
                and values["raw_macro_f1"] < baseline["raw_macro_f1"] - margin
            ):
                stopped_early = True
                stop_reason = f"both_f1_metrics_below_baseline_by_{margin:.2f}_at_{count}"
                break
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    events = pd.DataFrame(learner.events)
    metrics = learner.summarize()
    configured = metrics["overall"]
    post = metrics["post_warmup"]
    raw = _raw_metrics(events)
    cold_raw = _cold_raw_metrics(events, TABLE3_KS[dataset])
    diagnostics = _trace_diagnostics(learner)
    row = {
        "dataset": dataset,
        "variant": variant,
        "adapter_mode": mode,
        "refresh_alpha": refresh_alpha,
        "split_profile": variant_config.split_profile,
        "regroup_mode": variant_config.regroup_mode,
        "seed": int(seed),
        "device": device,
        "status": "early_stopped" if stopped_early else "completed",
        "stop_reason": stop_reason,
        "n_events": len(events),
        "wall_seconds": elapsed,
        "events_per_second": len(events) / elapsed if elapsed else float("nan"),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2) if device == "cuda" else 0.0
        ),
        "configured_accuracy": configured["accuracy"],
        "configured_precision": configured["precision"],
        "configured_recall": configured["recall"],
        "configured_specificity": configured["specificity"],
        "configured_balanced_accuracy": configured["balanced_accuracy"],
        "configured_f1": configured["f1"],
        "configured_post_f1": post["f1"],
        **raw,
        **diagnostics,
        "personal_adapter_updates": metrics["personal_adapter_updates"],
        "shadow_updates": metrics["shadow_updates"],
    }
    for cold in cold_raw:
        for key, value in cold.items():
            if key not in {"k", "n_users"}:
                row[f"cold{cold['k']}_{key}"] = value

    run_dir = output / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(learner.group_trace).to_csv(run_dir / "groups.csv", index=False)
    pd.DataFrame(metrics["cold_start"]).to_csv(run_dir / "cold_start_configured.csv", index=False)
    pd.DataFrame(cold_raw).to_csv(run_dir / "cold_start_raw.csv", index=False)
    pd.DataFrame([
        {"event_count": count, **values}
        for count, values in observed_checkpoints.items()
    ]).to_csv(run_dir / "checkpoints.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8",
    )
    (run_dir / "diagnostics.json").write_text(
        json.dumps(_json_safe(diagnostics), indent=2, sort_keys=True), encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8",
    )
    return row, observed_checkpoints


def run(
    dataset: str,
    data: str | Path,
    output: str | Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    device: str,
    torch_threads: int,
    max_events: int,
    early_stop: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    stream, feature_names = load_stream(data, legacy_global_scaling=False, no_scaling=True)
    torch.set_num_threads(max(1, int(torch_threads)))
    all_rows = []
    for seed in seeds:
        seed_variants = list(variants)
        if "standalone" in seed_variants:
            seed_variants.remove("standalone")
            seed_variants.insert(0, "standalone")
        elif "regroup_on" in seed_variants:
            seed_variants.remove("regroup_on")
            seed_variants.insert(0, "regroup_on")
        baseline_checkpoints: dict[int, dict[str, float]] = {}
        for variant in seed_variants:
            row, checkpoints = _run_one(
                dataset, stream, len(feature_names), output_path, variant, seed,
                device, max_events, baseline_checkpoints, early_stop,
            )
            all_rows.append(row)
            if variant in {"standalone", "regroup_on"}:
                baseline_checkpoints = checkpoints
            pd.DataFrame(all_rows).to_csv(output_path / "summary.csv", index=False)
            print(json.dumps(_json_safe(row), sort_keys=True), flush=True)

    frame = pd.DataFrame(all_rows)
    completed = frame[frame["status"] == "completed"]
    if not completed.empty:
        numeric = completed.select_dtypes(include=[np.number]).columns.difference(["seed"])
        means = completed.groupby("variant", sort=False)[list(numeric)].mean().add_suffix("_mean")
        stds = completed.groupby("variant", sort=False)[list(numeric)].std(ddof=0).fillna(0.0).add_suffix("_std")
        means.join(stds).reset_index().to_csv(output_path / "summary_mean_std.csv", index=False)
    (output_path / "manifest.json").write_text(json.dumps({
        "dataset": dataset,
        "data": str(data),
        "variants": {
            variant: asdict(VARIANTS[variant]) for variant in variants
        },
        "seeds": [int(seed) for seed in seeds],
        "device": device,
        "max_events": max_events,
        "early_stop": early_stop,
        "prediction_protocols": ["configured_online", "raw_probability>=0.5"],
    }, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Shadow-personal group adapter experiment")
    parser.add_argument("--dataset", default="globem", choices=tuple(sorted(TABLE3_KS)))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--early-stop", action="store_true")
    args = parser.parse_args(argv)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    run(
        args.dataset, args.data, args.output, variants, seeds, args.device,
        args.torch_threads, args.max_events, args.early_stop,
    )


if __name__ == "__main__":
    main()
