from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "ces_globem_parameter_comparison_20260720"

CES_CLASS = RESULTS / "online_sota_ces_20260720/final_comparison_causal_scaled/classification.csv"
CES_COLD = RESULTS / "online_sota_ces_20260720/final_comparison_causal_scaled/cold_start.csv"
GLOBEM_CLASS = RESULTS / "globem_full_complete_comparison_20260720/classification_all.csv"
GLOBEM_COLD = RESULTS / "globem_full_complete_comparison_20260720/cold_start_all.csv"
CES_SCREEN = RESULTS / "sg_share_parameter_search_ces_screen_20260720/summary.csv"
GLOBEM_SEARCH = RESULTS / "sg_share_parameter_search_globem_20260720_v2/summary.csv"
CES_TUNED_METRICS = (
    RESULTS / "sg_share_parameter_confirmation_ces_20260720/ces/"
    "confirm_class_raw_a0p200_lam20_k4_legacy_seed42/metrics.json"
)


METHOD_NAMES = {
    "oli2ds": "OLI2DS", "obal": "OBAL", "hbp": "HBP/ODL",
    "koil": "KOIL", "olifl": "OLIFL", "olfl": "OLFL",
    "sg_share": "SG-Share default",
}


def _cold_rows_from_metrics(dataset: str, method: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cold in metrics["cold_start"]:
        row = {
            "dataset": dataset, "method": method, "scope": "group_shared",
            "k": int(cold["k"]), "n_seeds": 1,
        }
        for source, target in (
            ("user_macro_f1", "user_macro_f1_mean"),
            ("user_macro_recall", "user_macro_recall_mean"),
            ("user_macro_specificity", "user_macro_specificity_mean"),
            ("user_macro_balanced_accuracy", "user_macro_balanced_accuracy_mean"),
            ("pooled_f1", "pooled_f1_mean"),
            ("pooled_recall", "pooled_recall_mean"),
            ("pooled_specificity", "pooled_specificity_mean"),
            ("pooled_balanced_accuracy", "pooled_balanced_accuracy_mean"),
        ):
            row[target] = cold[source]
            row[target.replace("_mean", "_std")] = 0.0
        rows.append(row)
    return rows


def classification_table() -> pd.DataFrame:
    ces = pd.read_csv(CES_CLASS)
    ces["method"] = ces["method"].map(METHOD_NAMES).fillna(ces["method"])
    ces_out = pd.DataFrame({
        "dataset": "CES", "method": ces["method"], "scope": ces["scope"],
        "n_seeds": 1, "provenance": ces["provenance"],
        "f1_mean": ces["f1"], "f1_std": 0.0,
        "recall_mean": ces["recall"], "recall_std": 0.0,
        "specificity_mean": ces["specificity"], "specificity_std": 0.0,
        "balanced_accuracy_mean": ces["balanced_accuracy"], "balanced_accuracy_std": 0.0,
        "accuracy_mean": ces["accuracy"], "accuracy_std": 0.0,
        "post_f1_mean": ces["post_f1"], "post_f1_std": 0.0,
    })
    if CES_TUNED_METRICS.exists():
        metrics = json.loads(CES_TUNED_METRICS.read_text(encoding="utf-8"))
        overall, post = metrics["overall"], metrics["post_warmup"]
        ces_out = pd.concat([ces_out, pd.DataFrame([{
            "dataset": "CES", "method": "SG-Share classification-tuned", "scope": "group_shared",
            "n_seeds": 1, "provenance": "full stream; alpha=0.20, lambda=20, k=4",
            "f1_mean": overall["f1"], "f1_std": 0.0,
            "recall_mean": overall["recall"], "recall_std": 0.0,
            "specificity_mean": overall["specificity"], "specificity_std": 0.0,
            "balanced_accuracy_mean": overall["balanced_accuracy"], "balanced_accuracy_std": 0.0,
            "accuracy_mean": overall["accuracy"], "accuracy_std": 0.0,
            "post_f1_mean": post["f1"], "post_f1_std": 0.0,
        }])], ignore_index=True)
    globem = pd.read_csv(GLOBEM_CLASS)
    globem.insert(0, "dataset", "GLOBEM")
    columns = list(ces_out.columns)
    return pd.concat([ces_out, globem.reindex(columns=columns)], ignore_index=True)


def cold_table() -> pd.DataFrame:
    ces = pd.read_csv(CES_COLD)
    ces["method"] = ces["method"].map(METHOD_NAMES).fillna(ces["method"])
    ces = ces.rename(columns={
        "user_macro_f1": "user_macro_f1_mean",
        "user_macro_recall": "user_macro_recall_mean",
        "user_macro_specificity": "user_macro_specificity_mean",
        "user_macro_balanced_accuracy": "user_macro_balanced_accuracy_mean",
        "pooled_f1": "pooled_f1_mean", "pooled_recall": "pooled_recall_mean",
        "pooled_specificity": "pooled_specificity_mean",
        "pooled_balanced_accuracy": "pooled_balanced_accuracy_mean",
    })
    ces["dataset"] = "CES"
    ces["n_seeds"] = 1
    mean_columns = [column for column in ces if column.endswith("_mean")]
    for column in mean_columns:
        ces[column.replace("_mean", "_std")] = 0.0
    if CES_TUNED_METRICS.exists():
        metrics = json.loads(CES_TUNED_METRICS.read_text(encoding="utf-8"))
        ces = pd.concat([
            ces, pd.DataFrame(_cold_rows_from_metrics("CES", "SG-Share classification-tuned", metrics)),
        ], ignore_index=True)
    globem = pd.read_csv(GLOBEM_COLD)
    globem.insert(0, "dataset", "GLOBEM")
    columns = [
        "dataset", "method", "scope", "k", "n_seeds",
        "user_macro_f1_mean", "user_macro_f1_std",
        "user_macro_recall_mean", "user_macro_recall_std",
        "user_macro_specificity_mean", "user_macro_specificity_std",
        "user_macro_balanced_accuracy_mean", "user_macro_balanced_accuracy_std",
        "pooled_f1_mean", "pooled_f1_std", "pooled_recall_mean", "pooled_recall_std",
        "pooled_specificity_mean", "pooled_specificity_std",
        "pooled_balanced_accuracy_mean", "pooled_balanced_accuracy_std",
    ]
    return pd.concat([ces.reindex(columns=columns), globem.reindex(columns=columns)], ignore_index=True)


def sensitivity_table() -> pd.DataFrame:
    common = [
        "alpha", "eligibility_lambda", "k_min", "strict_eligibility", "f1", "post_f1",
        "recall", "specificity", "balanced_accuracy", "cold5_macro_f1",
        "cold5_macro_recall", "cold5_pooled_f1", "cold5_pooled_recall",
        "cold10_macro_f1", "cold10_macro_recall", "cold10_pooled_f1",
        "cold10_pooled_recall", "n_groups_final",
    ]
    ces = pd.read_csv(CES_SCREEN)
    ces["dataset"] = "CES"
    ces["evaluation_scope"] = "first 10,000 events (screening)"
    ces["parameter"] = ces["stage"].replace({"baseline": "reference"})
    ces["value"] = np.select(
        [ces["stage"].eq("alpha"), ces["stage"].eq("lambda"), ces["stage"].eq("k")],
        [ces["alpha"], ces["eligibility_lambda"], ces["k_min"]], default=np.nan,
    )

    globem = pd.read_csv(GLOBEM_SEARCH)
    selected = [globem["stage"].eq("baseline")]
    selected.append(globem["stage"].eq("alpha") & ~globem["normalize_gradient"])
    selected.append(
        globem["stage"].eq("lambda") & ~globem["normalize_gradient"]
        & np.isclose(globem["alpha"], 0.20) & ~globem["strict_eligibility"]
    )
    selected.append(
        globem["stage"].eq("k") & ~globem["normalize_gradient"]
        & np.isclose(globem["alpha"], 0.20) & globem["strict_eligibility"]
        & globem["eligibility_lambda"].eq(2)
    )
    globem = globem[np.logical_or.reduce(selected)].copy()
    globem["dataset"] = "GLOBEM"
    globem["evaluation_scope"] = "full stream"
    globem["parameter"] = globem["stage"].replace({"baseline": "reference"})
    globem["value"] = np.select(
        [globem["stage"].eq("alpha"), globem["stage"].eq("lambda"), globem["stage"].eq("k")],
        [globem["alpha"], globem["eligibility_lambda"], globem["k_min"]], default=np.nan,
    )
    columns = ["dataset", "evaluation_scope", "parameter", "value", "variant"] + common
    return pd.concat([ces.reindex(columns=columns), globem.reindex(columns=columns)], ignore_index=True)


def pairwise_tables(classification: pd.DataFrame, cold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_rows = []
    cold_rows = []
    selected_class = {
        "CES": "SG-Share classification-tuned" if CES_TUNED_METRICS.exists() else "SG-Share default",
        "GLOBEM": "SG-Share classification-tuned",
    }
    selected_cold = {"CES": "SG-Share default", "GLOBEM": "SG-Share cold-safe"}
    for dataset in ("CES", "GLOBEM"):
        sg = classification[(classification.dataset == dataset) & (classification.method == selected_class[dataset])].iloc[0]
        peers = classification[(classification.dataset == dataset) & (classification.scope == "per_user")]
        for _, peer in peers.iterrows():
            class_rows.append({
                "dataset": dataset, "sg_variant": sg.method, "baseline": peer.method,
                "delta_f1": sg.f1_mean - peer.f1_mean,
                "delta_post_f1": sg.post_f1_mean - peer.post_f1_mean,
                "delta_recall": sg.recall_mean - peer.recall_mean,
                "delta_specificity": sg.specificity_mean - peer.specificity_mean,
                "delta_balanced_accuracy": sg.balanced_accuracy_mean - peer.balanced_accuracy_mean,
            })
        sg_cold = cold[(cold.dataset == dataset) & (cold.method == selected_cold[dataset])]
        for k in sorted(sg_cold.k.unique()):
            sgk = sg_cold[sg_cold.k == k].iloc[0]
            peers_k = cold[(cold.dataset == dataset) & (cold.scope == "per_user") & (cold.k == k)]
            for _, peer in peers_k.iterrows():
                cold_rows.append({
                    "dataset": dataset, "k": int(k), "sg_variant": sgk.method, "baseline": peer.method,
                    "delta_macro_f1": sgk.user_macro_f1_mean - peer.user_macro_f1_mean,
                    "delta_macro_recall": sgk.user_macro_recall_mean - peer.user_macro_recall_mean,
                    "delta_pooled_f1": sgk.pooled_f1_mean - peer.pooled_f1_mean,
                    "delta_pooled_recall": sgk.pooled_recall_mean - peer.pooled_recall_mean,
                })
    return pd.DataFrame(class_rows), pd.DataFrame(cold_rows)


def markdown(classification: pd.DataFrame, cold: pd.DataFrame, sensitivity: pd.DataFrame,
             class_pair: pd.DataFrame, cold_pair: pd.DataFrame) -> str:
    lines = ["# CES and GLOBEM parameter search and complete comparison", ""]
    for dataset in ("CES", "GLOBEM"):
        lines.extend([f"## {dataset} classification", ""])
        view = classification[classification.dataset == dataset][
            ["method", "scope", "f1_mean", "post_f1_mean", "recall_mean", "specificity_mean", "balanced_accuracy_mean"]
        ].sort_values("f1_mean", ascending=False)
        lines.extend([view.to_markdown(index=False, floatfmt=".4f"), ""])
        lines.extend([f"## {dataset} cold start", ""])
        view = cold[cold.dataset == dataset][
            ["method", "scope", "k", "user_macro_f1_mean", "user_macro_recall_mean", "pooled_f1_mean", "pooled_recall_mean"]
        ].sort_values(["k", "user_macro_f1_mean"], ascending=[True, False])
        lines.extend([view.to_markdown(index=False, floatfmt=".4f"), ""])
    lines.extend([
        "## Sensitivity protocol", "",
        "CES sensitivity uses the first 10,000 events as a screening experiment; GLOBEM sensitivity uses the full stream. Absolute metrics across these two scopes must not be compared directly.", "",
        sensitivity[["dataset", "evaluation_scope", "parameter", "value", "alpha", "eligibility_lambda", "k_min", "strict_eligibility", "f1", "post_f1", "specificity", "cold5_macro_f1", "n_groups_final"]].to_markdown(index=False, floatfmt=".4f"), "",
        "## Pairwise deltas against every per-user method", "",
        "### Classification", "", class_pair.to_markdown(index=False, floatfmt=".4f"), "",
        "### Cold start", "", cold_pair.to_markdown(index=False, floatfmt=".4f"), "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    classification = classification_table()
    cold = cold_table()
    sensitivity = sensitivity_table()
    class_pair, cold_pair = pairwise_tables(classification, cold)
    classification.to_csv(OUTPUT / "classification_all.csv", index=False)
    cold.to_csv(OUTPUT / "cold_start_all.csv", index=False)
    sensitivity.to_csv(OUTPUT / "sensitivity_alpha_lambda_k.csv", index=False)
    class_pair.to_csv(OUTPUT / "classification_vs_each_per_user.csv", index=False)
    cold_pair.to_csv(OUTPUT / "cold_start_vs_each_per_user.csv", index=False)
    (OUTPUT / "comparison.md").write_text(
        markdown(classification, cold, sensitivity, class_pair, cold_pair), encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
