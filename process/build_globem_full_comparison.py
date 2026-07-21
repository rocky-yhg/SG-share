from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SG_LABELS = {
    "confirm_baseline": "SG-Share default",
    "confirm_class": "SG-Share classification-tuned",
    "confirm_cold_max": "SG-Share cold-max (no groups)",
    "confirm_cold": "SG-Share cold-safe",
}


def _sg_label(variant: str) -> str:
    for prefix, label in SG_LABELS.items():
        if variant.startswith(prefix):
            return label
    raise ValueError(f"Unknown SG confirmation variant: {variant}")


def _sg_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification: list[dict[str, Any]] = []
    cold: list[dict[str, Any]] = []
    for path in sorted(root.glob("globem/*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        variant = str(metrics["variant"])
        label = _sg_label(variant)
        overall = metrics["overall"]
        post = metrics["post_warmup"]
        classification.append({
            "method": label, "scope": "group_shared", "seed": int(metrics["seed"]),
            "provenance": "SG-Share restored full_final parameter variant",
            "f1": overall["f1"], "recall": overall["recall"],
            "specificity": overall["specificity"],
            "balanced_accuracy": overall["balanced_accuracy"], "accuracy": overall["accuracy"],
            "post_f1": post["f1"], "post_recall": post["recall"],
            "post_specificity": post["specificity"],
            "post_balanced_accuracy": post["balanced_accuracy"],
        })
        for row in metrics["cold_start"]:
            cold.append({
                "method": label, "scope": "group_shared", "seed": int(metrics["seed"]),
                **row,
            })
    return pd.DataFrame(classification), pd.DataFrame(cold)


def _mean_std(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group["seed"].nunique())
        if "provenance" in group:
            row["provenance"] = str(group["provenance"].iloc[0])
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _md_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    display = frame[columns].copy()
    display.columns = headers
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_fmt)
    return display.to_markdown(index=False)


def build(sota_root: Path, sg_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sota_class = pd.read_csv(sota_root / "summary.csv")
    sota_cold = pd.read_csv(sota_root / "cold_start.csv")
    sg_class, sg_cold = _sg_rows(sg_root)
    class_raw = pd.concat([sota_class, sg_class], ignore_index=True, sort=False)
    cold_raw = pd.concat([sota_cold, sg_cold], ignore_index=True, sort=False)

    class_metrics = [
        "f1", "recall", "specificity", "balanced_accuracy", "accuracy",
        "post_f1", "post_recall", "post_specificity", "post_balanced_accuracy",
    ]
    cold_metrics = [
        "user_macro_f1", "user_macro_recall", "user_macro_specificity",
        "user_macro_balanced_accuracy", "pooled_f1", "pooled_recall",
        "pooled_specificity", "pooled_balanced_accuracy",
    ]
    classification = _mean_std(class_raw, ["method", "scope"], class_metrics)
    cold = _mean_std(cold_raw, ["method", "scope", "k"], cold_metrics)
    classification = classification.sort_values("f1_mean", ascending=False)
    cold = cold.sort_values(["k", "user_macro_f1_mean"], ascending=[True, False])
    classification.to_csv(output / "classification_all.csv", index=False)
    cold.to_csv(output / "cold_start_all.csv", index=False)

    sg_class_best = classification[classification["method"] == "SG-Share classification-tuned"].iloc[0]
    sg_cold_safe = cold[cold["method"] == "SG-Share cold-safe"].set_index("k")
    comparisons: list[dict[str, Any]] = []
    for _, baseline in classification[classification["scope"] == "per_user"].iterrows():
        method = baseline["method"]
        row: dict[str, Any] = {
            "per_user_method": method,
            "delta_sg_class_f1": sg_class_best["f1_mean"] - baseline["f1_mean"],
            "delta_sg_class_post_f1": sg_class_best["post_f1_mean"] - baseline["post_f1_mean"],
            "delta_sg_class_recall": sg_class_best["recall_mean"] - baseline["recall_mean"],
            "delta_sg_class_specificity": sg_class_best["specificity_mean"] - baseline["specificity_mean"],
        }
        baseline_cold = cold[
            (cold["method"] == method) & (cold["scope"] == "per_user")
        ].set_index("k")
        for k in (5.0, 10.0):
            if k in sg_cold_safe.index and k in baseline_cold.index:
                row[f"delta_sg_cold_macro_f1_k{int(k)}"] = (
                    sg_cold_safe.loc[k, "user_macro_f1_mean"]
                    - baseline_cold.loc[k, "user_macro_f1_mean"]
                )
                row[f"delta_sg_cold_macro_recall_k{int(k)}"] = (
                    sg_cold_safe.loc[k, "user_macro_recall_mean"]
                    - baseline_cold.loc[k, "user_macro_recall_mean"]
                )
                row[f"delta_sg_cold_pooled_f1_k{int(k)}"] = (
                    sg_cold_safe.loc[k, "pooled_f1_mean"]
                    - baseline_cold.loc[k, "pooled_f1_mean"]
                )
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "sg_vs_each_per_user.csv", index=False)

    class_cols = [
        "method", "scope", "f1_mean", "post_f1_mean", "recall_mean",
        "specificity_mean", "balanced_accuracy_mean",
    ]
    cold_cols = [
        "method", "scope", "user_macro_f1_mean", "user_macro_recall_mean",
        "pooled_f1_mean", "pooled_recall_mean", "pooled_specificity_mean",
    ]
    lines = [
        "# Complete GLOBEM comparison",
        "",
        "All values are means over seeds 42/43/44 on the same 8,225-event, "
        "704-user, 130-feature stored-value stream.",
        "",
        "## Overall classification",
        "",
        _md_table(
            classification, class_cols,
            ["Method", "Scope", "F1", "Post F1", "Recall", "Specificity", "Balanced Acc."],
        ),
    ]
    for k in (5.0, 10.0):
        lines.extend([
            "", f"## Cold start K={int(k)}", "",
            _md_table(
                cold[cold["k"] == k], cold_cols,
                ["Method", "Scope", "Macro F1", "Macro Recall", "Pooled F1", "Pooled Recall", "Pooled Spec."],
            ),
        ])
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- `SG-Share classification-tuned`: raw-gradient EMA alpha=0.20, lambda=2, k=4, legacy warmup eligibility.",
        "- `SG-Share cold-safe`: raw-gradient EMA alpha=0.20, lambda=2, k=4, strict eligibility.",
        "- `SG-Share cold-max (no groups)` is a diagnostic degeneration: lambda=20 leaves zero final groups and is not a group-sharing optimum.",
        "- Several global/HBP variants obtain high cold recall by predicting nearly every event positive; their pooled specificity exposes this failure mode.",
        "- OLI2DS/KOIL use author-code cores or ports. OBAL/OLIFL are paper-mechanism reproductions; OLFL is a labeled protocol surrogate; HBP is a public-equation reproduction.",
    ])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "n_classification_rows": len(classification), "n_cold_rows": len(cold),
        "n_sota_runs": len(sota_class), "n_sg_runs": len(sg_class),
        "seeds": [42, 43, 44],
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build complete GLOBEM SG-Share/SOTA tables")
    parser.add_argument("--sota-root", required=True)
    parser.add_argument("--sg-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(Path(args.sota_root), Path(args.sg_root), Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
