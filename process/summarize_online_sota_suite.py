from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgshare.metrics import first_k_metrics


STATUS = {
    "oli2ds": "author-code core; complete-feature specialization",
    "obal": "independent mechanism reproduction; no public author code found",
    "hbp": "public-code equation reproduction; HBP-19/100",
    "koil": "author-code Python port; FIFO compensation",
    "olifl": "complete-feature/label protocol specialization; no public author code found",
    "olfl": "labeled protocol surrogate; original method is label-free",
    "sg_share": "restored full_final run",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the CES online-SOTA suite with SG-Share")
    parser.add_argument("--suite", default="results/online_sota_ces_20260720")
    parser.add_argument("--hbp19", default="results/online_sota_ces_hbp19_20260720")
    parser.add_argument("--sg", default="results/real_ces_full_final_restored_v4_seed42")
    parser.add_argument("--output", default="results/online_sota_ces_20260720/final_comparison")
    args = parser.parse_args()

    suite = pd.read_csv(Path(args.suite) / "summary.csv")
    cold = pd.read_csv(Path(args.suite) / "cold_start.csv")
    hbp = pd.read_csv(Path(args.hbp19) / "summary.csv")
    hbp_cold = pd.read_csv(Path(args.hbp19) / "cold_start.csv")
    suite = pd.concat([suite[suite.method != "hbp"], hbp], ignore_index=True)
    cold = pd.concat([cold[cold.method != "hbp"], hbp_cold], ignore_index=True)

    sg_root = Path(args.sg)
    with (sg_root / "metrics.json").open("r", encoding="utf-8") as handle:
        sg_metrics = json.load(handle)
    overall, post = sg_metrics["overall"], sg_metrics["post_warmup"]
    sg_row = {
        "dataset": "ces", "method": "sg_share", "scope": "group_shared", "seed": 42,
        "provenance": STATUS["sg_share"], "n_events": sg_metrics["n_events"],
        "n_users": sg_metrics["n_users"], "n_models": float("nan"),
        "f1": overall["f1"], "recall": overall["recall"],
        "specificity": overall["specificity"], "balanced_accuracy": overall["balanced_accuracy"],
        "accuracy": overall["accuracy"], "post_f1": post["f1"],
        "post_recall": post["recall"], "post_specificity": post["specificity"],
        "post_balanced_accuracy": post["balanced_accuracy"],
    }
    suite = pd.concat([suite, pd.DataFrame([sg_row])], ignore_index=True)

    events = pd.read_csv(sg_root / "events.csv").to_dict("records")
    sg_cold = pd.DataFrame(first_k_metrics(events, [5, 10, 20, 50], True))
    sg_cold.insert(0, "seed", 42)
    sg_cold.insert(0, "scope", "group_shared")
    sg_cold.insert(0, "method", "sg_share")
    sg_cold.insert(0, "dataset", "ces")
    cold = pd.concat([cold, sg_cold], ignore_index=True)

    suite["reproduction_status"] = suite.method.map(STATUS)
    cold["reproduction_status"] = cold.method.map(STATUS)
    suite = suite.sort_values(["scope", "f1"], ascending=[True, False])
    cold = cold.sort_values(["k", "scope", "user_macro_f1"], ascending=[True, True, False])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    suite.to_csv(output / "classification.csv", index=False)
    cold.to_csv(output / "cold_start.csv", index=False)

    columns = ["method", "scope", "f1", "recall", "specificity", "balanced_accuracy", "post_f1"]
    cold_columns = ["method", "scope", "k", "user_macro_f1", "pooled_f1",
                    "user_macro_recall", "pooled_recall"]
    text = ["# CES online baseline comparison", "", "## Classification", "",
            suite[columns].to_markdown(index=False, floatfmt=".4f"), "",
            "## First-K cold start", "",
            cold[cold_columns].to_markdown(index=False, floatfmt=".4f"), "",
            "## Reproduction boundary", ""]
    for method, status in STATUS.items():
        text.append(f"- `{method}`: {status}.")
    text.extend([
        "",
        "All rows use predict-then-observe-then-update, the same causal online standardizer,",
        "the same causal window-F1 threshold protocol, and positive-class F1 as the primary metric.",
        "SG-Share first-K values are recomputed from its event log with the same function used by the baselines.",
    ])
    (output / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
