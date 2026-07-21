from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/ces_globem_parameter_comparison_20260720"
OUTPUT = ROOT / "results/offline_online_publication_tables_20260720"


OFFLINE = [
    ("GLOBEM", "ML", "XGBoost", .5370, .5240, .5239),
    ("GLOBEM", "ML", "SVM", .5702, .5317, .5238),
    ("GLOBEM", "ML", "LogisticRegression", .5297, .5247, .5227),
    ("GLOBEM", "ML", "RandomForest", .5677, .5287, .5202),
    ("GLOBEM", "ML", "LightGBM", .5377, .5182, .5180),
    ("GLOBEM", "ML", "DecisionTree", .5182, .5104, .5094),
    ("GLOBEM", "DL", "LSTM-Attention", .5724, .5753, .5441),
    ("GLOBEM", "DL", "Transformer", .5700, .5450, .5279),
    ("GLOBEM", "DL", "TCN", .5825, .5132, .5103),
    ("GLOBEM", "DL", "MLP", .5646, .5099, .5040),
    ("CES", "ML", "LightGBM", .8800, .8542, .8526),
    ("CES", "ML", "SVM", .8635, .8385, .8338),
    ("CES", "ML", "XGBoost", .8651, .8321, .8331),
    ("CES", "ML", "RandomForest", .8597, .8295, .8278),
    ("CES", "ML", "LogisticRegression", .8084, .7751, .7688),
    ("CES", "ML", "DecisionTree", .7374, .6330, .6424),
    ("CES", "DL", "TCN", .5985, .5708, .5681),
    ("CES", "DL", "Transformer", .6036, .5481, .5486),
    ("CES", "DL", "MLP", .6346, .5506, .5454),
    ("CES", "DL", "LSTM-Attention", .5258, .5749, .5257),
]


NAMES = {
    "obal": "OBAL", "olfl": "OLFL", "olifl": "OLIFL", "oli2ds": "OLI2DS",
    "koil": "KOIL", "hbp": "HBP/ODL",
}


def _decorate(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output[metric_columns] = output[metric_columns].astype(object)
    for _, indexes in frame.groupby(group_columns, sort=False).groups.items():
        group = frame.loc[indexes]
        for metric in metric_columns:
            ordered = group[metric].sort_values(ascending=False, kind="mergesort")
            if len(ordered) >= 1:
                output.loc[ordered.index[0], metric] = f"**{ordered.iloc[0]:.4f}**"
            if len(ordered) >= 2:
                output.loc[ordered.index[1], metric] = f"<u>{ordered.iloc[1]:.4f}</u>"
            for index in ordered.index[2:]:
                output.loc[index, metric] = f"{frame.loc[index, metric]:.4f}"
    return output


def offline_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(OFFLINE, columns=[
        "Dataset", "Family", "Model", "Accuracy", "Macro Recall", "Macro F1",
    ])
    decorated = _decorate(frame, ["Dataset"], ["Accuracy", "Macro Recall", "Macro F1"])
    return frame, decorated


def online_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(SOURCE / "classification_all.csv")
    frame["method"] = frame["method"].replace(NAMES)
    selected = (
        frame["scope"].isin(["global", "per_user"])
        | ((frame["dataset"] == "CES") & (frame["method"] == "SG-Share classification-tuned"))
        | ((frame["dataset"] == "GLOBEM") & (frame["method"] == "SG-Share classification-tuned"))
    )
    frame = frame[selected].copy()
    frame["scope"] = frame["scope"].replace({
        "global": "Global", "per_user": "Per-user", "group_shared": "Group-shared",
    })
    frame = frame.rename(columns={
        "dataset": "Dataset", "scope": "Scope", "method": "Model",
        "accuracy_mean": "Accuracy", "recall_mean": "Recall (+)",
        "f1_mean": "F1 (+)", "specificity_mean": "Specificity",
    })[["Dataset", "Scope", "Model", "Accuracy", "Recall (+)", "F1 (+)", "Specificity"]]
    scope_order = pd.CategoricalDtype(["Global", "Per-user", "Group-shared"], ordered=True)
    frame["Scope"] = frame["Scope"].astype(scope_order)
    frame = frame.sort_values(["Dataset", "Scope", "F1 (+)"], ascending=[True, True, False])
    frame["Scope"] = frame["Scope"].astype(str)
    decorated = _decorate(frame, ["Dataset"], ["Accuracy", "Recall (+)", "F1 (+)", "Specificity"])
    return frame, decorated


def cold_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(SOURCE / "cold_start_all.csv")
    frame["method"] = frame["method"].replace(NAMES)
    selected = (
        frame["scope"].eq("per_user")
        | ((frame["dataset"] == "CES") & (frame["method"] == "SG-Share default"))
        | ((frame["dataset"] == "GLOBEM") & (frame["method"] == "SG-Share cold-safe"))
    )
    frame = frame[selected].copy()
    frame["scope"] = frame["scope"].replace({"per_user": "Per-user", "group_shared": "Group-shared"})
    frame = frame.rename(columns={
        "dataset": "Dataset", "scope": "Scope", "method": "Model", "k": "K",
        "user_macro_f1_mean": "User-macro F1 (+)",
        "user_macro_recall_mean": "User-macro Recall (+)",
        "pooled_f1_mean": "Pooled F1 (+)", "pooled_recall_mean": "Pooled Recall (+)",
    })[[
        "Dataset", "K", "Scope", "Model", "User-macro F1 (+)",
        "User-macro Recall (+)", "Pooled F1 (+)", "Pooled Recall (+)",
    ]]
    scope_order = pd.CategoricalDtype(["Group-shared", "Per-user"], ordered=True)
    frame["Scope"] = frame["Scope"].astype(scope_order)
    frame = frame.sort_values(["Dataset", "K", "Scope", "User-macro F1 (+)"], ascending=[True, True, True, False])
    frame["Scope"] = frame["Scope"].astype(str)
    decorated = _decorate(frame, ["Dataset", "K"], [
        "User-macro F1 (+)", "User-macro Recall (+)", "Pooled F1 (+)", "Pooled Recall (+)",
    ])
    return frame, decorated


def _rank_markup(frame: pd.DataFrame, indexes, column: str) -> dict[int, str]:
    ordered = frame.loc[indexes, column].sort_values(ascending=False, kind="mergesort")
    output = {int(index): f"{value:.4f}" for index, value in ordered.items()}
    if len(ordered) >= 1:
        output[int(ordered.index[0])] = f"**{ordered.iloc[0]:.4f}**"
    if len(ordered) >= 2:
        output[int(ordered.index[1])] = f"<u>{ordered.iloc[1]:.4f}</u>"
    return output


def rq1_unified_table(
    offline_raw: pd.DataFrame, online_raw: pd.DataFrame, cold_raw: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in ("CES", "GLOBEM"):
        offline_dataset = offline_raw[offline_raw["Dataset"] == dataset]
        for family in ("ML", "DL"):
            family_rows = offline_dataset[offline_dataset["Family"] == family]
            best = family_rows.loc[family_rows["Macro F1"].idxmax()]
            rows.append({
                "Dataset": dataset, "Protocol": "Offline", "Scope": f"Best {family}",
                "Model": best["Model"], "Accuracy": best["Accuracy"],
                "Recall": best["Macro Recall"], "F1": best["Macro F1"],
            })

        online_dataset = online_raw[online_raw["Dataset"] == dataset]
        global_rows = online_dataset[online_dataset["Scope"] == "Global"]
        best_global = global_rows.loc[global_rows["F1 (+)"].idxmax()]
        selected_online = pd.concat([
            best_global.to_frame().T,
            online_dataset[online_dataset["Scope"] == "Per-user"],
            online_dataset[online_dataset["Scope"] == "Group-shared"],
        ], ignore_index=True)
        for _, online in selected_online.iterrows():
            rows.append({
                "Dataset": dataset, "Protocol": "Online", "Scope": (
                    "Best global" if online["Scope"] == "Global" else online["Scope"]
                ),
                "Model": online["Model"], "Accuracy": float(online["Accuracy"]),
                "Recall": float(online["Recall (+)"]), "F1": float(online["F1 (+)"]),
            })

    frame = pd.DataFrame(rows)
    for k in (5, 10):
        for metric in ("User-macro F1 (+)", "User-macro Recall (+)", "Pooled F1 (+)", "Pooled Recall (+)"):
            frame[f"K{k} {metric}"] = pd.NA

    for index, row in frame.iterrows():
        if row["Protocol"] != "Online" or row["Scope"] == "Best global":
            continue
        cold_model = row["Model"]
        if row["Scope"] == "Group-shared":
            cold_model = "SG-Share default" if row["Dataset"] == "CES" else "SG-Share cold-safe"
        matched = cold_raw[
            (cold_raw["Dataset"] == row["Dataset"])
            & (cold_raw["Model"] == cold_model)
        ]
        for _, cold in matched.iterrows():
            k = int(cold["K"])
            if k not in (5, 10):
                continue
            for metric in ("User-macro F1 (+)", "User-macro Recall (+)", "Pooled F1 (+)", "Pooled Recall (+)"):
                frame.loc[index, f"K{k} {metric}"] = float(cold[metric])

    display = frame.copy()
    for dataset in ("CES", "GLOBEM"):
        offline_indexes = frame.index[(frame["Dataset"] == dataset) & (frame["Protocol"] == "Offline")]
        online_indexes = frame.index[(frame["Dataset"] == dataset) & (frame["Protocol"] == "Online")]
        for column in ("Accuracy", "Recall", "F1"):
            for indexes in (offline_indexes, online_indexes):
                for index, value in _rank_markup(frame, indexes, column).items():
                    display.loc[index, column] = value
        cold_indexes = frame.index[
            (frame["Dataset"] == dataset) & (frame["Protocol"] == "Online")
            & frame["Scope"].isin(["Per-user", "Group-shared"])
        ]
        for k in (5, 10):
            for metric in ("User-macro F1 (+)", "User-macro Recall (+)", "Pooled F1 (+)", "Pooled Recall (+)"):
                column = f"K{k} {metric}"
                numeric = pd.to_numeric(frame.loc[cold_indexes, column], errors="coerce")
                valid_indexes = numeric.dropna().index
                for index, value in _rank_markup(frame, valid_indexes, column).items():
                    display.loc[index, column] = value

    def paired(row, k: int, prefix: str) -> str:
        left = row[f"K{k} {prefix} F1 (+)"]
        right = row[f"K{k} {prefix} Recall (+)"]
        if pd.isna(left) or pd.isna(right):
            return "--"
        return f"{left} / {right}"

    output = display[["Dataset", "Protocol", "Scope", "Model", "Accuracy", "Recall", "F1"]].copy()
    output["K=5 Macro F1/R"] = display.apply(lambda row: paired(row, 5, "User-macro"), axis=1)
    output["K=5 Pooled F1/R"] = display.apply(lambda row: paired(row, 5, "Pooled"), axis=1)
    output["K=10 Macro F1/R"] = display.apply(lambda row: paired(row, 10, "User-macro"), axis=1)
    output["K=10 Pooled F1/R"] = display.apply(lambda row: paired(row, 10, "Pooled"), axis=1)
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    offline_raw, offline = offline_table()
    online_raw, online = online_table()
    cold_raw, cold = cold_table()
    rq1 = rq1_unified_table(offline_raw, online_raw, cold_raw)
    offline_raw.to_csv(OUTPUT / "offline_classification.csv", index=False)
    online_raw.to_csv(OUTPUT / "online_classification.csv", index=False)
    cold_raw.to_csv(OUTPUT / "cold_start_per_user.csv", index=False)
    rq1.to_csv(OUTPUT / "rq1_unified_table.csv", index=False)
    text = "\n".join([
        "# Offline and online comparison tables", "",
        "**Formatting:** bold = best; underline = second-best within each dataset (and within each K for cold start).", "",
        "Offline Macro F1/Macro Recall and online F1(+)/Recall(+) use different protocols and are ranked separately.", "",
        "High Recall must be interpreted with Specificity: several global/HBP rows are close to all-positive prediction.", "",
        "## Offline classification", "", offline.to_markdown(index=False), "",
        "## Online prequential classification", "", online.to_markdown(index=False), "",
        "## Online cold start: SG-Share versus per-user methods", "", cold.to_markdown(index=False), "",
        "## RQ1 unified main-table draft", "",
        "Classification: offline columns are macro metrics; online columns are positive-class metrics. Cold cells are F1 / Recall. SG-Share classification uses classification-tuned parameters, while cold-start cells use the dataset-specific cold-safe parameters.", "",
        rq1.to_markdown(index=False), "",
    ])
    (OUTPUT / "tables.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
