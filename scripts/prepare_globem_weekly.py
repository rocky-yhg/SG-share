from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


META_COLUMNS = {
    "cohort",
    "uid",
    "pid",
    "date",
    "label_dep_weekly",
    "target_bdi2_weekly",
    "platform",
}


def _safe_nominal(value: object) -> str:
    s = str(value)
    s = s.replace(" ", "_")
    s = s.replace(",", "_")
    s = s.replace("{", "_").replace("}", "_")
    return s


def _write_arff(
    out_path: Path,
    relation: str,
    uid_values: Iterable[str],
    feature_names: List[str],
    rows: pd.DataFrame,
) -> None:
    uid_nominal = ",".join(_safe_nominal(v) for v in sorted(set(uid_values)))
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"@relation {relation}\n\n")
        f.write(f"@attribute uid {{{uid_nominal}}}\n")
        for name in feature_names:
            f.write(f"@attribute {name} numeric\n")
        f.write("@attribute class {0,1}\n\n")
        f.write("@data\n")
        for row in rows.itertuples(index=False):
            uid = _safe_nominal(getattr(row, "uid"))
            values = [uid]
            for name in feature_names:
                val = float(getattr(row, name))
                if not np.isfinite(val):
                    val = 0.0
                values.append(f"{val:.8f}")
            label = int(getattr(row, "label_dep_weekly"))
            values.append(str(label))
            f.write(",".join(values) + "\n")


def build_globem_weekly_arff(
    input_csv: str | Path,
    output_arff: str | Path,
    top_k_features: int = 128,
    relation: str = "GLOBEM_WEEKLY_V5",
    keep_aux_features: bool = True,
) -> dict:
    input_path = Path(input_csv)
    output_path = Path(output_arff)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "cohort", "uid"], kind="stable").reset_index(drop=True)
    df["uid"] = df["uid"].astype(str)

    numeric_cols = [c for c in df.columns if c not in META_COLUMNS]
    aux_cols = [c for c in ["is_ios", "is_android"] if c in numeric_cols] if keep_aux_features else []
    behavior_cols = [c for c in numeric_cols if c not in set(aux_cols)]

    medians = df[behavior_cols].median(numeric_only=True)
    behavior_matrix = df[behavior_cols].copy()
    behavior_matrix = behavior_matrix.fillna(medians)
    variances = behavior_matrix.var(axis=0, ddof=0).sort_values(ascending=False)
    selected_behavior = variances.head(max(1, int(top_k_features))).index.tolist()
    selected_cols = aux_cols + selected_behavior

    selected_df = df[["uid", "label_dep_weekly"] + selected_cols].copy()
    selected_df[selected_cols] = selected_df[selected_cols].apply(pd.to_numeric, errors="coerce")
    selected_df[selected_cols] = selected_df[selected_cols].fillna(selected_df[selected_cols].median())

    means = selected_df[selected_cols].mean(axis=0)
    stds = selected_df[selected_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    selected_df[selected_cols] = (selected_df[selected_cols] - means) / stds
    selected_df[selected_cols] = selected_df[selected_cols].fillna(0.0)

    arff_feature_names = []
    rename_map = {}
    for idx, col in enumerate(selected_cols):
        safe_name = (
            str(col)
            .replace(":", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace(" ", "_")
        )
        safe_name = f"g_{idx:04d}_{safe_name}"
        rename_map[col] = safe_name
        arff_feature_names.append(safe_name)
    selected_df = selected_df.rename(columns=rename_map)

    _write_arff(
        out_path=output_path,
        relation=relation,
        uid_values=df["uid"].tolist(),
        feature_names=arff_feature_names,
        rows=selected_df[["uid", "label_dep_weekly"] + arff_feature_names],
    )

    manifest_rows = []
    for original_col in selected_cols:
        manifest_rows.append(
            {
                "arff_name": rename_map[original_col],
                "source_column": original_col,
                "feature_group": "aux" if original_col in aux_cols else "behavior",
                "variance_rank": int(variances.index.get_loc(original_col) + 1) if original_col in variances.index else 0,
                "mean_before_standardize": float(means[original_col]),
                "std_before_standardize": float(stds[original_col]),
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_path.with_suffix(".manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_arff": str(output_path),
        "rows": int(len(selected_df)),
        "participants": int(df["uid"].nunique()),
        "top_k_features": int(top_k_features),
        "selected_behavior_features": int(len(selected_behavior)),
        "selected_aux_features": int(len(aux_cols)),
        "total_selected_features": int(len(selected_cols)),
        "label_positive_rate": float(df["label_dep_weekly"].mean()),
        "manifest_csv": str(manifest_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="data_streams/globem_weekly/stage1/globem_weekly_clean_all_cohorts.csv.gz",
    )
    parser.add_argument(
        "--output-arff",
        default="data_streams/globem_weekly/stage1/globem_weekly_v5_top128.arff",
    )
    parser.add_argument("--top-k-features", type=int, default=128)
    args = parser.parse_args()

    summary = build_globem_weekly_arff(
        input_csv=args.input_csv,
        output_arff=args.output_arff,
        top_k_features=args.top_k_features,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
