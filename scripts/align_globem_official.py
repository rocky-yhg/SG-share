from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_ROOT = Path(
    "数据集2/globem-dataset-multi-year-datasets-for-longitudinal-human-behavior-modeling-generalization-1.1"
)
DEFAULT_OUT = Path("data_streams/globem_weekly/stage1")
DEFAULT_MODALITIES = [
    "bluetooth",
    "call",
    "location",
    "screen",
    "sleep",
    "steps",
    "wifi",
]


SEMANTIC_BUCKET_BY_MODALITY = {
    "bluetooth": "social_communication",
    "call": "social_communication",
    "location": "mobility_routine",
    "screen": "phone_digital_behavior",
    "sleep": "sleep_circadian",
    "steps": "physical_activity",
    "wifi": "context_environment",
    "rapids": "composite_derived",
}


def _drop_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")


def _normalize_pid_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pid"] = out["pid"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def _load_weekly_labels(cohort_dir: Path) -> pd.DataFrame:
    weekly = pd.read_csv(cohort_dir / "SurveyData" / "dep_weekly.csv", low_memory=False)
    weekly = _drop_index_columns(weekly)
    weekly = _normalize_pid_date(weekly)
    weekly = weekly.dropna(subset=["pid", "date"]).copy()
    weekly["label_dep_weekly"] = weekly["dep"].astype(str).str.lower().eq("true").astype("int8")
    keep_cols = ["pid", "date", "label_dep_weekly"]
    if "BDI2" in weekly.columns:
        weekly["target_bdi2_weekly"] = pd.to_numeric(weekly["BDI2"], errors="coerce")
        keep_cols.append("target_bdi2_weekly")
    weekly = weekly[keep_cols].drop_duplicates(subset=["pid", "date"], keep="last")
    return weekly


def _load_platform(cohort_dir: Path) -> pd.DataFrame:
    platform = pd.read_csv(cohort_dir / "ParticipantsInfoData" / "platform.csv", low_memory=False)
    platform = _drop_index_columns(platform)
    platform["pid"] = platform["pid"].astype(str)
    platform["platform"] = platform["platform"].astype(str).str.lower()
    platform = platform.drop_duplicates(subset=["pid"], keep="last")
    platform["is_ios"] = platform["platform"].eq("ios").astype("int8")
    platform["is_android"] = platform["platform"].eq("android").astype("int8")
    return platform[["pid", "platform", "is_ios", "is_android"]]


def _coerce_numeric_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    bad_cols: List[str] = []
    converted: Dict[str, pd.Series] = {}
    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if int(numeric.notna().sum()) < int(df[col].notna().sum()):
            bad_cols.append(col)
            continue
        converted[col] = numeric.astype("float32")
    out = df.drop(columns=bad_cols, errors="ignore").copy()
    for col, series in converted.items():
        out[col] = series
    return out, bad_cols


def _load_feature_table(
    cohort_dir: Path,
    modality: str,
    weekly_keys: pd.DataFrame,
    max_missing_ratio: float,
    drop_discrete_rank: bool,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    path = cohort_dir / "FeatureData" / f"{modality}.csv"
    df = pd.read_csv(path, low_memory=False)
    raw_rows = int(len(df))
    df = _drop_index_columns(df)
    df = _normalize_pid_date(df)
    df = df.merge(weekly_keys, on=["pid", "date"], how="inner")

    feature_cols = [c for c in df.columns if c not in {"pid", "date"}]
    dropped_discrete: List[str] = []
    if drop_discrete_rank:
        dropped_discrete = [c for c in feature_cols if "_dis:" in c]
        if dropped_discrete:
            df = df.drop(columns=dropped_discrete)
            feature_cols = [c for c in feature_cols if c not in dropped_discrete]

    df, dropped_mixed = _coerce_numeric_features(df, feature_cols)
    feature_cols = [c for c in df.columns if c not in {"pid", "date"}]

    na_ratio = df[feature_cols].isna().mean()
    dropped_high_missing = na_ratio[na_ratio > max_missing_ratio].index.tolist()
    if dropped_high_missing:
        df = df.drop(columns=dropped_high_missing)
        feature_cols = [c for c in feature_cols if c not in dropped_high_missing]

    manifest = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "cohort": cohort_dir.name,
            "source_table": modality,
            "semantic_bucket": SEMANTIC_BUCKET_BY_MODALITY.get(modality, modality),
        }
    )
    summary = {
        "source_rows": raw_rows,
        "matched_weekly_rows": int(len(df)),
        "initial_feature_columns": int(len([c for c in pd.read_csv(path, nrows=1).columns if not c.startswith("Unnamed:")])) - 2,
        "dropped_discrete_rank_columns": int(len(dropped_discrete)),
        "dropped_mixed_type_columns": int(len(dropped_mixed)),
        "dropped_high_missing_columns": int(len(dropped_high_missing)),
        "remaining_feature_columns": int(len(feature_cols)),
        "mean_missing_ratio_after_filter": float(df[feature_cols].isna().mean().mean()) if feature_cols else 0.0,
    }
    return df, summary, manifest


def _prepare_one_cohort(
    cohort_dir: Path,
    modalities: List[str],
    max_missing_ratio: float,
    drop_discrete_rank: bool,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    weekly = _load_weekly_labels(cohort_dir)
    platform = _load_platform(cohort_dir)

    base = weekly.copy()
    keys = weekly[["pid", "date"]].copy()
    manifests: List[pd.DataFrame] = []
    feature_summaries: Dict[str, object] = {}

    for modality in modalities:
        feat_df, feat_summary, manifest = _load_feature_table(
            cohort_dir=cohort_dir,
            modality=modality,
            weekly_keys=keys,
            max_missing_ratio=max_missing_ratio,
            drop_discrete_rank=drop_discrete_rank,
        )
        feat_cols = [c for c in feat_df.columns if c not in {"pid", "date"}]
        base = base.merge(feat_df[["pid", "date"] + feat_cols], on=["pid", "date"], how="left")
        feature_summaries[modality] = feat_summary
        manifests.append(manifest)

    base = base.merge(platform, on="pid", how="left")
    base["cohort"] = cohort_dir.name
    base["uid"] = base["cohort"] + "::" + base["pid"]
    base = base.sort_values(["uid", "date"]).reset_index(drop=True)

    ordered_front = [
        "cohort",
        "uid",
        "pid",
        "date",
        "label_dep_weekly",
        "target_bdi2_weekly",
        "platform",
        "is_ios",
        "is_android",
    ]
    ordered_front = [c for c in ordered_front if c in base.columns]
    other_cols = [c for c in base.columns if c not in ordered_front]
    base = base[ordered_front + other_cols]

    summary = {
        "cohort": cohort_dir.name,
        "weekly_rows_before_join": int(len(weekly)),
        "weekly_rows_after_join": int(len(base)),
        "participants_in_weekly": int(weekly["pid"].nunique()),
        "participants_in_output": int(base["uid"].nunique()),
        "positive_rate": float(base["label_dep_weekly"].mean()) if len(base) else 0.0,
        "feature_tables": feature_summaries,
        "final_feature_columns": int(
            len(
                [
                    c
                    for c in base.columns
                    if c
                    not in {
                        "cohort",
                        "uid",
                        "pid",
                        "date",
                        "label_dep_weekly",
                        "target_bdi2_weekly",
                        "platform",
                        "is_ios",
                        "is_android",
                    }
                ]
            )
        ),
    }
    manifest_df = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    return base, summary, manifest_df


def build_dataset(
    root: Path,
    out_dir: Path,
    modalities: List[str],
    max_missing_ratio: float,
    drop_discrete_rank: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_out = out_dir / "cohorts"
    cohort_out.mkdir(parents=True, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    all_manifests: List[pd.DataFrame] = []
    summary: Dict[str, object] = {
        "source_root": str(root),
        "modalities": modalities,
        "max_missing_ratio": max_missing_ratio,
        "drop_discrete_rank": drop_discrete_rank,
        "cohorts": {},
    }

    for cohort_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        frame, cohort_summary, manifest = _prepare_one_cohort(
            cohort_dir=cohort_dir,
            modalities=modalities,
            max_missing_ratio=max_missing_ratio,
            drop_discrete_rank=drop_discrete_rank,
        )
        cohort_name = cohort_dir.name
        frame.to_csv(cohort_out / f"{cohort_name}_weekly_clean.csv.gz", index=False, compression="gzip")
        manifest.to_csv(cohort_out / f"{cohort_name}_feature_manifest.csv", index=False)
        summary["cohorts"][cohort_name] = cohort_summary
        all_frames.append(frame)
        all_manifests.append(manifest)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["uid", "date"]).reset_index(drop=True)
    combined.to_csv(out_dir / "globem_weekly_clean_all_cohorts.csv.gz", index=False, compression="gzip")

    manifest_all = pd.concat(all_manifests, ignore_index=True)
    manifest_all = manifest_all.drop_duplicates(subset=["feature_name", "source_table"])
    manifest_all.to_csv(out_dir / "globem_weekly_feature_manifest.csv", index=False)

    summary["combined"] = {
        "rows": int(len(combined)),
        "participants": int(combined["uid"].nunique()),
        "cohorts": sorted(combined["cohort"].unique().tolist()),
        "positive_rate": float(combined["label_dep_weekly"].mean()) if len(combined) else 0.0,
        "feature_columns": int(
            len(
                [
                    c
                    for c in combined.columns
                    if c
                    not in {
                        "cohort",
                        "uid",
                        "pid",
                        "date",
                        "label_dep_weekly",
                        "target_bdi2_weekly",
                        "platform",
                        "is_ios",
                        "is_android",
                    }
                ]
            )
        ),
    }
    (out_dir / "globem_weekly_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GLOBEM weekly dataset for modeling.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=DEFAULT_MODALITIES,
        help="FeatureData tables to include. Default excludes rapids.",
    )
    parser.add_argument("--max-missing-ratio", type=float, default=0.90)
    parser.add_argument(
        "--keep-discrete-rank",
        action="store_true",
        help="Keep *_dis:* ordinal rank columns instead of dropping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        root=args.root,
        out_dir=args.out_dir,
        modalities=list(args.modalities),
        max_missing_ratio=float(args.max_missing_ratio),
        drop_discrete_rank=not bool(args.keep_discrete_rank),
    )


if __name__ == "__main__":
    main()
