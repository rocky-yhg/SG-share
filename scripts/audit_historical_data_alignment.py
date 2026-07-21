#!/usr/bin/env python3
"""Audit reconstructed CES/GLOBEM inputs against recovered paper contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import arff


CES_EXPECTED = {
    "rows": 35289,
    "users": 218,
    "positives": 10053,
    "features": 37,
    "positive_rate": 0.2848763070645244,
}
GLOBEM_FULL_EXPECTED = {
    "rows": 8225,
    "users": 704,
    "features": 130,
    "positive_rate": 0.4621276595744681,
}
GLOBEM_MODALITIES = [
    "bluetooth", "call", "location", "screen", "sleep", "steps", "wifi",
]


def _close(observed: float, expected: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(observed) - float(expected)) <= tolerance)


def audit_ces(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable", "path": str(path)}
    frame = pd.read_csv(path)
    features = [column for column in frame.columns if column not in {"user_id", "timestamp", "label"}]
    sensing = [column for column in features if column != "day"]
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    order = pd.DataFrame({"timestamp": timestamps, "user_id": frame["user_id"].astype(str)})
    sorted_order = order.sort_values(["timestamp", "user_id"], kind="stable").reset_index(drop=True)
    grouped_means = frame.groupby("user_id", sort=False)[sensing].mean().to_numpy()
    grouped_stds = frame.groupby("user_id", sort=False)[sensing].std(ddof=0).to_numpy()
    finite_stds = grouped_stds[np.isfinite(grouped_stds)]
    standardized_stds = finite_stds[(finite_stds > 1e-8)]
    observed = {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "positives": int(frame["label"].sum()),
        "features": int(len(features)),
        "positive_rate": float(frame["label"].mean()),
        "feature_order_starts_with_day": bool(features and features[0] == "day"),
        "chronological_stable_order": bool(order.reset_index(drop=True).equals(sorted_order)),
        "nonfinite_feature_values": int(
            (~np.isfinite(frame[features].to_numpy(dtype=float))).sum()
        ),
        "max_abs_per_user_sensing_mean": float(np.nanmax(np.abs(grouped_means))),
        "max_abs_nonconstant_std_error": (
            float(np.max(np.abs(standardized_stds - 1.0))) if standardized_stds.size else 0.0
        ),
    }
    checks = {
        "rows": observed["rows"] == CES_EXPECTED["rows"],
        "users": observed["users"] == CES_EXPECTED["users"],
        "positives": observed["positives"] == CES_EXPECTED["positives"],
        "features": observed["features"] == CES_EXPECTED["features"],
        "positive_rate": _close(observed["positive_rate"], CES_EXPECTED["positive_rate"]),
        "feature_order": observed["feature_order_starts_with_day"],
        "ordering": observed["chronological_stable_order"],
        "finite": observed["nonfinite_feature_values"] == 0,
        "per_user_centered": observed["max_abs_per_user_sensing_mean"] < 1e-6,
        "per_user_population_scaled": observed["max_abs_nonconstant_std_error"] < 1e-6,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "path": str(path),
        "expected": CES_EXPECTED,
        "observed": observed,
        "checks": checks,
    }


def audit_globem(
    aligned_path: Path,
    arff_path: Path,
    preprocessing_summary_path: Path,
    full_release: bool,
) -> Dict[str, Any]:
    required = [aligned_path, arff_path, preprocessing_summary_path]
    if not all(path.is_file() for path in required):
        return {
            "status": "unavailable",
            "paths": [str(path) for path in required],
            "full_release": bool(full_release),
        }
    aligned = pd.read_csv(aligned_path)
    summary = json.loads(preprocessing_summary_path.read_text(encoding="utf-8"))
    rows, _ = arff.loadarff(arff_path)
    final = pd.DataFrame(rows)
    feature_columns = [column for column in final.columns if column not in {"uid", "class"}]
    matrix = final[feature_columns].astype(float)
    means = matrix.mean(axis=0).to_numpy()
    stds = matrix.std(axis=0, ddof=0).to_numpy()
    nonconstant = stds > 1e-8
    uid_text = aligned["uid"].astype(str)
    expected = GLOBEM_FULL_EXPECTED if full_release else {
        "rows": 120, "users": 40, "features": 130, "positive_rate": 0.425,
    }
    observed = {
        "rows": int(len(final)),
        "users": int(final["uid"].nunique()),
        "features": int(len(feature_columns)),
        "positive_rate": float(pd.to_numeric(final["class"]).mean()),
        "aligned_rows": int(len(aligned)),
        "modalities": list(summary.get("modalities", [])),
        "drop_discrete_rank": bool(summary.get("drop_discrete_rank", False)),
        "max_missing_ratio": float(summary.get("max_missing_ratio", np.nan)),
        "uid_uses_cohort_double_colon_pid": bool(uid_text.str.contains("::", regex=False).all()),
        "nonfinite_feature_values": int((~np.isfinite(matrix.to_numpy())).sum()),
        "max_abs_global_feature_mean": float(np.max(np.abs(means))),
        "max_abs_nonconstant_std_error": (
            float(np.max(np.abs(stds[nonconstant] - 1.0))) if nonconstant.any() else 0.0
        ),
    }
    checks = {
        "rows": observed["rows"] == expected["rows"],
        "aligned_rows_preserved": observed["aligned_rows"] == observed["rows"],
        "users": observed["users"] == expected["users"],
        "features": observed["features"] == expected["features"],
        "positive_rate": _close(observed["positive_rate"], expected["positive_rate"]),
        "seven_modalities": observed["modalities"] == GLOBEM_MODALITIES,
        "drop_discrete_rank": observed["drop_discrete_rank"],
        "missing_threshold": _close(observed["max_missing_ratio"], 0.90),
        "uid_rule": observed["uid_uses_cohort_double_colon_pid"],
        "finite": observed["nonfinite_feature_values"] == 0,
        "global_centered": observed["max_abs_global_feature_mean"] < 1e-6,
        "global_population_scaled": observed["max_abs_nonconstant_std_error"] < 1e-6,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "full_release" if full_release else "official_sample_contract",
        "paths": {
            "aligned": str(aligned_path),
            "arff": str(arff_path),
            "preprocessing_summary": str(preprocessing_summary_path),
        },
        "expected": expected,
        "observed": observed,
        "checks": checks,
    }


def _markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = ["# Historical Data Processing Alignment Audit", ""]
    for dataset in ("ces", "globem"):
        item = report[dataset]
        lines.extend([
            f"## {dataset.upper()}",
            "",
            f"- Status: **{item['status']}**",
        ])
        if dataset == "globem" and item.get("scope"):
            lines.append(f"- Scope: **{item['scope']}**")
        if "observed" in item:
            for key in ("rows", "users", "features", "positives", "positive_rate"):
                if key in item["observed"]:
                    lines.append(f"- {key}: {item['observed'][key]}")
        if "checks" in item:
            failed = [key for key, passed in item["checks"].items() if not passed]
            lines.append(f"- Failed checks: {', '.join(failed) if failed else 'none'}")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ces", type=Path, default=Path("data/processed/ces_historical_usernorm.csv")
    )
    parser.add_argument(
        "--globem-aligned", type=Path,
        default=Path("data/audit/globem_sample_historical/globem_weekly_clean_all_cohorts.csv.gz"),
    )
    parser.add_argument(
        "--globem-arff", type=Path,
        default=Path("data/audit/globem_sample_historical/globem_weekly_top128.arff"),
    )
    parser.add_argument(
        "--globem-summary", type=Path,
        default=Path("data/audit/globem_sample_historical/globem_weekly_preprocessing_summary.json"),
    )
    parser.add_argument("--globem-full-release", action="store_true")
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("data/audit/historical_data_alignment_audit.json"),
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=Path("data/audit/HISTORICAL_DATA_ALIGNMENT_AUDIT.md"),
    )
    args = parser.parse_args()

    report = {
        "ces": audit_ces(args.ces),
        "globem": audit_globem(
            args.globem_aligned,
            args.globem_arff,
            args.globem_summary,
            bool(args.globem_full_release),
        ),
    }
    report["interpretation"] = (
        "CES is audited against the recovered full paper stream. GLOBEM is "
        + (
            "audited against the recovered full-release contract."
            if args.globem_full_release
            else "currently audited only as an official-sample file-contract run; "
                 "the protected full-release counts and model results remain unverified."
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(
        0 if report["ces"]["status"] == "pass" and report["globem"]["status"] == "pass" else 1
    )


if __name__ == "__main__":
    main()
