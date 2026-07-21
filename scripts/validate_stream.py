#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED = {
    "ces": {"users": 218, "rows": 35289},
    "globem": {"users": 704, "rows": 8225, "features": 130},
    "studentlife": {"users": 46},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    frame = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    required = {"user_id", "timestamp", "label"}
    errors = []
    if not required.issubset(frame.columns):
        errors.append(f"missing columns: {sorted(required - set(frame.columns))}")
    else:
        if not set(pd.Series(frame["label"]).dropna().unique()).issubset({0, 1}):
            errors.append("label is not binary")
        if not frame["timestamp"].is_monotonic_increasing:
            errors.append("stream is not sorted by timestamp")
    observed = {
        "rows": len(frame),
        "users": int(frame["user_id"].nunique()) if "user_id" in frame else 0,
        "features": len(frame.columns) - 3,
        "positive_rate": float(frame["label"].mean()) if "label" in frame else None,
    }
    expected = EXPECTED[args.dataset]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items() if observed.get(key) != value
    }
    report = {"dataset": args.dataset, "observed": observed, "expected": expected,
              "mismatches": mismatches, "errors": errors, "valid": not errors and not mismatches}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
