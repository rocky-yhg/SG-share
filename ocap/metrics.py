from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if y.size == 0:
        return {key: float("nan") for key in (
            "accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1"
        )}
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / y.size,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "n": float(y.size),
    }


def first_k_metrics(
    data_triplets: Iterable[Mapping[str, object]], ks: Sequence[int], require_complete: bool = True,
) -> List[Dict[str, float]]:
    by_user: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for triplet in data_triplets:
        by_user[str(triplet["user_id"])].append(triplet)
    rows: List[Dict[str, float]] = []
    for k in sorted(set(int(v) for v in ks)):
        pooled_y: List[int] = []
        pooled_p: List[int] = []
        macro: Dict[str, List[float]] = defaultdict(list)
        included_users = 0
        for user_history in by_user.values():
            if require_complete and len(user_history) < k:
                continue
            subset = user_history[:k]
            if not subset:
                continue
            included_users += 1
            y = [int(row["label"]) for row in subset]
            p = [int(row["prediction"]) for row in subset]
            pooled_y.extend(y)
            pooled_p.extend(p)
            values = binary_metrics(y, p)
            for key in ("f1", "recall", "specificity", "balanced_accuracy"):
                macro[key].append(values[key])
        pooled = binary_metrics(pooled_y, pooled_p)
        rows.append({
            "k": float(k),
            "n_users": float(included_users),
            "n_data_triplets": float(len(pooled_y)),
            "f1_at_k": float(np.mean(macro["f1"])) if macro["f1"] else float("nan"),
            "recall_at_k": float(np.mean(macro["recall"])) if macro["recall"] else float("nan"),
            "specificity_at_k": float(np.mean(macro["specificity"])) if macro["specificity"] else float("nan"),
            "balanced_accuracy_at_k": float(np.mean(macro["balanced_accuracy"])) if macro["balanced_accuracy"] else float("nan"),
            "pooled_f1": pooled["f1"],
            "pooled_recall": pooled["recall"],
            "pooled_specificity": pooled["specificity"],
            "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        })
    return rows
