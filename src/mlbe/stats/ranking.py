"""Weighted multi-criteria ranking and Pareto analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

CRITERIA = {
    "accuracy": "higher",
    "generalization": "higher",
    "performance": "higher",
}
DEFAULT_WEIGHTS = {"accuracy": 0.5, "generalization": 0.25, "performance": 0.25}


def minmax(values: Sequence[float], higher_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.ones_like(arr)
    scaled = (arr - lo) / (hi - lo)
    return scaled if higher_better else 1.0 - scaled


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def score_models(
    rows: Sequence[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Add normalised criteria and a 0-100 weighted score to each row.

    ``accuracy`` is the mean CV accuracy, ``generalization`` is 1 - |train-val
    gap| (clipped to [0,1]) and ``performance`` is a speed score
    1 / (1 + latency_p50_ms / 10), so lower latency is better while staying on a
    bounded scale. Ties break on model name, making ranking deterministic.
    """
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    out = []
    for row in rows:
        acc = float(row.get("accuracy", 0.0))
        gap = abs(float(row.get("gap", 0.0)))
        latency = max(1e-9, float(row.get("latency_p50_ms", 0.0)))
        generalization = float(np.clip(1.0 - gap, 0.0, 1.0))
        performance = 1.0 / (1.0 + latency / 10.0)
        score = 100.0 * (
            w["accuracy"] * acc + w["generalization"] * generalization + w["performance"] * performance
        )
        out.append(
            {
                **row,
                "criteria": {
                    "accuracy": acc,
                    "generalization": generalization,
                    "performance": performance,
                },
                "weights": w,
                "score": round(float(score), 3),
            }
        )
    out.sort(key=lambda r: (-r["score"], r.get("model", ""), r.get("dataset", "")))
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def pareto_front(
    points: Sequence[dict[str, Any]],
    maximize: Sequence[str] = ("accuracy",),
    minimize: Sequence[str] = ("latency_p50_ms", "model_size_mb"),
) -> list[str]:
    """Return IDs on the Pareto front (not dominated in all objectives)."""
    ids = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            better_or_equal = all(q[k] <= p[k] for k in minimize) and all(
                q[k] >= p[k] for k in maximize
            )
            strictly_better = any(q[k] < p[k] for k in minimize) or any(q[k] > p[k] for k in maximize)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            ids.append(str(p.get("id", p.get("model", i))))
    return ids


def rank_datasets(rows: Sequence[dict[str, Any]], weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Rank models within each dataset, returning rows with per-dataset ranks."""
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    out = []
    for _dataset, group in sorted(by_dataset.items()):
        for row in score_models(group, weights):
            out.append({**row, "dataset_rank": row["rank"]})
    return out
