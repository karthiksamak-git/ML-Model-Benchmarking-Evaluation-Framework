"""Systematic error analysis: confusion structure, confident errors, clusters,
calibration and slice metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from ..metrics import expected_calibration_error, per_class_report


def reliability_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return rows


def slice_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    max_features: int = 3,
    n_bins: int = 4,
) -> list[dict[str, Any]]:
    """Accuracy by quartile of the most target-correlated features."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    y_true = np.asarray(y_true)
    correct = (np.asarray(y_pred) == y_true).astype(float)
    corr = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            continue
        y = y_true.astype(float)
        if len(np.unique(y)) > 2:
            y = (y == np.bincount(y_true.astype(int)).argmax()).astype(float)
        r = abs(np.corrcoef(col, y)[0, 1])
        if np.isfinite(r):
            corr.append((r, j))
    corr.sort(reverse=True)
    out = []
    for _, j in corr[:max_features]:
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        bins = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = bins == b
            if mask.sum() == 0:
                continue
            out.append(
                {
                    "feature": feature_names[j] if j < len(feature_names) else f"x{j}",
                    "quartile": int(b),
                    "range": [float(edges[b]), float(edges[b + 1])],
                    "n": int(mask.sum()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return out


def error_clusters(
    X: np.ndarray, feature_names: list[str], n_clusters: int = 3, seed: int = 0
) -> dict[str, Any]:
    """Cluster misclassified samples in feature space and name the extremes."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    if len(X) < 3:
        return {"n_errors": int(len(X)), "clusters": []}
    k = int(min(n_clusters, max(1, len(X) // 3)))
    scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(scaled)
    clusters = []
    for c in range(k):
        mask = km.labels_ == c
        if not mask.any():
            continue
        centre = scaled[mask].mean(axis=0)
        top = np.argsort(-np.abs(centre))[:3]
        clusters.append(
            {
                "cluster": int(c),
                "n": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "defining_features": [
                    {
                        "feature": feature_names[j] if j < len(feature_names) else f"x{j}",
                        "mean_z": round(float(centre[j]), 3),
                    }
                    for j in top
                ],
            }
        )
    return {"n_errors": int(len(X)), "clusters": clusters}


def analyze_predictions(
    dataset: str,
    model: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    seed: int = 0,
) -> dict[str, Any]:
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    labels = np.asarray(labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    errors = np.flatnonzero(y_pred != y_test)
    conf = proba.max(axis=1)
    order = errors[np.argsort(-conf[errors])][:top_n]

    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": int(labels[i]),
                        "predicted": int(labels[j]),
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda d: -d["count"])

    return {
        "dataset": dataset,
        "model": model,
        "n_test": int(len(y_test)),
        "accuracy": float(np.mean(y_pred == y_test)),
        "labels": [int(v) for v in labels],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report(y_test, y_pred, labels),
        "top_confident_errors": [
            {
                "index": int(i),
                "true": int(y_test[i]),
                "predicted": int(y_pred[i]),
                "confidence": round(float(conf[i]), 4),
                "probabilities": [round(float(p), 4) for p in proba[i]],
            }
            for i in order
        ],
        "most_confused_pairs": pairs[:8],
        "error_clusters": error_clusters(
            np.asarray(X_test)[errors], feature_names, seed=seed
        ),
        "calibration": {
            "ece": float(expected_calibration_error(y_test, proba)),
            "bins": reliability_bins(y_test, proba),
        },
        "slices": slice_metrics(np.asarray(X_test), y_test, y_pred, feature_names),
    }


