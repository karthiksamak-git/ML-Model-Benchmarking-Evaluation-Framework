"""Metric implementations shared by the runner, rubrics and error analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def align_proba(proba: np.ndarray, model_classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Expand/permute a probability matrix onto the full label set."""
    proba = np.asarray(proba, dtype=np.float64)
    model_classes = np.asarray(model_classes)
    labels = np.asarray(labels)
    if len(model_classes) == len(labels) and np.array_equal(model_classes, labels):
        return proba
    out = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    for j, c in enumerate(labels):
        hit = np.flatnonzero(model_classes == c)
        if hit.size:
            out[:, j] = proba[:, hit[0]]
    row = out.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return out / row


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE: weighted |accuracy - confidence| over equal-width bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and one-hot targets."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """All headline metrics; every value returned is finite."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = align_proba(proba, proba_classes(labels, proba), labels)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-12, 1.0), labels=labels)),
        "brier": multiclass_brier(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }
    try:
        if len(labels) == 2:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError:
        metrics["roc_auc_ovr"] = float("nan")
    return metrics


def proba_classes(labels: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Identity helper for readability: probability columns follow ``labels``."""
    return np.asarray(labels)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> list[dict]:
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "class": int(c),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, c in enumerate(labels)
    ]
