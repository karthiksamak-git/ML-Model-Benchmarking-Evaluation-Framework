"""Generalization measurements: train/val gap, learning curves, seed
stability and robustness under noise / covariate shift."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..data.corruptions import inject_covariate_shift
from ..models.registry import build_model
from ..seed import rng


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def generalization_gap(train_scores: dict[str, float], test_scores: dict[str, float]) -> dict[str, float]:
    """Positive values mean train performance is higher (overfitting)."""
    keys = [k for k in ("accuracy", "f1_macro") if k in train_scores and k in test_scores]
    return {f"gap_{k}": float(train_scores[k] - test_scores[k]) for k in keys}


def seed_stability(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "range": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "range": float(arr.max() - arr.min()),
    }


def learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seed: int = 0,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    cv_folds: int = 3,
    model_params: dict[str, Any] | None = None,
    dataset_name: str = "",
) -> list[dict[str, Any]]:
    """Train on increasing fractions of each training fold and score the fold."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    image_shape = None
    if model_name == "cnn":
        side = int(round(np.sqrt(X.shape[1])))
        image_shape = (side, side)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        for frac in fractions:
            n_take = max(n_classes * 2, int(round(frac * len(tr))))
            if n_take >= len(tr):
                idx = tr
            else:
                gen = rng(seed + fold)
                idx = np.sort(gen.choice(tr, size=n_take, replace=False))
                # keep every class present
                for c in np.unique(y[tr]):
                    if not np.any(y[idx] == c):
                        extra = tr[y[tr] == c][:1]
                        idx = np.union1d(idx, extra)
            model = build_model(
                model_name,
                seed=seed + fold,
                params=model_params,
                input_dim=X.shape[1],
                n_classes=n_classes,
                image_shape=image_shape,
            )
            model.fit(X[idx], y[idx])
            pred = model.predict(X[te])
            scores = metrics_for(y[te], pred)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold,
                    "fraction": float(frac),
                    "n_train": int(len(idx)),
                    **scores,
                }
            )
    return rows


def robustness_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    seed: int = 0,
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
    shift_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    model_params: dict[str, Any] | None = None,
    dataset_name: str = "",
) -> list[dict[str, Any]]:
    """Accuracy under Gaussian feature noise and covariate shift."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    n_classes = len(np.unique(y_train))
    image_shape = None
    if model_name == "cnn":
        image_shape = (int(round(np.sqrt(X_train.shape[1]))),) * 2
    model = build_model(
        model_name,
        seed=seed,
        params=model_params,
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        image_shape=image_shape,
    )
    model.fit(X_train, y_train)
    std = np.nanstd(X_train, axis=0)
    std[std == 0] = 1.0
    rows: list[dict[str, Any]] = []
    baseline = None
    gen = rng(seed)
    for level in noise_levels:
        noisy = X_test + gen.normal(0.0, 1.0, size=X_test.shape) * std * level
        acc = float(accuracy_score(y_test, model.predict(noisy)))
        baseline = acc if baseline is None else baseline
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "kind": "gaussian_noise",
                "level": float(level),
                "accuracy": acc,
                "drop": float(baseline - acc),
            }
        )
    for level in shift_levels:
        if level == 0:
            shifted = X_test.copy()
        else:
            shifted, _ = inject_covariate_shift(X_train, X_test, shift=level, seed=seed + 11)
        acc = float(accuracy_score(y_test, model.predict(shifted)))
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "kind": "covariate_shift",
                "level": float(level),
                "accuracy": acc,
                "drop": float(baseline - acc),
            }
        )
    return rows


def train_val_gap(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict[str, float]:  # pragma: no cover - convenience for notebooks
    """Utility used interactively: split once and report the gap."""
    if X_val is None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
        )
    else:
        X_tr, y_tr = X_train, y_train
    model = fit_predict(X_tr, y_tr, X_val)  # type: ignore[arg-type]
    train_scores = metrics_for(y_tr, model.predict(X_tr))
    val_scores = metrics_for(y_val, model.predict(X_val))
    return {**{f"train_{k}": v for k, v in train_scores.items()},
            **{f"test_{k}": v for k, v in val_scores.items()},
            **generalization_gap(train_scores, val_scores)}
