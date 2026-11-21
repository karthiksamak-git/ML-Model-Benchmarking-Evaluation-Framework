"""Deterministic, seeded corruption injection.

Every injector returns the corrupted arrays *and* a manifest that records exactly
which samples were altered, so rubric checks can be validated against ground
truth (including precision/recall of noisy-label identification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ..seed import rng


@dataclass
class CorruptedData:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    def defect(self, kind: str) -> dict[str, Any] | None:
        for d in self.manifest.get("defects", []):
            if d["type"] == kind:
                return d
        return None

    def noisy_indices(self) -> np.ndarray:
        out: list[int] = []
        for d in self.manifest.get("defects", []):
            if d["type"] == "label_noise":
                out.extend(int(i) for i in d["indices"])
        return np.asarray(sorted(set(out)), dtype=int)


def _pick(seed: int, n: int, size: int) -> np.ndarray:
    size = int(min(max(size, 0), n))
    return np.sort(rng(seed).choice(n, size=size, replace=False))


def inject_label_noise(
    X: np.ndarray, y: np.ndarray, rate: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Flip a fraction of labels to a different class (uniformly chosen)."""
    X = np.asarray(X).copy()
    y = np.asarray(y).copy()
    classes = np.unique(y)
    n = len(y)
    idx = _pick(seed, n, int(round(rate * n)))
    gen = rng(seed + 1)
    new = y.copy()
    for i in idx:
        choices = classes[classes != y[i]]
        new[i] = gen.choice(choices)
    changed = idx[new[idx] != y[idx]]
    original = y[changed].copy()
    y[changed] = new[changed]
    manifest = {
        "type": "label_noise",
        "rate": float(rate),
        "n_affected": int(len(changed)),
        "indices": [int(i) for i in changed],
        "original_labels": [int(v) for v in original],
        "new_labels": [int(v) for v in y[changed]],
        "seed": seed,
    }
    return X, y, manifest


def inject_duplicates(
    X: np.ndarray,
    y: np.ndarray,
    rate: float,
    seed: int = 0,
    conflict_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Append exact copies of existing rows; optionally flip labels on copies."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    n_dup = int(round(rate * n))
    src = _pick(seed, n, n_dup)
    if len(src) == 0:
        return X.copy(), y.copy(), {
            "type": "duplicates",
            "rate": float(rate),
            "n_duplicated": 0,
            "source_indices": [],
            "new_row_indices": [],
            "conflicting_new_row_indices": [],
            "seed": seed,
        }
    classes = np.unique(y)
    gen = rng(seed + 7)
    new_y = y[src].copy()
    conflicting = []
    if conflict_rate > 0:
        k = int(round(conflict_rate * len(src)))
        conf_pos = _pick(seed + 3, len(src), k)
        for pos in conf_pos:
            cls = y[src[pos]]
            options = classes[classes != cls]
            new_y[pos] = gen.choice(options)
            conflicting.append(int(n + pos))
    X_new = np.vstack([X, X[src]])
    y_new = np.concatenate([y, new_y])
    manifest = {
        "type": "duplicates",
        "rate": float(rate),
        "n_duplicated": int(len(src)),
        "source_indices": [int(i) for i in src],
        "new_row_indices": [int(n + i) for i in range(len(src))],
        "conflicting_new_row_indices": conflicting,
        "seed": seed,
    }
    return X_new, y_new, manifest


def add_missing_values(
    X: np.ndarray, rate: float, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Set a fraction of individual cells to NaN."""
    X = np.asarray(X, dtype=np.float64).copy()
    n, d = X.shape
    total = n * d
    n_missing = int(round(rate * total))
    flat = rng(seed).choice(total, size=n_missing, replace=False)
    rows, cols = np.unravel_index(flat, (n, d))
    X[rows, cols] = np.nan
    affected_rows = np.unique(rows)
    manifest = {
        "type": "missing_values",
        "rate": float(rate),
        "n_missing_cells": int(n_missing),
        "n_cells": int(total),
        "n_affected_rows": int(len(affected_rows)),
        "affected_rows": [int(r) for r in affected_rows],
        "affected_columns": [int(c) for c in np.unique(cols)],
        "seed": seed,
    }
    return X, manifest


def inject_outliers(
    X: np.ndarray,
    y: np.ndarray,
    rate: float,
    seed: int = 0,
    magnitude: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Move a fraction of samples far outside the feature distribution."""
    X = np.asarray(X, dtype=np.float64).copy()
    y = np.asarray(y)
    n, d = X.shape
    rows = _pick(seed, n, int(round(rate * n)))
    gen = rng(seed + 2)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    cols = gen.integers(0, d, size=len(rows))
    signs = gen.choice([-1.0, 1.0], size=len(rows))
    X[rows, cols] = mean[cols] + signs * magnitude * std[cols]
    manifest = {
        "type": "outliers",
        "rate": float(rate),
        "magnitude": float(magnitude),
        "n_affected": int(len(rows)),
        "indices": [int(i) for i in rows],
        "columns": [int(c) for c in cols],
        "seed": seed,
    }
    return X, y.copy(), manifest


def inject_leaky_feature(
    X: np.ndarray, y: np.ndarray, seed: int = 0, noise: float = 0.01
) -> tuple[np.ndarray, dict[str, Any]]:
    """Append a feature that is almost a copy of the target (textbook leakage)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    gen = rng(seed)
    leak = y.astype(float) + gen.normal(0, noise, size=len(y))
    X_new = np.hstack([X, leak.reshape(-1, 1)])
    manifest = {
        "type": "leakage",
        "kind": "leaky_feature",
        "feature_index": int(X.shape[1]),
        "n_features": int(X_new.shape[1]),
        "seed": seed,
    }
    return X_new, manifest


def inject_covariate_shift(
    X_train: np.ndarray,
    X_test: np.ndarray,
    shift: float = 1.5,
    seed: int = 0,
    feature_fraction: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shift a subset of test features by ``shift`` training standard deviations."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64).copy()
    d = X_train.shape[1]
    n_shift = max(1, int(round(feature_fraction * d)))
    cols = np.sort(rng(seed).choice(d, size=n_shift, replace=False))
    std = np.nanstd(X_train, axis=0)
    std[std == 0] = 1.0
    X_test[:, cols] += shift * std[cols]
    manifest = {
        "type": "covariate_shift",
        "shift_std": float(shift),
        "feature_fraction": float(feature_fraction),
        "columns": [int(c) for c in cols],
        "seed": seed,
    }
    return X_test, manifest


def inject_imbalance(
    X: np.ndarray, y: np.ndarray, seed: int = 0, minority_keep: float = 0.08
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Downsample the majority class to create severe imbalance."""
    X = np.asarray(X)
    y = np.asarray(y)
    counts = np.bincount(y.astype(int))
    majority = int(np.argmax(counts))
    maj_idx = np.flatnonzero(y == majority)
    keep = int(max(4, round(len(maj_idx) * minority_keep)))
    kept_maj = np.sort(rng(seed).choice(maj_idx, size=keep, replace=False))
    other = np.flatnonzero(y != majority)
    idx = np.sort(np.concatenate([kept_maj, other]))
    manifest = {
        "type": "class_imbalance",
        "majority_class": majority,
        "minority_keep": float(minority_keep),
        "removed": int(len(maj_idx) - keep),
        "seed": seed,
    }
    return X[idx], y[idx], manifest


def inject_overlap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rate: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Copy training rows into the test set (train/test leakage)."""
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test).copy()
    y_test = np.asarray(y_test).copy()
    n_copy = int(round(rate * len(X_train)))
    src = _pick(seed, len(X_train), n_copy)
    if len(src):
        X_test = np.vstack([X_test, X_train[src]])
        y_test = np.concatenate([y_test, y_train[src]])
    manifest = {
        "type": "leakage",
        "kind": "train_test_overlap",
        "rate": float(rate),
        "n_copied": int(len(src)),
        "source_train_indices": [int(i) for i in src],
        "copied_test_indices": [int(len(X_test) - len(src) + i) for i in range(len(src))],
        "seed": seed,
    }
    return X_test, y_test, manifest


# ---------------------------------------------------------------------------
# Planted-defect benchmark used by rubric validation
# ---------------------------------------------------------------------------

DEFECT_SPECS = {
    "clean": None,
    "label_noise_5": ("label_noise", 0.05),
    "label_noise_10": ("label_noise", 0.10),
    "label_noise_20": ("label_noise", 0.20),
    "duplicates_10": ("duplicates", 0.10),
    "conflicting_duplicates_5": ("conflicting_duplicates", 0.05),
    "leaky_feature": ("leaky_feature", 0.0),
    "missing_15": ("missing", 0.15),
    "outliers_10": ("outliers", 0.10),
    "covariate_shift": ("covariate_shift", 1.5),
    "imbalance": ("imbalance", 0.02),
    "train_test_overlap_2": ("overlap", 0.02),
}


def make_planted_dataset(
    defect: str, seed: int = 42, source: str = "breast_cancer", test_size: float = 0.3
) -> CorruptedData:
    """Build one reproducible dataset with a known, manifested defect."""
    from .registry import load_dataset

    if defect not in DEFECT_SPECS:
        raise KeyError(f"unknown defect {defect!r}; choose from {sorted(DEFECT_SPECS)}")
    spec = DEFECT_SPECS[defect]
    ds = load_dataset(source, {"kind": "sklearn", "loader": f"load_{source}"}, seed=seed)
    X, y = ds.X, ds.y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    manifest: dict[str, Any] = {
        "defect_name": defect,
        "source": source,
        "seed": seed,
        "defects": [],
    }
    if spec is None:
        pass
    elif spec[0] == "label_noise":
        X_train, y_train, m = inject_label_noise(X_train, y_train, spec[1], seed=seed)
        manifest["defects"].append(m)
    elif spec[0] == "duplicates":
        X_train, y_train, m = inject_duplicates(X_train, y_train, spec[1], seed=seed)
        manifest["defects"].append(m)
    elif spec[0] == "conflicting_duplicates":
        X_train, y_train, m = inject_duplicates(
            X_train, y_train, spec[1], seed=seed, conflict_rate=0.5
        )
        manifest["defects"].append(m)
    elif spec[0] == "leaky_feature":
        X_train, m1 = inject_leaky_feature(X_train, y_train, seed=seed, noise=0.01)
        X_test_rows, m2 = inject_leaky_feature(X_test, y_test, seed=seed + 1, noise=0.01)
        X_test = X_test_rows
        manifest["defects"].extend([m1, m2])
    elif spec[0] == "missing":
        X_train, m = add_missing_values(X_train, spec[1], seed=seed)
        manifest["defects"].append(m)
    elif spec[0] == "outliers":
        X_train, y_train, m = inject_outliers(X_train, y_train, spec[1], seed=seed)
        manifest["defects"].append(m)
    elif spec[0] == "covariate_shift":
        X_test, m = inject_covariate_shift(X_train, X_test, shift=spec[1], seed=seed)
        manifest["defects"].append(m)
    elif spec[0] == "imbalance":
        X_train, y_train, m = inject_imbalance(X_train, y_train, seed=seed, minority_keep=spec[1])
        manifest["defects"].append(m)
    elif spec[0] == "overlap":
        X_test, y_test, m = inject_overlap(X_train, y_train, X_test, y_test, spec[1], seed=seed)
        manifest["defects"].append(m)

    feature_names = list(getattr(ds, "feature_names", [])) or [f"x{i}" for i in range(X.shape[1])]
    for d in manifest["defects"]:
        if d["type"] == "leakage" and d.get("kind") == "leaky_feature":
            feature_names = feature_names + ["leak_target"]
            break
    return CorruptedData(
        name=defect,
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=int),
        X_test=np.asarray(X_test, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=feature_names,
        manifest=manifest,
    )


def all_planted_datasets(seed: int = 42) -> list[CorruptedData]:
    return [make_planted_dataset(name, seed=seed) for name in DEFECT_SPECS]
