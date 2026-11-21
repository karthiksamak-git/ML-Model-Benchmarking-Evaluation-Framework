"""Repeated stratified splits with stable IDs saved with every run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from ..seed import rng


@dataclass(frozen=True)
class Split:
    split_id: str
    dataset: str
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "dataset": self.dataset,
            "repeat": self.repeat,
            "fold": self.fold,
            "seed": self.seed,
            "n_train": int(len(self.train_idx)),
            "n_test": int(len(self.test_idx)),
            "train_idx": [int(i) for i in self.train_idx],
            "test_idx": [int(i) for i in self.test_idx],
        }


def split_id(dataset: str, repeat: int, fold: int) -> str:
    return f"{dataset}/r{repeat}/f{fold}"


def make_splits(
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    dataset: str = "dataset",
) -> list[Split]:
    """Build RepeatedStratifiedKFold splits; identical inputs -> identical folds."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits: list[Split] = []
    for i, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(i, n_splits)
        splits.append(
            Split(
                split_id=split_id(dataset, repeat, fold),
                dataset=dataset,
                repeat=repeat,
                fold=fold,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
                seed=seed,
            )
        )
    return splits


def split_checksum(splits: list[Split]) -> str:
    payload = json.dumps(
        [[s.split_id, s.train_idx.tolist(), s.test_idx.tolist()] for s in splits],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_splits(splits: list[Split], y: np.ndarray) -> None:
    """Fail loudly if splits overlap, drop samples, or lose stratification."""
    n = len(y)
    classes = np.unique(y)
    overall = {c: float(np.mean(y == c)) for c in classes}
    for s in splits:
        if np.intersect1d(s.train_idx, s.test_idx).size:
            raise ValueError(f"{s.split_id}: train/test overlap")
        if len(np.union1d(s.train_idx, s.test_idx)) != n:
            raise ValueError(f"{s.split_id}: split does not cover all samples")
        train_share = {c: float(np.mean(y[s.train_idx] == c)) for c in classes}
        for c in classes:
            if abs(train_share[c] - overall[c]) > 0.05:
                raise ValueError(f"{s.split_id}: stratification drift for class {c}")
            if not np.any(y[s.test_idx] == c):
                raise ValueError(f"{s.split_id}: class {c} missing from test fold")


def save_splits(splits: list[Split], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checksum": split_checksum(splits),
        "n_splits": len(splits),
        "splits": [s.to_json() for s in splits],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def load_splits(path: str | Path) -> list[Split]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in payload["splits"]:
        out.append(
            Split(
                split_id=item["split_id"],
                dataset=item["dataset"],
                repeat=item["repeat"],
                fold=item["fold"],
                train_idx=np.asarray(item["train_idx"], dtype=int),
                test_idx=np.asarray(item["test_idx"], dtype=int),
                seed=item["seed"],
            )
        )
    return out


@dataclass
class FoldData:
    """Per-fold arrays and the split they came from."""

    split: Split
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


def materialize(splits: list[Split], X: np.ndarray, y: np.ndarray) -> list[FoldData]:
    return [
        FoldData(
            split=s,
            X_train=X[s.train_idx],
            y_train=y[s.train_idx],
            X_test=X[s.test_idx],
            y_test=y[s.test_idx],
        )
        for s in splits
    ]


def stratified_subsample(y: np.ndarray, max_samples: int, seed: int = 42) -> np.ndarray:
    """Indices of a class-proportional subsample (used by --quick)."""
    if max_samples >= len(y):
        return np.arange(len(y))
    generator = rng(seed)
    idx = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        take = max(2, int(round(len(cls) * max_samples / len(y))))
        idx.append(generator.choice(cls, size=min(take, len(cls)), replace=False))
    out = np.sort(np.concatenate(idx))
    return out[:max_samples] if len(out) > max_samples else out
