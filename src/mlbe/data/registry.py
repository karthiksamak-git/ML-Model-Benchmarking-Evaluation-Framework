"""Dataset registry: sklearn built-ins plus a controllable synthetic task.

Everything is offline or auto-downloaded from scikit-learn's bundled data, and
no raw dataset is ever committed (``data/`` is gitignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sklearn.datasets as skds

from ..seed import rng


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)
    image_shape: tuple[int, int] | None = None
    description: str = ""

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def is_image(self) -> bool:
        return self.image_shape is not None

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    def meta(self) -> dict[str, Any]:
        counts = np.bincount(self.y.astype(int)).tolist()
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_counts": counts,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "description": self.description,
        }


def _load_sklearn(name: str) -> Any:
    if not hasattr(skds, name):
        raise KeyError(f"unknown sklearn loader {name!r}")
    return getattr(skds, name)()


def load_dataset(
    name: str,
    spec: dict[str, Any] | Any,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Materialise one dataset from its config entry."""
    spec = spec if isinstance(spec, dict) else spec.to_dict()
    kind = spec.get("kind", "sklearn")
    if kind == "sklearn":
        bunch = _load_sklearn(spec["loader"])
        X = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(bunch.target).astype(int).ravel()
        feature_names = [str(f) for f in getattr(bunch, "feature_names", [])] or [
            f"x{i}" for i in range(X.shape[1])
        ]
        target_names = [str(t) for t in getattr(bunch, "target_names", [])] or [
            str(c) for c in np.unique(y)
        ]
        image_shape = tuple(spec["image_shape"]) if spec.get("image_shape") else None
        ds = Dataset(
            name=name,
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            image_shape=image_shape,
            description=spec.get("description", ""),
        )
    elif kind == "synthetic":
        n = int(spec.get("n_samples", 600))
        X, y = skds.make_classification(
            n_samples=n,
            n_features=int(spec.get("n_features", 20)),
            n_informative=int(spec.get("n_informative", 8)),
            n_redundant=int(spec.get("n_redundant", 2)),
            n_classes=int(spec.get("n_classes", 3)),
            class_sep=float(spec.get("class_sep", 1.0)),
            flip_y=float(spec.get("flip_y", 0.02)),
            random_state=seed,
        )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        ds = Dataset(
            name=name,
            X=X,
            y=y,
            feature_names=[f"x{i}" for i in range(X.shape[1])],
            target_names=[str(c) for c in np.unique(y)],
            description=spec.get("description", ""),
        )
    else:
        raise ValueError(f"unknown dataset kind {kind!r}")

    if max_samples is not None and ds.n_samples > max_samples:
        from .splits import stratified_subsample

        idx = stratified_subsample(ds.y, max_samples, seed=seed)
        ds = Dataset(
            name=ds.name,
            X=ds.X[idx],
            y=ds.y[idx],
            feature_names=ds.feature_names,
            target_names=ds.target_names,
            image_shape=ds.image_shape,
            description=ds.description,
        )
    return ds


def to_image_tensor(X: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Reshape flat pixel features into (n, 1, h, w) float32 arrays."""
    return np.asarray(X, dtype=np.float32).reshape(-1, 1, *image_shape)


def load_all(
    cfg: Any, max_samples: int | None = None, seed: int = 42
) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for name in cfg.datasets:
        out[name] = load_dataset(name, cfg.datasets[name], max_samples=max_samples, seed=seed)
    return out


def sample_rows(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic row subsample used by memory/latency benchmarks."""
    if n >= len(X):
        return X
    idx = rng(seed).choice(len(X), size=n, replace=False)
    return X[np.sort(idx)]
