"""scikit-learn model factories. All models are wrapped in a scaling pipeline
so preprocessing is fitted inside each CV fold (no leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 2000},
    "svm": {"C": 1.0, "gamma": "scale"},
    "knn": {"n_neighbors": 5, "weights": "distance"},
    "rf": {
        "n_estimators": 150,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "hgb": {
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "max_iter": 150,
        "min_samples_leaf": 5,
        "l2_regularization": 1e-6,
    },
}

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "RBF SVM",
    "knn": "k-Nearest Neighbours",
    "rf": "Random Forest",
    "hgb": "HistGradientBoosting",
    "mlp": "PyTorch MLP",
    "cnn": "PyTorch CNN",
}


def _classifier(name: str, seed: int, params: dict[str, Any] | None = None) -> Any:
    p = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    if name == "logreg":
        return LogisticRegression(random_state=seed, solver="lbfgs", **p)
    if name == "svm":
        return SVC(random_state=seed, probability=True, cache_size=500, **p)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=1, **p)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            random_state=seed, early_stopping=True, validation_fraction=0.12, **p
        )
    raise KeyError(name)


def build_sklearn_model(
    name: str, seed: int = 0, params: dict[str, Any] | None = None, **_: Any
) -> Pipeline:
    """Return a Pipeline(StandardScaler, classifier) with fit/predict/predict_proba."""
    clf = _classifier(name, seed, params)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_model(name: str, seed: int = 0, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    if name in DEFAULT_PARAMS:
        return build_sklearn_model(name, seed=seed, params=params, **kwargs)
    from .torch_models import TorchClassifier

    if name in {"mlp", "cnn"}:
        return TorchClassifier(name=name, seed=seed, params=params, **kwargs)
    raise KeyError(f"unknown model {name!r}")


def model_size_bytes(model: Any) -> int:
    """Serialised size without writing a file (workaround for pickle protocol)."""
    import io
    import pickle

    buf = io.BytesIO()
    try:
        pickle.dump(model, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return int(buf.tell())
    except Exception:  # pragma: no cover - unpicklable estimators
        return 0


def param_count(model: Any) -> int:
    """Number of fitted scalar parameters (trees count nodes, not leaves only)."""
    try:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    except KeyError:  # pragma: no cover
        clf = model
    if hasattr(clf, "coef_"):
        return int(np.sum([c.size for c in np.atleast_1d(clf.coef_)])) + int(
            np.sum([b.size for b in np.atleast_1d(clf.intercept_)])
        )
    if hasattr(clf, "support_vectors_"):
        return int(clf.support_vectors_.size + clf.dual_coef_.size)
    if hasattr(clf, "_fit_X"):
        return int(clf._fit_X.size)
    if hasattr(clf, "estimators_"):
        total = 0
        for est in np.ravel(clf.estimators_):
            tree = est.tree_
            total += int(tree.node_count)
        return total
    if hasattr(clf, "_predictors"):  # HistGradientBoosting: list per class of predictors
        total = 0
        for group in clf._predictors:
            for tree in group if isinstance(group, (list, tuple)) else [group]:
                nodes = getattr(tree, "nodes", None)
                total += int(getattr(nodes, "size", 1))
        return total
    return 0
