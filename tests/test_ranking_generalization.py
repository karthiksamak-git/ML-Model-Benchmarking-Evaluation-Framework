from __future__ import annotations

import numpy as np

from mlbe.benchmark.generalization import (
    learning_curve,
    robustness_analysis,
    seed_stability,
)
from mlbe.data.registry import load_dataset
from mlbe.stats.ranking import pareto_front, score_models


def _rows():
    return [
        {"model": "fast", "dataset": "d", "accuracy": 0.90, "gap": 0.02, "latency_p50_ms": 0.1},
        {"model": "slow", "dataset": "d", "accuracy": 0.92, "gap": 0.08, "latency_p50_ms": 10.0},
        {"model": "weak", "dataset": "d", "accuracy": 0.70, "gap": 0.01, "latency_p50_ms": 0.05},
    ]


def test_ranking_is_deterministic_and_weight_sensitive():
    a = score_models(_rows())
    b = score_models(list(reversed(_rows())))
    assert [r["model"] for r in a] == [r["model"] for r in b]
    assert all("score" in r and "rank" in r for r in a)
    accuracy_only = score_models(_rows(), {"accuracy": 1.0, "generalization": 0.0, "performance": 0.0})
    assert accuracy_only[0]["model"] == "slow"
    speed_only = score_models(_rows(), {"accuracy": 0.0, "generalization": 0.0, "performance": 1.0})
    assert speed_only[0]["model"] == "weak" or speed_only[0]["model"] == "fast"


def test_pareto_front_excludes_dominated_points():
    points = [
        {"id": "a", "accuracy": 0.9, "latency_p50_ms": 1.0, "model_size_mb": 1.0},
        {"id": "b", "accuracy": 0.9, "latency_p50_ms": 2.0, "model_size_mb": 1.0},
        {"id": "c", "accuracy": 0.8, "latency_p50_ms": 0.5, "model_size_mb": 0.5},
    ]
    front = pareto_front(points)
    assert set(front) == {"a", "c"}


def test_seed_stability():
    out = seed_stability([0.90, 0.92, 0.88])
    assert out["mean"] == 0.90
    assert out["std"] > 0
    assert out["range"] > 0


def test_learning_curve_and_robustness_are_non_empty():
    ds = load_dataset("wine", {"kind": "sklearn", "loader": "load_wine"})
    rows = learning_curve(
        ds.X, ds.y, "logreg", seed=0, fractions=(0.4, 1.0), cv_folds=2, dataset_name="wine"
    )
    assert rows and all(r["n_train"] > 0 for r in rows)
    assert {r["fraction"] for r in rows} == {0.4, 1.0}
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)

    split = int(len(ds.y) * 0.7)
    rob = robustness_analysis(
        ds.X[:split],
        ds.y[:split],
        ds.X[split:],
        ds.y[split:],
        "logreg",
        seed=0,
        noise_levels=(0.0, 0.5),
        shift_levels=(0.0, 1.0),
        dataset_name="wine",
    )
    assert len(rob) == 4
    assert {r["kind"] for r in rob} == {"gaussian_noise", "covariate_shift"}
    assert all(r["drop"] >= 0 for r in rob)
    assert np.isfinite([r["accuracy"] for r in rob]).all()
    assert any(r["level"] == 0.0 for r in rob)
