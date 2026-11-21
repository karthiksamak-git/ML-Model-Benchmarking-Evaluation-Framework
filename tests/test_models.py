from __future__ import annotations

import numpy as np
import torch

from mlbe.data.registry import load_dataset
from mlbe.models.registry import build_model, models_for
from mlbe.seed import seed_everything


def _data():
    ds = load_dataset("digits", {"kind": "sklearn", "loader": "load_digits"})
    return ds


def test_models_for_respects_image_datasets():
    from mlbe.config import benchmark_config

    cfg = benchmark_config()
    tabular = load_dataset("wine", cfg.datasets["wine"], max_samples=50)
    image = load_dataset("digits_image", cfg.datasets["digits_image"], max_samples=50)
    assert "cnn" not in models_for(tabular, cfg)
    assert models_for(image, cfg) == ["cnn"]


def test_sklearn_models_have_probabilities_summing_to_one():
    ds = _data()
    X, y = ds.X[:400], ds.y[:400]
    for name in ("logreg", "knn", "rf", "hgb", "svm"):
        model = build_model(name, seed=2, n_classes=10, input_dim=64)
        model.fit(X[:300], y[:300])
        proba = model.predict_proba(X[300:])
        assert np.isfinite(proba).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), name
        assert model.predict(X[300:]).shape == (100,)


def test_torch_mlp_is_deterministic_on_cpu():
    ds = _data()
    X, y = ds.X[:250], ds.y[:250]

    def run() -> np.ndarray:
        seed_everything(0)
        model = build_model("mlp", seed=0, n_classes=10, input_dim=64, params={"epochs": 6})
        model.fit(X[:200], y[:200])
        return model.predict_proba(X[200:])

    a, b = run(), run()
    assert np.allclose(a, b, atol=0.0), "CPU training with a fixed seed must be bit-identical"
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-9)


def test_torch_mlp_early_stopping_records_history():
    ds = _data()
    model = build_model(
        "mlp", seed=0, n_classes=10, input_dim=64, params={"epochs": 20, "patience": 2, "hidden_dim": 16}
    )
    model.fit(ds.X[:200], ds.y[:200])
    assert 1 <= len(model.history_) <= 20
    assert model.best_epoch_ < len(model.history_)
    assert model.param_count() > 0


def test_cnn_accepts_flat_pixels_and_is_deterministic():
    ds = _data()
    X, y = ds.X[:200], ds.y[:200]

    def run():
        seed_everything(1)
        model = build_model(
            "cnn", seed=1, n_classes=10, input_dim=64, image_shape=(8, 8), params={"epochs": 3}
        )
        model.fit(X[:160], y[:160])
        return model.predict_proba(X[160:])

    a, b = run(), run()
    assert a.shape == (40, 10)
    assert np.allclose(a, b, atol=0.0)


def test_model_size_and_param_count_are_positive():
    from mlbe.models.sklearn_models import model_size_bytes, param_count

    ds = _data()
    model = build_model("logreg", seed=0, n_classes=10, input_dim=64)
    model.fit(ds.X[:300], ds.y[:300])
    assert param_count(model) > 0
    assert model_size_bytes(model) > 0
    assert torch.__version__
