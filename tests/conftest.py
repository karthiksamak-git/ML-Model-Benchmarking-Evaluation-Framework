from __future__ import annotations

import numpy as np
import pytest

from mlbe.config import Config, load_configs


@pytest.fixture(scope="session")
def cfgs():
    return load_configs()


@pytest.fixture
def quick_cfg(tmp_path, monkeypatch):
    """Benchmark config with a cheap CV scheme and an isolated MLflow backend."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    from mlbe.config import PROJECT_ROOT

    cfg = Config.load(PROJECT_ROOT / "configs" / "benchmark.yaml").with_overrides(
        ["cv.n_splits=2", "cv.n_repeats=1", "performance.latency_repeats=3"]
    )
    cfg._data["paths"]["artifact_root"] = str(tmp_path / "artifacts")
    cfg._data["paths"]["experiment"] = "test-experiment"
    return cfg


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y
