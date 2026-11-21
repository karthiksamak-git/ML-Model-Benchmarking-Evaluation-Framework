from __future__ import annotations

import pytest

from mlbe.config import Config, benchmark_config, load_configs


def test_loads_yaml_with_attribute_access():
    cfg = benchmark_config()
    assert cfg.seed == 42
    assert cfg.cv.n_splits == 5
    assert "breast_cancer" in cfg.datasets
    assert cfg.datasets["wine"].loader == "load_wine"


def test_cli_overrides_are_typed():
    cfg = benchmark_config(overrides=["cv.n_splits=3", "seed=7", "performance.latency_repeats=2"])
    assert cfg.cv.n_splits == 3
    assert cfg.seed == 7
    assert isinstance(cfg.performance.latency_repeats, int)
    assert cfg.cv.n_repeats == 3  # untouched


def test_override_requires_assignment():
    with pytest.raises(ValueError):
        benchmark_config(overrides=["cv.n_splits"])


def test_unknown_key_raises_attribute_error():
    cfg = benchmark_config()
    with pytest.raises(AttributeError):
        _ = cfg.not_a_key


def test_three_config_files_load():
    cfgs = load_configs()
    assert set(cfgs) == {"benchmark", "hpo", "rubrics"}
    assert cfgs["rubrics"].dataset_quality.weights.label_noise == 0.15
    assert cfgs["hpo"].spaces.hgb.learning_rate.type == "float"
    assert cfgs["benchmark"].quick.n_splits == 3


def test_config_to_dict_is_a_copy():
    cfg = benchmark_config()
    data = cfg.to_dict()
    data["seed"] = 999
    assert cfg.seed == 42
    assert Config.from_dict({"a": {"b": 1}}).a.b == 1
