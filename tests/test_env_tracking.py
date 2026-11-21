from __future__ import annotations

import numpy as np
import pytest

from mlbe.env_info import env_info, git_hash
from mlbe.tracking import mlflow_utils as mu
from mlbe.tracking.queries import experiment_id


def test_env_info_has_all_fields():
    info = env_info()
    for key in (
        "python",
        "platform",
        "machine",
        "processor",
        "cpu_count_logical",
        "ram_gb",
        "threads",
        "accelerator",
        "versions",
        "git",
    ):
        assert key in info
    assert info["ram_gb"] > 0
    assert "scikit-learn" in info["versions"]
    assert "torch" in info["versions"]
    assert isinstance(info["threads"]["torch"], int)
    assert info["git"]


def test_git_hash_is_string():
    assert isinstance(git_hash(), str)


def test_logged_run_contains_params_tags_metrics(quick_cfg):
    from mlflow.tracking import MlflowClient

    name = mu.init_tracking(quick_cfg)
    with mu.run("unit-test-run", tags={"kind": "unit_test"}) as active:
        mu.log_env_tags(env_info(include_git=False))
        mu.log_params_safe({"alpha": 0.1, "nested": {"x": 1}})
        mu.log_metrics({"accuracy": 0.9, "nan_metric": np.nan, "inf": np.inf})
        mu.log_json_artifact({"hello": "world"}, "payload.json")
        run_id = active.info.run_id

    client = MlflowClient()
    exp_id = experiment_id(client, name)
    assert exp_id
    run = client.get_run(run_id)
    assert run.data.params["alpha"] == "0.1"
    assert run.data.params["nested.x"] == "1"
    assert run.data.metrics["accuracy"] == pytest.approx(0.9)
    assert "nan_metric" not in run.data.metrics  # non-finite values are never logged
    assert run.data.tags["kind"] == "unit_test"
    assert run.data.tags["env.accelerator"]
    assert any(a.path == "payload.json" for a in client.list_artifacts(run_id))
    assert run.data.tags.get("mlflow.runName") == "unit-test-run"
