from __future__ import annotations

import numpy as np

from mlbe.benchmark.runner import run_benchmark
from mlbe.config import PROJECT_ROOT, Config
from mlbe.tracking.queries import benchmark_parents, experiment_id, fold_runs
from mlbe.tracking.repro import reproduce_run


def _cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = Config.load(PROJECT_ROOT / "configs" / "benchmark.yaml").with_overrides(
        [
            "cv.n_splits=2",
            "cv.n_repeats=1",
            "quick.n_splits=2",
            "quick.n_repeats=1",
            "quick.max_samples=180",
            "performance.latency_repeats=3",
            "performance.batch_sizes=[1, 32]",
        ]
    )
    cfg._data["paths"]["artifact_root"] = str(tmp_path / "artifacts")
    cfg._data["paths"]["experiment"] = "runner-test"
    return cfg


def test_quick_benchmark_tree_metrics_and_shared_splits(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    summary = run_benchmark(
        cfg, quick=True, max_datasets=1, only_models=["logreg", "mlp"], max_samples=180
    )
    assert len(summary.parents) == 2
    assert summary.elapsed_s > 0

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = experiment_id(client, "runner-test")
    suite = client.search_runs([exp], filter_string="tags.kind = 'benchmark_suite'")
    assert len(suite) == 1
    parents = benchmark_parents(client, exp)
    assert {(p["dataset"], p["model"]) for p in parents} == {("breast_cancer", "logreg"), ("breast_cancer", "mlp")}
    assert len({p["tags"]["split_checksum"] for p in parents}) == 1

    fold_ids = {}
    for p in parents:
        folds = fold_runs(client, p["run_id"])
        assert len(folds) == 2
        for f in folds:
            for metric, value in f["metrics"].items():
                assert np.isfinite(value), (metric, value)
            assert 0.0 <= f["metrics"]["accuracy"] <= 1.0
            assert 0.0 <= f["metrics"]["ece"] <= 1.0
            assert f["metrics"]["perf_param_count"] > 0
            assert f["metrics"]["perf_latency_p50_ms"] > 0
        fold_ids[p["model"]] = sorted(f["tags"]["split_id"] for f in folds)
        for key in ("accuracy_mean", "balanced_accuracy_mean", "f1_macro_mean", "roc_auc_ovr_mean",
                    "log_loss_mean", "brier_mean", "ece_mean", "gap_accuracy_mean"):
            assert np.isfinite(p["metrics"][key])

    # identical fold indices for every model: same split IDs and checksums
    assert fold_ids["logreg"] == fold_ids["mlp"]
    from mlbe.tracking.queries import load_oof

    left = load_oof(client, parents[0]["run_id"])
    right = load_oof(client, parents[1]["run_id"])
    assert left is not None and right is not None
    assert np.array_equal(left["test_idx"], right["test_idx"])
    assert np.array_equal(left["y_true"], right["y_true"])


def test_repro_passes_and_fails_with_changed_seed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    run_benchmark(
        cfg, quick=True, max_datasets=1, only_models=["logreg", "mlp"], max_samples=180
    )
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = experiment_id(client, "runner-test")
    folds = client.search_runs(
        [exp], filter_string="tags.kind = 'benchmark_fold'", order_by=["metrics.log_loss DESC"], max_results=1
    )
    assert folds
    best = folds[0].info.run_id
    passed = reproduce_run(best, cfg)
    assert passed["status"] == "pass", passed
    assert passed["max_abs_diff"] <= passed["tolerance"]
    failed = reproduce_run(best, cfg, seed_offset=1, dry_run=True)
    assert failed["status"] == "fail"
    assert failed["max_abs_diff"] > failed["tolerance"]
    # the result is written back onto the original run
    run = client.get_run(best)
    assert run.data.tags["repro_status"] == "pass"
    assert run.data.metrics["repro_max_abs_diff"] >= 0


def test_performance_measurements_are_finite(tmp_path):
    from mlbe.benchmark.performance import profile_model
    from mlbe.data.registry import load_dataset
    from mlbe.models.registry import build_model

    ds = load_dataset("wine", {"kind": "sklearn", "loader": "load_wine"})
    model = build_model("logreg", seed=0, n_classes=3, input_dim=ds.n_features)
    model.fit(ds.X[:120], ds.y[:120])
    cfg = Config.from_dict({"latency_repeats": 5, "latency_warmup": 1, "batch_sizes": [1, 16], "memory_sample": 32})
    out = profile_model(model, ds.X[120:], cfg, train_time_s=0.01)
    for key, value in out.items():
        assert np.isfinite(value), key
    assert out["param_count"] > 0
    assert out["model_size_mb"] > 0
    assert out["latency_p95_ms"] >= out["latency_p50_ms"] > 0
    assert out["throughput_bs16_sps"] > 0
