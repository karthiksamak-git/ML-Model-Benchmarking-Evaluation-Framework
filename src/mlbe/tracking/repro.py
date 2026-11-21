"""Reproduce a logged run from its params/seed/split ID and compare metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

from ..data.registry import load_dataset
from ..data.splits import make_splits
from ..env_info import git_hash
from ..metrics import compute_metrics
from ..models.registry import build_model
from ..seed import seed_everything


def reproduce_run(
    run_id: str,
    cfg: Any,
    tolerance: float | None = None,
    seed_offset: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-run one fold run and compare against its logged metrics.

    ``seed_offset`` exists so tests can demonstrate the failure path. The
    comparison covers accuracy, balanced accuracy, macro-F1, log-loss, Brier and
    ECE; the maximum absolute difference is recorded on the original run.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    tags = run.data.tags
    model_name = params.get("model") or tags.get("model")
    dataset_name = params.get("dataset") or tags.get("dataset")
    split_id = params.get("split_id") or tags.get("split_id")
    if not (model_name and dataset_name and split_id):
        raise ValueError(f"run {run_id} is not a fold run (missing model/dataset/split_id)")
    base_seed = int(params.get("seed", cfg.seed)) + int(seed_offset)

    # the run records how the dataset was materialised (quick mode subsamples) and
    # which CV scheme produced its split IDs, so reproduction does not depend on
    # the config currently on disk.
    full = load_dataset(dataset_name, cfg.datasets[dataset_name], seed=int(cfg.seed))
    n_samples = int(float(params.get("n_samples", full.n_samples)))
    dataset = (
        load_dataset(dataset_name, cfg.datasets[dataset_name], seed=int(cfg.seed), max_samples=n_samples)
        if n_samples < full.n_samples
        else full
    )
    n_splits = int(float(params.get("n_splits", cfg.cv.n_splits)))
    n_repeats = int(float(params.get("n_repeats", cfg.cv.n_repeats)))
    splits = make_splits(
        dataset.y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=int(cfg.seed),
        dataset=dataset_name,
    )
    match = next((s for s in splits if s.split_id == split_id), None)
    if match is None:
        raise ValueError(f"split {split_id!r} not found in current config")
    seed_everything(base_seed)
    X_tr, y_tr = dataset.X[match.train_idx], dataset.y[match.train_idx]
    X_te, y_te = dataset.X[match.test_idx], dataset.y[match.test_idx]
    labels = np.unique(dataset.y)
    model = build_model(
        model_name,
        seed=base_seed,
        n_classes=len(labels),
        input_dim=dataset.n_features,
        image_shape=dataset.image_shape if dataset.is_image else None,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred = model.predict(X_te)
    fresh = compute_metrics(y_te, pred, proba, labels)

    logged = {k: float(v) for k, v in run.data.metrics.items() if k in fresh}
    diffs = {k: abs(float(fresh[k]) - logged[k]) for k in logged}
    max_diff = max(diffs.values()) if diffs else float("inf")
    tol = float(
        tolerance
        if tolerance is not None
        else (1e-5 if model_name in {"mlp", "cnn"} else 1e-9)
    )
    status = "pass" if max_diff <= tol else "fail"
    payload = {
        "source_run_id": run_id,
        "model": model_name,
        "dataset": dataset_name,
        "split_id": split_id,
        "seed_used": base_seed,
        "seed_offset": seed_offset,
        "tolerance": tol,
        "max_abs_diff": max_diff,
        "metrics_compared": {
            k: {"logged": logged[k], "rerun": float(fresh[k]), "abs_diff": diffs[k]} for k in logged
        },
        "status": status,
        "git_hash": git_hash(),
    }
    if not dry_run:
        client.set_tag(run_id, "repro_status", status)
        client.set_tag(run_id, "repro_git_hash", payload["git_hash"])
        client.log_metric(run_id, "repro_max_abs_diff", max_diff)
    return payload


def sample_run_id(cfg: Any, experiment: str | None = None) -> str:
    """Pick one completed benchmark fold run (highest accuracy) for repro."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment or str(cfg.paths.experiment))
    if exp is None:
        raise RuntimeError("experiment not found; run `mlbe benchmark` first")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.kind = 'benchmark_fold'",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("no benchmark fold runs found")
    return runs[0].info.run_id
