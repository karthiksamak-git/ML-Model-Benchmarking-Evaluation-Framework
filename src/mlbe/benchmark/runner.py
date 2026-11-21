"""Benchmark runner: models x datasets x folds x repeats, logged to MLflow.

Every model sees the *same* splits for a dataset (split indices are logged as an
artifact with a checksum), and each fold is a nested MLflow run whose parent is
one (dataset, model) pair.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from tqdm import tqdm

from ..data.registry import Dataset, load_all
from ..data.splits import Split, make_splits, save_splits, split_checksum, validate_splits
from ..env_info import env_info
from ..metrics import compute_metrics
from ..models.registry import MODEL_LABELS, build_model, models_for
from ..seed import seed_everything
from ..tracking import mlflow_utils as mu
from ..tracking.queries import log_npz_artifact
from .performance import profile_model


@dataclass
class BenchmarkSummary:
    parents: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"parents": self.parents, "skipped": self.skipped, "elapsed_s": self.elapsed_s}


def _model_kwargs(dataset: Dataset, n_classes: int) -> dict[str, Any]:
    kw: dict[str, Any] = {"input_dim": dataset.n_features, "n_classes": n_classes}
    if dataset.is_image:
        kw["image_shape"] = dataset.image_shape
    return kw


def _fit_predict(model_name: str, X_tr, y_tr, X_te, seed: int, params, dataset: Dataset):
    model = build_model(
        model_name,
        seed=seed,
        params=params,
        input_dim=dataset.n_features,
        n_classes=len(np.unique(y_tr)),
        image_shape=dataset.image_shape if dataset.is_image else None,
    )
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - start
    proba = model.predict_proba(X_te)
    pred = model.predict(X_te)
    return model, train_time, proba, pred


def run_benchmark(
    cfg: Any,
    quick: bool = False,
    max_datasets: int | None = None,
    only_models: list[str] | None = None,
    max_samples: int | None = None,
    tracking: bool = True,
) -> BenchmarkSummary:
    seed = int(cfg.seed)
    seed_everything(seed)
    n_splits = int(cfg.cv.n_splits)
    n_repeats = int(cfg.cv.n_repeats)
    if quick:
        n_splits = int(cfg.quick.n_splits)
        n_repeats = int(cfg.quick.n_repeats)
        max_samples = max_samples or int(cfg.quick.max_samples)

    if tracking:
        mu.init_tracking(cfg)
    env = env_info()
    datasets = load_all(cfg, max_samples=max_samples, seed=seed)
    names = list(datasets)[:max_datasets] if max_datasets else list(datasets)

    summary = BenchmarkSummary()
    started = time.time()
    per_dataset_splits: dict[str, list[Split]] = {}
    for name in names:
        ds = datasets[name]
        splits = make_splits(ds.y, n_splits=n_splits, n_repeats=n_repeats, seed=seed, dataset=name)
        validate_splits(splits, ds.y)
        per_dataset_splits[name] = splits

    if tracking:
        with mu.run(
            f"benchmark-suite-{'quick' if quick else 'full'}",
            tags={"kind": "benchmark_suite", "quick": quick, "cv": f"{n_splits}x{n_repeats}"},
        ) as suite:
            mu.log_env_tags(env)
            mu.log_params_safe(
                {
                    "seed": seed,
                    "n_splits": n_splits,
                    "n_repeats": n_repeats,
                    "datasets": names,
                    "quick": quick,
                }
            )
            suite_id = suite.info.run_id
            _run_all(
                cfg, datasets, names, per_dataset_splits, only_models, suite_id, summary
            )
            mu.log_metrics({"suite_elapsed_s": time.time() - started})
    else:  # pragma: no cover - used by unit tests
        _run_all(cfg, datasets, names, per_dataset_splits, only_models, None, summary)

    summary.elapsed_s = time.time() - started
    return summary


def _run_all(cfg, datasets, names, per_dataset_splits, only_models, parent_run_id, summary):
    seed = int(cfg.seed)
    for name in tqdm(names, desc="datasets", disable=None):
        ds = datasets[name]
        splits = per_dataset_splits[name]
        model_names = models_for(ds, cfg)
        if only_models:
            model_names = [m for m in model_names if m in only_models]
        for model_name in model_names:
            summary.parents.append(
                _run_model(cfg, ds, splits, model_name, seed, parent_run_id)
            )


def _run_model(
    cfg: Any,
    ds: Dataset,
    splits: list[Split],
    model_name: str,
    seed: int,
    parent_run_id: str | None,
) -> dict[str, Any]:
    labels = np.unique(ds.y)
    fold_metrics: list[dict[str, float]] = []
    fold_train: list[dict[str, float]] = []
    fold_perf: list[dict[str, float]] = []
    oof_payload: dict[str, np.ndarray] | None = None
    n_folds_per_repeat = (max(s.fold for s in splits) + 1) if splits else 1
    n_repeats = len(splits) // n_folds_per_repeat if n_folds_per_repeat else 1
    params = {
        "model": model_name,
        "dataset": ds.name,
        "n_classes": int(len(labels)),
        "n_samples": ds.n_samples,
        "n_splits": n_folds_per_repeat,
        "n_repeats": n_repeats,
    }

    with mu.run(
        f"{ds.name}__{model_name}",
        tags={
            "kind": "benchmark_parent",
            "dataset": ds.name,
            "model": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "split_checksum": split_checksum(splits),
            "suite_run_id": parent_run_id or "standalone",
        },
        nested=parent_run_id is not None,
    ) as parent:
        mu.log_env_tags(env_info())
        mu.log_params_safe({**params, "n_splits": len(splits)})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_splits(splits, Path(tmp) / f"splits_{ds.name}.json")
            mlflow.log_artifact(str(path))

        for split in tqdm(splits, desc=f"{ds.name}/{model_name}", leave=False, disable=None):
            fold_seed = seed + 1000 * split.repeat + split.fold
            seed_everything(fold_seed)
            X_train, X_test = ds.X[split.train_idx], ds.X[split.test_idx]
            y_train, y_test = ds.y[split.train_idx], ds.y[split.test_idx]
            model, train_time, proba, pred = _fit_predict(
                model_name, X_train, y_train, X_test, fold_seed, None, ds
            )
            test_metrics = compute_metrics(y_test, pred, proba, labels)
            train_pred = model.predict(X_train)
            train_proba = model.predict_proba(X_train)
            train_metrics = compute_metrics(y_train, train_pred, train_proba, labels)
            perf = profile_model(model, X_test, cfg.performance, train_time)

            with mu.run(
                f"r{split.repeat}-f{split.fold}",
                tags={
                    "kind": "benchmark_fold",
                    "dataset": ds.name,
                    "model": model_name,
                    "split_id": split.split_id,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "seed": fold_seed,
                },
                nested=True,
            ):
                mu.log_params_safe(
                    {
                        "split_id": split.split_id,
                        "seed": fold_seed,
                        "n_train": len(split.train_idx),
                        "n_test": len(split.test_idx),
                        "model": model_name,
                        "dataset": ds.name,
                        "n_samples": ds.n_samples,
                        "n_splits": n_folds_per_repeat,
                        "n_repeats": n_repeats,
                    }
                )
                mu.log_metrics({**test_metrics, **{f"train_{k}": v for k, v in train_metrics.items()}})
                mu.log_metrics({f"perf_{k}": v for k, v in perf.items()})
                mu.log_metrics(
                    {f"gap_{k}": train_metrics[k] - test_metrics[k] for k in ("accuracy", "f1_macro")}
                )
                mlflow.set_tag("run_status", "completed")

            fold_metrics.append(test_metrics)
            fold_train.append(train_metrics)
            fold_perf.append(perf)
            if oof_payload is None:  # first CV split, used for paired tests
                oof_payload = {
                    "y_true": np.asarray(y_test),
                    "y_pred": np.asarray(pred),
                    "proba": np.asarray(proba),
                    "labels": np.asarray(labels),
                    "test_idx": np.asarray(split.test_idx),
                    "split_id": np.asarray([split.split_id]),
                }

        aggregate: dict[str, float] = {}
        for key in fold_metrics[0]:
            values = np.asarray([m[key] for m in fold_metrics], dtype=float)
            aggregate[f"{key}_mean"] = float(np.nanmean(values))
            aggregate[f"{key}_std"] = float(np.nanstd(values))
        for key in fold_perf[0]:
            values = np.asarray([m[key] for m in fold_perf], dtype=float)
            aggregate[f"perf_{key}_mean"] = float(np.nanmean(values))
            aggregate[f"perf_{key}_std"] = float(np.nanstd(values))
        for key in ("accuracy", "f1_macro"):
            aggregate[f"gap_{key}_mean"] = float(
                np.mean([tr[key] - te[key] for tr, te in zip(fold_train, fold_metrics, strict=False)])
            )
        mu.log_metrics(aggregate)
        mlflow.set_tags(
            {
                "n_folds": len(fold_metrics),
                "model_label": MODEL_LABELS.get(model_name, model_name),
            }
        )
        if oof_payload is not None:
            log_npz_artifact(oof_payload, "oof.npz")

        summary = {
            "run_id": parent.info.run_id,
            "dataset": ds.name,
            "model": model_name,
            "metrics": aggregate,
            "n_folds": len(fold_metrics),
        }
    return summary
