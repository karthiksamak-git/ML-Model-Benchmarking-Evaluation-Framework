"""Read benchmark/HPO results back from MLflow (single source of truth)."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

from mlflow.entities import Run  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


def experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise RuntimeError(f"experiment {name!r} does not exist; run `mlbe benchmark` first")
    return exp.experiment_id


def _runs(client: MlflowClient, exp_id: str, filter_string: str = "") -> list[Run]:
    return client.search_runs(
        [exp_id], filter_string=filter_string or None, order_by=["attributes.start_time ASC"], max_results=20000
    )


def benchmark_parents(client: MlflowClient, exp_id: str) -> list[dict[str, Any]]:
    """All (dataset, model) parent runs, keeping the most recent one per pair.

    Re-running the benchmark creates a new suite of runs; downstream phases must
    use one consistent suite (equal fold counts) rather than a mix.
    """
    runs = _runs(client, exp_id, "tags.kind = 'benchmark_parent'")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in runs:
        key = (r.data.tags.get("dataset", ""), r.data.tags.get("model", ""))
        latest[key] = {
            "run_id": r.info.run_id,
            "dataset": r.data.tags.get("dataset"),
            "model": r.data.tags.get("model"),
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
            "tags": dict(r.data.tags),
            "start_time": r.info.start_time,
        }
    return sorted(latest.values(), key=lambda r: (r["dataset"] or "", r["model"] or ""))


def fold_runs(client: MlflowClient, parent_id: str) -> list[dict[str, Any]]:
    runs = _runs(client, client.get_run(parent_id).info.experiment_id, f"tags.mlflow.parentRunId = '{parent_id}'")
    out = []
    for r in runs:
        out.append(
            {
                "run_id": r.info.run_id,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
                "tags": dict(r.data.tags),
            }
        )
    out.sort(key=lambda x: (int(x["tags"].get("repeat", 0)), int(x["tags"].get("fold", 0))))
    return out


def load_oof(client: MlflowClient, run_id: str) -> dict[str, np.ndarray] | None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, "oof.npz", tmp)
            with np.load(local) as data:
                return {k: data[k] for k in data.files}
    except Exception:
        return None


def load_json_artifact(client: MlflowClient, run_id: str, path: str) -> Any:
    import json

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, path, tmp)
            return json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception:
        return None


def log_npz_artifact(arrays: dict[str, np.ndarray], filename: str) -> None:
    import mlflow

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_bytes(buf.getvalue())
        mlflow.log_artifact(str(path))


def hpo_studies(client: MlflowClient, exp_id: str) -> list[dict[str, Any]]:
    """HPO studies, keeping only the most recent study per (model, dataset)."""
    runs = _runs(client, exp_id, "tags.kind = 'hpo_study'")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in runs:
        key = (r.data.tags.get("model", ""), r.data.tags.get("dataset", ""))
        latest[key] = {
            "run_id": r.info.run_id,
            "model": r.data.tags.get("model"),
            "dataset": r.data.tags.get("dataset"),
            "metrics": dict(r.data.metrics),
            "params": dict(r.data.params),
            "tags": dict(r.data.tags),
            "start_time": r.info.start_time,
        }
    return sorted(latest.values(), key=lambda r: (r["model"] or "", r["dataset"] or ""))


def hpo_trials(client: MlflowClient, study_run_id: str) -> list[dict[str, Any]]:
    exp_id = client.get_run(study_run_id).info.experiment_id
    runs = _runs(client, exp_id, f"tags.mlflow.parentRunId = '{study_run_id}'")
    out = []
    for r in runs:
        out.append(
            {
                "run_id": r.info.run_id,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
                "tags": dict(r.data.tags),
            }
        )
    out.sort(key=lambda x: int(x["tags"].get("trial_number", 0)))
    return out
