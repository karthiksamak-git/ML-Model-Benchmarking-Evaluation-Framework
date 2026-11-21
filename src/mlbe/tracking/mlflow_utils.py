"""MLflow helpers: local SQLite backend, environment tags, safe logging."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from ..config import PROJECT_ROOT


def tracking_uri(cfg: Any) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or str(cfg.paths.tracking_uri)


def init_tracking(cfg: Any, experiment: str | None = None) -> str:
    """Point MLflow at the local backend and ensure the experiment exists."""
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    name = experiment or str(cfg.paths.experiment)
    client = MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None:
        artifact_root = Path(cfg.paths.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = PROJECT_ROOT / artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        client.create_experiment(name=name, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(name)
    return name


def client() -> MlflowClient:
    return MlflowClient()


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(name, value))
        else:
            out[name] = value
    return out


def _tag_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), default=str)[:5000]
    return str(value)[:5000]


def log_env_tags(env: dict[str, Any]) -> None:
    for key, value in flatten("env", env).items():
        mlflow.set_tag(key, _tag_value(value))


def log_params_safe(params: dict[str, Any]) -> None:
    clean = {}
    for key, value in flatten("", params).items():
        if isinstance(value, bool):
            clean[key] = str(value).lower()
        elif isinstance(value, (int, float, str)):
            clean[key] = value
        elif value is None:
            clean[key] = "null"
        elif isinstance(value, (list, tuple)):
            clean[key] = json.dumps(list(value))
        else:
            clean[key] = str(value)
    if clean:
        mlflow.log_params(clean)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in flatten("", metrics).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        val = float(value)
        if not math.isfinite(val):
            continue
        clean[key] = val
    if clean:
        mlflow.log_metrics(clean, step=step)
    return clean


def log_json_artifact(obj: Any, filename: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))
    return filename


@contextmanager
def run(name: str, tags: dict[str, Any] | None = None, nested: bool = False) -> Iterator[Any]:
    with mlflow.start_run(run_name=name, nested=nested, tags={k: _tag_value(v) for k, v in (tags or {}).items()}) as active:
        yield active


def log_model_with_signature(model: Any, X_sample: Any, artifact_path: str = "model") -> None:
    """Log a model with an inferred signature where MLflow supports it."""
    try:
        from mlflow.models import infer_signature

        prediction = model.predict(X_sample)
        signature = infer_signature(X_sample, prediction)
        if hasattr(model, "named_steps") or hasattr(model, "predict_proba"):
            mlflow.sklearn.log_model(model, artifact_path=artifact_path, signature=signature)
    except Exception:  # pragma: no cover - optional, model flavours differ
        return
