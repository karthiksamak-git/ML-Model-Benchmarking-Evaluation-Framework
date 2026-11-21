"""Model registry: name -> factory, labels, applicability and search spaces."""

from __future__ import annotations

from typing import Any

from .sklearn_models import DEFAULT_PARAMS as SKLEARN_DEFAULTS
from .sklearn_models import MODEL_LABELS, build_model
from .torch_models import DEFAULT_PARAMS as TORCH_DEFAULTS

ALL_MODELS = ["logreg", "svm", "knn", "rf", "hgb", "mlp", "cnn"]
TABULAR_MODELS = [m for m in ALL_MODELS if m != "cnn"]

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {**SKLEARN_DEFAULTS, **TORCH_DEFAULTS}

__all__ = [
    "ALL_MODELS",
    "DEFAULT_PARAMS",
    "MODEL_LABELS",
    "TABULAR_MODELS",
    "build_model",
    "models_for",
]


def models_for(dataset: Any, cfg: Any) -> list[str]:
    """Models applicable to a dataset (CNN only for image data)."""
    image = bool(getattr(dataset, "is_image", False))
    names = list(cfg.models.image if image else cfg.models.tabular)
    return [n for n in names if image or n != "cnn"]


def is_torch(model: str) -> bool:
    return model in {"mlp", "cnn"}


def search_space(model: str, hpo_cfg: Any) -> dict[str, Any] | None:
    spaces = hpo_cfg.spaces
    if model not in spaces._data:
        return None
    return spaces[model].to_dict()
