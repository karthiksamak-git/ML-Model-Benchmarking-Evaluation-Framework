"""Typed-ish configuration access for the YAML files in ``configs/``.

The config is a thin recursive wrapper around plain YAML data. It deliberately
accepts CLI overrides of the form ``section.key=value`` (values are parsed as
YAML scalars) so experiments can be reproduced from a shell command.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class Config:
    """Recursive read-only mapping with attribute access."""

    __slots__ = ("_data", "path")

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "path", path)

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self._data[name])
        except KeyError:
            raise AttributeError(f"config has no key {name!r}") from None

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return _wrap(self._data.get(key, default))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, other: dict[str, Any]) -> Config:
        return Config(deep_merge(self.to_dict(), other), self.path)

    def with_overrides(self, overrides: list[str] | None) -> Config:
        data = self.to_dict()
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must look like key.subkey=value, got {item!r}")
            dotted, raw = item.split("=", 1)
            node = data
            parts = dotted.strip().split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ValueError(f"cannot override through non-mapping key {part!r}")
            node[parts[-1]] = yaml.safe_load(raw)
        return Config(data, self.path)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{p} must contain a mapping at the top level")
        return cls(data, p).with_overrides(overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(data)


def deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_configs(
    config_dir: str | Path | None = None, overrides: list[str] | None = None
) -> dict[str, Config]:
    """Load benchmark.yaml, hpo.yaml and rubrics.yaml from one directory."""
    root = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    configs = {}
    for name in ("benchmark", "hpo", "rubrics"):
        path = root / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(path)
        configs[name] = Config.load(path, overrides)
    return configs


def benchmark_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "benchmark.yaml"
    return Config.load(p, overrides)


def hpo_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "hpo.yaml"
    return Config.load(p)


def rubric_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "rubrics.yaml"
    return Config.load(p)


def quick_overrides(quick: bool) -> list[str]:
    """Overrides that make the CV scheme cheap for CI/tests."""
    if not quick:
        return []
    cfg = benchmark_config()
    q = cfg.quick
    return [
        f"cv.n_splits={q.n_splits}",
        f"cv.n_repeats={q.n_repeats}",
        f"performance.latency_repeats={min(10, cfg.performance.latency_repeats)}",
    ]


def dump_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
