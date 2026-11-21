"""Environment fingerprint recorded in every MLflow run (reproducibility)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "mlflow",
    "optuna",
    "pydantic",
    "matplotlib",
    "seaborn",
]


def git_hash(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return "unknown"
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if dirty.stdout.strip():
            sha += "-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"


def _cpu_model() -> str:
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        return str(info.get("brand_raw") or info.get("brand") or "unknown")
    except Exception:  # pragma: no cover - optional dependency details
        return platform.processor() or "unknown"


def _versions() -> dict[str, str]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def env_info(include_git: bool = True) -> dict[str, Any]:
    """Complete fingerprint: CPU, RAM, OS, libraries, threads, git, hardware."""
    import psutil

    try:
        import torch

        torch_threads = torch.get_num_threads()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mps = getattr(torch.backends, "mps", None)
        if device == "cpu" and mps is not None and mps.is_available():
            device = "mps"
        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
    except Exception:  # pragma: no cover
        torch_threads = 0
        accelerator = "cpu"

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "hostname": socket.gethostname(),
        "threads": {
            "torch": torch_threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        },
        "accelerator": accelerator,
        "gpu_available": accelerator == "cuda",
        "mps_available": accelerator == "mps",
        "versions": _versions(),
        "git": git_hash() if include_git else "skipped",
    }
    return info
