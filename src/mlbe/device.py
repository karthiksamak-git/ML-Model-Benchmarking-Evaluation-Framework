"""Device selection: CUDA/MPS when available, CPU otherwise (CPU-first)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    kind: str
    torch_threads: int
    reason: str


def get_device(prefer: str | None = None) -> str:
    """Return a torch device string, honouring an explicit preference."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def device_info(prefer: str | None = None) -> DeviceInfo:
    dev = get_device(prefer)
    kind = dev.split(":")[0]
    try:
        import torch

        threads = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        threads = 0
    if kind == "cuda":
        import torch

        reason = torch.cuda.get_device_name(0)
    elif kind == "mps":
        reason = "Apple Metal Performance Shaders"
    else:
        reason = "CPU fallback (no CUDA/MPS available)" if not prefer else "explicit CPU"
    return DeviceInfo(device=dev, kind=kind, torch_threads=threads, reason=reason)
