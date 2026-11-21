"""Global seeding for python, numpy and torch (including DataLoader workers)."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG we use and request deterministic kernels where possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    return seed


def rng(seed: int) -> np.random.Generator:
    """Independent numpy generator; does not perturb the global stream."""
    return np.random.default_rng(seed)


@contextlib.contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - exercised via DataLoader
    """Make DataLoader workers deterministic and independent."""
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
