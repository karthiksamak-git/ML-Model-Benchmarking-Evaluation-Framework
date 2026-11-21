from __future__ import annotations

import random

import numpy as np
import torch

from mlbe.seed import numpy_seed, rng, seed_everything, worker_init_fn


def _draw() -> dict:
    return {
        "py": [random.random() for _ in range(3)],
        "np": np.random.rand(3).tolist(),
        "torch": torch.rand(3).tolist(),
    }


def test_seed_everything_is_reproducible_across_processes():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    second = _draw()
    assert first == second


def test_different_seeds_differ():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    b = _draw()
    assert a["np"] != b["np"]


def test_generator_is_independent_of_global_state():
    seed_everything(0)
    before = np.random.rand()
    g1 = rng(5).normal(size=4)
    seed_everything(0)
    g2 = rng(5).normal(size=4)
    assert np.array_equal(g1, g2)
    assert np.random.rand() == before


def test_numpy_seed_context_restores_state():
    seed_everything(3)
    expected = np.random.rand()
    seed_everything(3)
    with numpy_seed(9):
        np.random.rand()
    assert np.random.rand() == expected


def test_worker_init_is_deterministic():
    torch.manual_seed(4)
    worker_init_fn(0)
    a = (np.random.rand(), random.random())
    torch.manual_seed(4)
    worker_init_fn(0)
    b = (np.random.rand(), random.random())
    assert a == b
