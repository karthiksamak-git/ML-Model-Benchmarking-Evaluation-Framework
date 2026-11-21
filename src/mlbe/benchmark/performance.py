"""Computational performance: latency, throughput, memory, model size."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import psutil

from ..models.sklearn_models import model_size_bytes, param_count


def _time_calls(fn, n: int) -> np.ndarray:
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - start) * 1000.0
    return times


def measure_latency(model: Any, X: np.ndarray, repeats: int = 40, warmup: int = 5) -> dict[str, float]:
    """Single-sample inference latency in milliseconds (p50/p95)."""
    if len(X) == 0:
        return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_mean_ms": 0.0}
    sample = X[:1]

    def call() -> None:
        if hasattr(model, "predict_proba"):
            model.predict_proba(sample)
        else:  # pragma: no cover
            model.predict(sample)

    for _ in range(warmup):
        call()
    times = _time_calls(call, repeats)
    return {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_mean_ms": float(times.mean()),
    }


def measure_throughput(
    model: Any, X: np.ndarray, batch_sizes: list[int], repeats: int = 5
) -> dict[str, float]:
    """Samples/second for full batches of each configured size."""
    out: dict[str, float] = {}
    for bs in batch_sizes:
        n = min(int(bs), len(X))
        batch = X[:n]
        for _ in range(2):
            model.predict_proba(batch)
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict_proba(batch)
            best = min(best, time.perf_counter() - start)
        out[f"throughput_bs{bs}_sps"] = float(n / best) if best > 0 else 0.0
    return out


def measure_peak_memory(model: Any, X: np.ndarray) -> dict[str, float]:
    """Peak inference allocation (tracemalloc) and process RSS delta."""
    process = psutil.Process()
    rss_before = process.memory_info().rss / 1024**2
    tracemalloc.start()
    model.predict_proba(X)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss / 1024**2
    return {
        "inference_peak_mb": float(peak / 1024**2),
        "process_rss_mb": float(rss_after),
        "rss_delta_mb": float(rss_after - rss_before),
    }


def model_complexity(model: Any) -> dict[str, float]:
    if hasattr(model, "param_count"):
        params = int(model.param_count())
    else:
        params = param_count(model)
    return {
        "param_count": float(params),
        "model_size_mb": float(model_size_bytes(model) / 1024**2),
    }


def profile_model(
    model: Any,
    X_test: np.ndarray,
    cfg: Any,
    train_time_s: float,
) -> dict[str, float]:
    """Full performance profile recorded for each fitted model."""
    out: dict[str, float] = {"train_time_s": float(train_time_s)}
    out.update(measure_latency(model, X_test, int(cfg.latency_repeats), int(cfg.latency_warmup)))
    out.update(measure_throughput(model, X_test, [int(b) for b in cfg.batch_sizes]))
    out.update(measure_peak_memory(model, X_test[: min(len(X_test), int(cfg.memory_sample))]))
    out.update(model_complexity(model))
    return out
