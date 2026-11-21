"""Rubric scoring primitives.

All checks return a score in [0, 1] plus a severity. Score bands and severity
thresholds come from ``configs/rubrics.yaml`` (``severity.info_min`` and
``severity.warn_min``); the piecewise mapping below is the documented,
deterministic way raw evidence becomes a score:

* at or better than the ``ok`` anchor -> 1.00
* at the ``warn`` anchor -> 0.85
* at the ``fail`` anchor -> 0.55
* at or worse than twice the ``fail`` anchor -> 0.00 (linear in between)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _tail(value: float, fail: float, direction: int) -> float:
    span = max(abs(fail), 1e-9)
    excess = max(0.0, direction * (value - fail))
    return max(0.0, 0.55 - 0.55 * min(1.0, excess / span))


def score_lower_better(value: float, ok: float, warn: float, fail: float) -> float:
    """Map a metric where smaller is better onto [0, 1]."""
    value = float(value)
    if value <= ok:
        return 1.0
    if value <= warn:
        span = max(warn - ok, 1e-12)
        return 1.0 - 0.15 * (value - ok) / span
    if value <= fail:
        span = max(fail - warn, 1e-12)
        return 0.85 - 0.30 * (value - warn) / span
    return _tail(value, fail, +1)


def score_higher_better(value: float, ok: float, warn: float, fail: float) -> float:
    """Map a metric where larger is better onto [0, 1]."""
    value = float(value)
    if value >= ok:
        return 1.0
    if value >= warn:
        span = max(ok - warn, 1e-12)
        return 1.0 - 0.15 * (ok - value) / span
    if value >= fail:
        span = max(warn - fail, 1e-12)
        return 0.85 - 0.30 * (warn - value) / span
    return _tail(-value, -fail, +1)


def severity_for(score: float, cfg: Any) -> str:
    if score >= float(cfg.severity.info_min):
        return "info"
    if score >= float(cfg.severity.warn_min):
        return "warn"
    return "fail"


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def make_check(
    check_id: str,
    name: str,
    category: str,
    score: float,
    evidence: dict[str, Any],
    recommendation: str,
    cfg: Any,
    severity: str | None = None,
) -> dict[str, Any]:
    score = clamp01(score)
    return {
        "id": check_id,
        "name": name,
        "category": category,
        "score": round(score, 4),
        "severity": severity or severity_for(score, cfg),
        "evidence": evidence,
        "recommendation": recommendation,
    }


def aggregate(checks: Iterable[dict[str, Any]], weights: dict[str, float]) -> float:
    checks = list(checks)
    if not checks:
        return 0.0
    total_w = 0.0
    total = 0.0
    for check in checks:
        w = float(weights.get(check["id"], 0.0))
        if w <= 0:
            continue
        total += w * float(check["score"])
        total_w += w
    if total_w == 0:
        return 0.0
    return 100.0 * total / total_w


def letter_grade(score: float, cfg: Any) -> str:
    for band in cfg.scoring.grades:
        if score >= float(band.min):
            return str(band.grade)
    return "F"


def build_report(
    checks: list[dict[str, Any]],
    weights: dict[str, float],
    category: str,
    cfg: Any,
) -> dict[str, Any]:
    score = aggregate(checks, weights)
    failing = [c for c in checks if c["severity"] != "info"]
    worst = "info"
    for check in checks:
        if check["severity"] == "fail":
            worst = "fail"
            break
        if check["severity"] == "warn":
            worst = "warn"
    return {
        "category": category,
        "score": round(score, 2),
        "grade": letter_grade(score, cfg),
        "severity": worst,
        "checks": checks,
        "failing_checks": [
            {"id": c["id"], "severity": c["severity"], "recommendation": c["recommendation"]}
            for c in failing
        ],
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["severity"] == "info"),
    }
