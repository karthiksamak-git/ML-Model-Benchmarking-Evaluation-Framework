"""Statistical model comparison.

Implements the corrected resampled t-test (Nadeau & Bengio, 2003), Wilcoxon
signed-rank, McNemar's test on paired predictions, bootstrap confidence
intervals, and Friedman + Nemenyi across datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from scipy import stats

# Two-sided 0.05 critical values of the Nemenyi statistic (Studentized range / sqrt(2))
NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


def corrected_resampled_ttest(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    """Nadeau-Bengio corrected resampled t-test for repeated CV.

    Corrects the naive t-test for the overlap between resampled training sets:
    var_corrected = (1/J + n_test/n_train) * s^2 where J is the number of
    resamples. Positive t means model A > model B.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("score vectors must have equal length")
    n = len(a)
    if n < 2:
        return {"t": 0.0, "p_value": 1.0, "mean_diff": float(a.mean() - b.mean()) if n else 0.0, "df": 0}
    diff = a - b
    mean_diff = float(diff.mean())
    var = float(diff.var(ddof=1))
    if var == 0.0:
        t = 0.0
        p = 1.0
    else:
        correction = 1.0 / n + float(n_test) / float(max(n_train, 1))
        se = np.sqrt(correction * var)
        t = float(mean_diff / se)
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return {"t": t, "p_value": p, "mean_diff": mean_diff, "df": n - 1}


def wilcoxon_signed_rank(
    scores_a: Sequence[float], scores_b: Sequence[float]
) -> dict[str, float]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": float(diff.mean())}
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
    return {"statistic": float(stat), "p_value": float(p), "mean_diff": float(diff.mean())}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test on paired predictions (binomial on discordant pairs)."""
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    return {
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "statistic": float((abs(b - c) - 1) ** 2 / n) if n else 0.0,
        "p_value": p,
        "n_discordant": n,
    }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    gen = np.random.default_rng(seed)
    stats_boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = gen.choice(arr, size=arr.size, replace=True)
        stats_boot[i] = statistic(sample)
    lo, hi = np.percentile(stats_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def paired_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    gen = np.random.default_rng(seed)
    boot_means = np.array(
        [float(gen.choice(diff, size=diff.size, replace=True).mean()) for _ in range(n_boot)]
    ) if diff.size else np.array([0.0])
    p = min(1.0, 2.0 * min(float(np.mean(boot_means <= 0.0)), float(np.mean(boot_means >= 0.0))))
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p,
    }


def friedman_test(scores: np.ndarray) -> dict[str, Any]:
    """scores: (n_datasets, n_models) accuracy matrix. Ranks are computed per dataset."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return {"statistic": 0.0, "p_value": 1.0, "ranks": [], "n_datasets": int(matrix.shape[0]) if matrix.ndim == 2 else 0}
    stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    ranks = stats.rankdata(-matrix, axis=1).mean(axis=0)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "ranks": [float(r) for r in ranks],
        "n_datasets": int(matrix.shape[0]),
        "n_models": int(matrix.shape[1]),
    }


def critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    q = NEMENYI_Q05.get(n_models, 3.4)
    if alpha != 0.05:
        # monotone approximation for other alpha values, documented as approximate
        q *= (1.0 + (0.05 - alpha))
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * max(n_datasets, 1))))


def nemenyi_posthoc(
    avg_ranks: Sequence[float], n_datasets: int, model_names: Sequence[str]
) -> dict[str, Any]:
    k = len(avg_ranks)
    cd = critical_difference(k, n_datasets)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(float(avg_ranks[i]) - float(avg_ranks[j]))
            pairs.append(
                {
                    "model_a": model_names[i],
                    "model_b": model_names[j],
                    "rank_diff": diff,
                    "significant": bool(diff > cd),
                }
            )
    return {"critical_difference": cd, "pairs": pairs, "alpha": 0.05}


def compare_models_on_dataset(
    scores: dict[str, Sequence[float]],
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pairwise corrected t-test + Wilcoxon + bootstrap CI for one dataset."""
    names = sorted(scores)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa, sb = np.asarray(scores[a], dtype=float), np.asarray(scores[b], dtype=float)
            t = corrected_resampled_ttest(sa, sb, n_train, n_test)
            w = wilcoxon_signed_rank(sa, sb)
            boot = paired_bootstrap_difference(sa, sb, seed=seed)
            out.append(
                {
                    "dataset": None,
                    "model_a": a,
                    "model_b": b,
                    "mean_a": float(sa.mean()),
                    "mean_b": float(sb.mean()),
                    "mean_diff": t["mean_diff"],
                    "t_stat": t["t"],
                    "t_p_value": t["p_value"],
                    "wilcoxon_p_value": w["p_value"],
                    "bootstrap_ci_low": boot["ci_low"],
                    "bootstrap_ci_high": boot["ci_high"],
                    "significant": bool(min(t["p_value"], w["p_value"]) < 0.05),
                }
            )
    return out


def holm_correction(p_values: Iterable[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, order-preserving."""
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return [float(x) for x in adjusted]
