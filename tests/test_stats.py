from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mlbe.stats.compare import (
    bootstrap_ci,
    corrected_resampled_ttest,
    critical_difference,
    friedman_test,
    holm_correction,
    mcnemar,
    nemenyi_posthoc,
    paired_bootstrap_difference,
    wilcoxon_signed_rank,
)


def test_corrected_resampled_ttest_matches_formula():
    a = np.array([0.90, 0.92, 0.91, 0.93, 0.89, 0.92])
    b = np.array([0.88, 0.89, 0.90, 0.91, 0.87, 0.90])
    n_train, n_test = 400, 100
    out = corrected_resampled_ttest(a, b, n_train, n_test)
    diff = a - b
    k = len(a)
    correction = 1 / k + n_test / n_train
    expected_t = diff.mean() / np.sqrt(correction * diff.var(ddof=1))
    assert out["t"] == pytest.approx(expected_t)
    assert out["p_value"] == pytest.approx(2 * stats.t.sf(abs(expected_t), df=k - 1))
    assert out["p_value"] < 0.05


def test_corrected_ttest_handles_identical_scores():
    a = np.array([0.9, 0.9, 0.9])
    out = corrected_resampled_ttest(a, a, 100, 50)
    assert out["t"] == 0.0
    assert out["p_value"] == 1.0


def test_wilcoxon_matches_scipy():
    a = np.array([0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86])
    b = np.array([0.84, 0.83, 0.85, 0.90, 0.82, 0.88, 0.84])
    out = wilcoxon_signed_rank(a, b)
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
    assert out["statistic"] == pytest.approx(stat)
    assert out["p_value"] == pytest.approx(p)


def test_mcnemar_known_example():
    y = np.zeros(60, dtype=int)
    a = y.copy()
    b = y.copy()
    # 10 samples correct only for A, 2 correct only for B
    a[:10] = 1
    b[10:12] = 1
    out = mcnemar(y, a, b)
    assert out["b_only_a_correct"] == 2
    assert out["c_only_b_correct"] == 10
    expected = stats.binomtest(10, 12, 0.5).pvalue
    assert out["p_value"] == pytest.approx(expected)
    assert out["p_value"] < 0.05


def test_mcnemar_no_discordance():
    y = np.array([0, 1, 1, 0])
    out = mcnemar(y, y, y)
    assert out["p_value"] == 1.0


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    rng = np.random.default_rng(0)
    values = rng.normal(0.8, 0.05, size=50)
    lo, hi = bootstrap_ci(values, seed=3, n_boot=500)
    assert lo < values.mean() < hi
    lo2, hi2 = bootstrap_ci(values, seed=3, n_boot=500)
    assert (lo, hi) == (lo2, hi2)


def test_paired_bootstrap_difference_detects_clear_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.90, 0.01, 20)
    b = rng.normal(0.80, 0.01, 20)
    out = paired_bootstrap_difference(a, b, n_boot=400, seed=1)
    assert out["mean_diff"] == pytest.approx(0.1, abs=0.02)
    assert out["ci_low"] > 0
    assert out["p_two_sided"] < 0.05


def test_friedman_matches_scipy():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(6, 4)) + np.array([0.3, 0.1, 0.0, 0.2])
    out = friedman_test(matrix)
    stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(4)])
    assert out["statistic"] == pytest.approx(stat)
    assert out["p_value"] == pytest.approx(p)
    assert len(out["ranks"]) == 4


def test_critical_difference_formula_and_nemenyi():
    cd = critical_difference(4, 6)
    expected = 2.569 * np.sqrt(4 * 5 / (6 * 6))
    assert cd == pytest.approx(expected)
    out = nemenyi_posthoc([1.0, 2.0, 3.0, 4.5], 6, ["a", "b", "c", "d"])
    assert len(out["pairs"]) == 6
    assert out["critical_difference"] == pytest.approx(expected)
    assert any(pair["significant"] for pair in out["pairs"])


def test_holm_correction_is_order_preserving_and_bounded():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = holm_correction(p)
    assert adj[0] <= adj[1] + 1e-12
    assert all(0 <= x <= 1 for x in adj)
    assert adj[0] >= p[0]
