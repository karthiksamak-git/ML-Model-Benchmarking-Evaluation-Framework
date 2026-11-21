from __future__ import annotations

import pytest

from mlbe.config import load_configs, rubric_config
from mlbe.data.corruptions import DEFECT_SPECS, make_planted_dataset
from mlbe.data.registry import load_dataset
from mlbe.data.splits import make_splits
from mlbe.rubrics.dataset_quality import evaluate_corrupted
from mlbe.rubrics.model_correctness import evaluate_model_correctness
from mlbe.rubrics.report import EXPECTED_CHECK, validate_dataset_rubric
from mlbe.rubrics.scoring import score_higher_better, score_lower_better, severity_for


@pytest.fixture(scope="module")
def rub():
    return rubric_config()


@pytest.fixture(scope="module")
def validation(rub):
    return validate_dataset_rubric(rub, seed=42)


def test_clean_dataset_scores_high(rub):
    report = evaluate_corrupted(make_planted_dataset("clean", seed=42), rub, seed=42)
    assert report["score"] >= 90
    assert report["grade"] in {"A", "B"}
    assert all(c["severity"] == "info" for c in report["checks"])


def test_every_planted_defect_is_detected(validation):
    rows = {r["defect"]: r for r in validation["rows"]}
    for defect, expected in EXPECTED_CHECK.items():
        if expected is None:
            continue
        row = rows[defect]
        assert row["expected_check"] == expected
        assert row["detected"] is True, f"{defect} not detected (severity {row['actual_severity']})"
        assert row["actual_severity"] in {"warn", "fail"}
    assert validation["summary"]["detection_rate"] == 1.0


def test_label_noise_estimate_within_documented_tolerance(validation):
    rows = [r for r in validation["rows"] if "label_noise" in r]
    assert {r["defect"] for r in rows} == {"label_noise_5", "label_noise_10", "label_noise_20"}
    for row in rows:
        assert row["label_noise"]["within_tolerance"], row
        assert 0.0 <= row["label_noise"]["identification"]["recall"] <= 1.0
    assert validation["summary"]["label_noise_mean_precision"] > 0.5
    assert validation["summary"]["label_noise_mean_recall"] > 0.5
    estimates = [r["label_noise"]["estimated_rate"] for r in rows]
    assert estimates == sorted(estimates)  # monotone in the planted rate


def test_noisy_label_identification_beats_random(validation):
    row = next(r for r in validation["rows"] if r["defect"] == "label_noise_20")
    ident = row["label_noise"]["identification"]
    assert ident["precision"] > 0.5
    assert ident["recall"] > 0.5


def test_score_bands_are_monotone():
    assert score_lower_better(0.0, 0.0, 0.05, 0.2) == 1.0
    assert score_lower_better(0.05, 0.0, 0.05, 0.2) == pytest.approx(0.85)
    assert score_lower_better(0.2, 0.0, 0.05, 0.2) == pytest.approx(0.55)
    assert score_lower_better(0.4, 0.0, 0.05, 0.2) < 0.55
    assert score_higher_better(1.0, 0.9, 0.7, 0.5) == 1.0
    assert score_higher_better(0.7, 0.9, 0.7, 0.5) == pytest.approx(0.85)
    cfg = load_configs()["rubrics"]
    assert severity_for(0.9, cfg) == "info"
    assert severity_for(0.7, cfg) == "warn"
    assert severity_for(0.2, cfg) == "fail"


def test_model_correctness_report_structure(rub):
    cfg = load_configs()["benchmark"]
    ds = load_dataset("breast_cancer", cfg.datasets["breast_cancer"], max_samples=300)
    splits = make_splits(ds.y, 2, 1, seed=1, dataset="bc")
    report = evaluate_model_correctness(
        "breast_cancer", ds.X, ds.y, splits, "logreg", cfg, rub, seed=1
    )
    ids = {c["id"] for c in report["checks"]}
    assert {
        "baseline_significance",
        "generalization_gap",
        "stability",
        "calibration",
        "per_class_performance",
        "robustness",
        "prediction_sanity",
        "reproducibility",
        "efficiency",
    } == ids
    assert 0 <= report["score"] <= 100
    assert report["grade"] in {"A", "B", "C", "D", "F"}
    assert report["model"] == "logreg"
    sanity = next(c for c in report["checks"] if c["id"] == "prediction_sanity")
    assert sanity["severity"] == "info"
    repro = next(c for c in report["checks"] if c["id"] == "reproducibility")
    assert repro["severity"] == "info"


def test_all_defect_specs_include_expected_check():
    for defect in DEFECT_SPECS:
        assert defect in EXPECTED_CHECK
