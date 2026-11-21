"""Rubric validation against planted defects, plus JSON/markdown reports.

The validation table is produced from real runs of the rubric on datasets built
by ``mlbe.data.corruptions``. No numbers are hand-written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import dump_json
from ..data.corruptions import DEFECT_SPECS, make_planted_dataset
from .dataset_quality import evaluate_corrupted
from .scoring import letter_grade

EXPECTED_CHECK = {
    "clean": None,
    "label_noise_5": "label_noise",
    "label_noise_10": "label_noise",
    "label_noise_20": "label_noise",
    "duplicates_10": "duplicates",
    "conflicting_duplicates_5": "conflicting_labels",
    "leaky_feature": "leakage",
    "missing_15": "missing_values",
    "outliers_10": "outliers",
    "covariate_shift": "covariate_shift",
    "imbalance": "class_imbalance",
    "train_test_overlap_2": "leakage",
}


def _prf(truth: set[int], predicted: set[int]) -> dict[str, float]:
    tp = len(truth & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(tp),
        "n_predicted": int(len(predicted)),
        "n_truth": int(len(truth)),
    }


def validate_dataset_rubric(cfg: Any, seed: int = 42, defects: list[str] | None = None) -> dict[str, Any]:
    tolerance = float(cfg.dataset_quality.thresholds.label_noise_rate_tolerance)
    names = defects or list(DEFECT_SPECS)
    rows = []
    for defect in names:
        data = make_planted_dataset(defect, seed=seed)
        report = evaluate_corrupted(data, cfg, seed=seed)
        checks = {c["id"]: c for c in report["checks"]}
        expected = EXPECTED_CHECK.get(defect)
        if expected is None:
            detected = None
            actual_severity = checks[next(iter(checks))]["severity"] if checks else "info"
            evidence = {}
        else:
            check = checks[expected]
            actual_severity = check["severity"]
            # a defect counts as detected when the responsible check warns or fails
            detected = actual_severity in {"warn", "fail"}
            evidence = check["evidence"]
        row: dict[str, Any] = {
            "defect": defect,
            "expected_check": expected,
            "actual_severity": actual_severity,
            "detected": detected,
            "overall_score": report["score"],
            "grade": report["grade"],
            "evidence": evidence,
        }
        if defect.startswith("label_noise"):
            truth = set(int(i) for i in data.noisy_indices())
            predicted = set(int(i) for i in report.get("label_noise_flags", []))
            planted = float(checks["label_noise"]["evidence"]["estimated_noise_rate"]) * 0 + {
                "label_noise_5": 0.05,
                "label_noise_10": 0.10,
                "label_noise_20": 0.20,
            }[defect]
            estimate = float(checks["label_noise"]["evidence"]["estimated_noise_rate"])
            rel_tol = float(getattr(cfg.dataset_quality.thresholds, "label_noise_relative_tolerance", 0.5))
            allowed = max(tolerance, rel_tol * planted)
            row["label_noise"] = {
                "planted_rate": planted,
                "estimated_rate": estimate,
                "abs_error": round(abs(estimate - planted), 4),
                "within_tolerance": bool(abs(estimate - planted) <= allowed),
                "tolerance": round(allowed, 4),
                "absolute_tolerance": tolerance,
                "relative_tolerance": rel_tol,
                "identification": _prf(truth, predicted),
            }
        rows.append(row)

    defect_rows = [r for r in rows if r["detected"] is not None]
    detections = [r["detected"] for r in defect_rows]
    clean_rows = [r for r in rows if r["defect"] == "clean"]
    noise_rows = [r for r in rows if "label_noise" in r]
    summary = {
        "n_defect_datasets": len(defect_rows),
        "n_detected": int(sum(bool(d) for d in detections)),
        "detection_rate": round(float(np.mean(detections)) if detections else 0.0, 4),
        "clean_score": clean_rows[0]["overall_score"] if clean_rows else None,
        "clean_grade": clean_rows[0]["grade"] if clean_rows else None,
        "clean_false_alarm_checks": (
            [
                {"id": c["id"], "severity": c["severity"]}
                for c in evaluate_corrupted(make_planted_dataset("clean", seed=seed), cfg, seed=seed)["checks"]
                if c["severity"] != "info"
            ]
            if clean_rows
            else []
        ),
        "label_noise_tolerance": tolerance,
        "label_noise_within_tolerance": round(
            float(np.mean([r["label_noise"]["within_tolerance"] for r in noise_rows])) if noise_rows else 0.0, 4
        ),
        "label_noise_mean_precision": round(
            float(np.mean([r["label_noise"]["identification"]["precision"] for r in noise_rows])) if noise_rows else 0.0,
            4,
        ),
        "label_noise_mean_recall": round(
            float(np.mean([r["label_noise"]["identification"]["recall"] for r in noise_rows])) if noise_rows else 0.0,
            4,
        ),
        "seed": seed,
    }
    return {"rows": rows, "summary": summary}


def rubric_validation_markdown(validation: dict[str, Any]) -> str:
    s = validation["summary"]
    lines = [
        "# Rubric validation (planted defects)",
        "",
        f"Generated from real rubric runs with seed {s['seed']}. "
        f"Detection = the responsible check returned `warn` or `fail`.",
        "",
        f"- Defect datasets: **{s['n_defect_datasets']}**, detected: **{s['n_detected']}** "
        f"(detection rate **{s['detection_rate']:.0%}**)",
        f"- Clean dataset score: **{s['clean_score']}** ({s['clean_grade']})",
        f"- Label-noise estimates within ±{s['label_noise_tolerance']:.2f}: "
        f"**{s['label_noise_within_tolerance']:.0%}** of noise datasets",
        f"- Noisy-label identification precision/recall: "
        f"**{s['label_noise_mean_precision']:.2f} / {s['label_noise_mean_recall']:.2f}**",
        "",
        "| Defect | Expected check | Severity | Detected | Overall | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for row in validation["rows"]:
        ev = row.get("evidence") or {}
        key_bits = []
        for k in (
            "overall_missing_rate",
            "duplicate_rate",
            "rate",
            "estimated_noise_rate",
            "exact_overlap_rate",
            "outlier_rate",
            "adversarial_auc",
            "best_single_feature_auc",
            "imbalance_ratio",
            "minority_share",
        ):
            if k in ev:
                key_bits.append(f"{k}={ev[k]}")
        if "label_noise" in row:
            ln = row["label_noise"]
            key_bits.append(
                f"planted={ln['planted_rate']:.2f}, est={ln['estimated_rate']:.3f}, "
                f"P={ln['identification']['precision']:.2f}, R={ln['identification']['recall']:.2f}"
            )
        lines.append(
            f"| {row['defect']} | {row['expected_check'] or '—'} | {row['actual_severity']} | "
            f"{row['detected'] if row['detected'] is not None else '—'} | {row['overall_score']} | "
            f"{'; '.join(key_bits) or '—'} |"
        )
    return "\n".join(lines) + "\n"


def write_rubric_outputs(
    validation: dict[str, Any],
    dataset_reports: list[dict[str, Any]],
    model_reports: list[dict[str, Any]],
    cfg: Any,
    out_dir: str | Path,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg.to_dict()
    dataset_score = (
        float(np.mean([r["score"] for r in dataset_reports])) if dataset_reports else 0.0
    )
    model_score = (
        float(np.mean([r["score"] for r in model_reports])) if model_reports else 0.0
    )
    overall = 100.0 * (
        float(cfg.scoring.overall_weights.dataset_quality) * dataset_score
        + float(cfg.scoring.overall_weights.model_correctness) * model_score
    ) / 100.0
    summary = {
        "dataset_quality_score": round(dataset_score, 2),
        "model_correctness_score": round(model_score, 2),
        "overall_score": round(overall, 2),
        "overall_grade": letter_grade(overall, cfg),
        "weights": cfg.scoring.overall_weights.to_dict(),
    }
    payload = {
        "validation": validation,
        "datasets": dataset_reports,
        "models": model_reports,
        "summary": summary,
        "config": cfg_dict,
    }
    json_path = out / "rubric_report.json"
    dump_json(payload, json_path)

    lines = [
        "# Automated evaluation rubric",
        "",
        f"Overall score **{summary['overall_score']}** ({summary['overall_grade']}) — "
        f"dataset quality {summary['dataset_quality_score']}, "
        f"model correctness {summary['model_correctness_score']}.",
        "",
        "## Dataset quality",
        "",
        "| Dataset | Score | Grade | Failed checks |",
        "|---|---|---|---|",
    ]
    for r in dataset_reports:
        failed = ", ".join(c["id"] for c in r["checks"] if c["severity"] == "fail") or "—"
        lines.append(f"| {r.get('dataset', 'dataset')} | {r['score']} | {r['grade']} | {failed} |")
    lines += ["", "## Model correctness", "", "| Dataset | Model | Score | Grade | Failed checks |", "|---|---|---|---|---|"]
    for r in model_reports:
        failed = ", ".join(c["id"] for c in r["checks"] if c["severity"] == "fail") or "—"
        lines.append(
            f"| {r.get('dataset', '')} | {r.get('model', '')} | {r['score']} | {r['grade']} | {failed} |"
        )
    md_path = out / "rubric_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    val_path = out / "rubric_validation.md"
    val_path.write_text(rubric_validation_markdown(validation), encoding="utf-8")
    dump_json(validation, out / "rubric_validation.json")
    return {"json": json_path, "markdown": md_path, "validation_md": val_path}


def load_rubric_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
