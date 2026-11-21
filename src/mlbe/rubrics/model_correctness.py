"""Hand-written model-correctness rubric checks.

The checks cover: beating trivial baselines with significance, generalization
gap, stability across folds, calibration, per-class performance, robustness,
prediction sanity, reproducibility and efficiency. Thresholds come from
``configs/rubrics.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from ..benchmark.performance import measure_latency, model_complexity
from ..metrics import compute_metrics, per_class_report
from ..models.registry import build_model
from ..seed import seed_everything
from ..stats.compare import corrected_resampled_ttest
from .scoring import build_report, make_check, score_higher_better, score_lower_better


def _fit(model_name, X_train, y_train, seed, params, image_shape, n_classes, n_features):
    model = build_model(
        model_name,
        seed=seed,
        params=params,
        n_classes=n_classes,
        input_dim=n_features,
        image_shape=image_shape,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model_correctness(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    model_name: str,
    cfg: Any,
    rubric_cfg: Any,
    seed: int = 0,
    params: dict | None = None,
) -> dict[str, Any]:
    t = rubric_cfg.model_correctness.thresholds
    X = np.asarray(X)
    y = np.asarray(y)
    labels = np.unique(y)
    n_classes = len(labels)
    # image datasets are stored flattened; recover the side length from the config
    image_shape = None
    if model_name == "cnn":
        side = int(round(np.sqrt(X.shape[1])))
        image_shape = (side, side)

    fold_scores: list[float] = []
    fold_train_scores: list[float] = []
    dummy_scores: dict[str, list[float]] = {"most_frequent": [], "stratified": []}
    first: dict[str, Any] | None = None
    latencies: list[float] = []

    for split in splits:
        fold_seed = seed + 1000 * split.repeat + split.fold
        seed_everything(fold_seed)
        X_tr, X_te = X[split.train_idx], X[split.test_idx]
        y_tr, y_te = y[split.train_idx], y[split.test_idx]
        model = _fit(model_name, X_tr, y_tr, fold_seed, params, image_shape, n_classes, X.shape[1])
        proba = model.predict_proba(X_te)
        pred = model.predict(X_te)
        acc = float(accuracy_score(y_te, pred))
        fold_scores.append(acc)
        fold_train_scores.append(float(accuracy_score(y_tr, model.predict(X_tr))))
        for strategy in dummy_scores:
            dummy = DummyClassifier(strategy=strategy, random_state=fold_seed)
            dummy.fit(X_tr, y_tr)
            dummy_scores[strategy].append(float(accuracy_score(y_te, dummy.predict(X_te))))
        latencies.append(measure_latency(model, X_te, repeats=10, warmup=2)["latency_p50_ms"])
        if first is None:
            first = {
                "model": model,
                "proba": proba,
                "pred": pred,
                "y_true": y_te,
                "y_train": y_tr,
                "X_train": X_tr,
                "X_test": X_te,
                "split": split,
                "metrics": compute_metrics(y_te, pred, proba, labels),
            }

    assert first is not None
    checks = []
    n_train = len(first["split"].train_idx)
    n_test = len(first["split"].test_idx)

    # 1. beat trivial baselines with statistical significance
    best_baseline = max(
        ("most_frequent", dummy_scores["most_frequent"]),
        ("stratified", dummy_scores["stratified"]),
        key=lambda item: float(np.mean(item[1])),
    )
    ttest = corrected_resampled_ttest(fold_scores, best_baseline[1], n_train, n_test)
    diff = float(np.mean(fold_scores) - np.mean(best_baseline[1]))
    alpha = float(t.baseline_alpha)
    if diff <= 0:
        base_score = 0.0
    elif ttest["p_value"] < alpha and diff > float(t.baseline_margin_min):
        base_score = 1.0
    elif ttest["p_value"] < alpha:
        base_score = 0.85
    else:
        base_score = 0.6
    checks.append(
        make_check(
            "baseline_significance",
            "Performance vs trivial baselines",
            "model_correctness",
            base_score,
            {
                "model_accuracy": round(float(np.mean(fold_scores)), 4),
                "baseline": best_baseline[0],
                "baseline_accuracy": round(float(np.mean(best_baseline[1])), 4),
                "accuracy_diff": round(diff, 4),
                "corrected_t_p_value": round(float(ttest["p_value"]), 6),
                "alpha": alpha,
            },
            "A model that cannot significantly beat a dummy baseline adds no value.",
            rubric_cfg,
        )
    )

    # 2. generalization gap
    gap = float(np.mean(fold_train_scores) - np.mean(fold_scores))
    gap_score = score_lower_better(
        gap, ok=0.0, warn=float(t.generalization_gap_warn), fail=float(t.generalization_gap_fail)
    )
    checks.append(
        make_check(
            "generalization_gap",
            "Generalization gap (train vs test)",
            "model_correctness",
            gap_score,
            {
                "mean_train_accuracy": round(float(np.mean(fold_train_scores)), 4),
                "mean_test_accuracy": round(float(np.mean(fold_scores)), 4),
                "gap": round(gap, 4),
            },
            "Increase regularisation, reduce capacity or collect more data.",
            rubric_cfg,
        )
    )

    # 3. stability across folds
    std = float(np.std(fold_scores))
    checks.append(
        make_check(
            "stability",
            "Stability across folds",
            "model_correctness",
            score_lower_better(
                std, ok=0.005, warn=float(t.stability_std_warn), fail=float(t.stability_std_fail)
            ),
            {
                "fold_accuracy_std": round(std, 5),
                "fold_scores": [round(s, 4) for s in fold_scores],
                "n_folds": len(fold_scores),
            },
            "High fold variance points to an unstable model or too-small folds.",
            rubric_cfg,
        )
    )

    # 4. calibration
    m = first["metrics"]
    cal_score = min(
        score_lower_better(m["ece"], ok=0.01, warn=float(t.ece_warn), fail=float(t.ece_fail)),
        score_lower_better(m["brier"], ok=0.05, warn=float(t.brier_warn), fail=float(t.brier_fail)),
    )
    checks.append(
        make_check(
            "calibration",
            "Calibration (ECE / Brier)",
            "model_correctness",
            cal_score,
            {"ece": round(m["ece"], 4), "brier": round(m["brier"], 4)},
            "Recalibrate (Platt/isotonic) or use a probabilistic model if probabilities are used for decisions.",
            rubric_cfg,
        )
    )

    # 5. per-class minimum recall
    per_class = per_class_report(first["y_true"], first["pred"], labels)
    recalls = [c["recall"] for c in per_class]
    min_recall = float(min(recalls)) if recalls else 0.0
    checks.append(
        make_check(
            "per_class_performance",
            "Per-class minimum performance",
            "model_correctness",
            score_higher_better(
                min_recall,
                ok=0.85,
                warn=float(t.per_class_recall_warn),
                fail=float(t.per_class_recall_fail),
            ),
            {
                "min_recall": round(min_recall, 4),
                "per_class": [
                    {"class": c["class"], "recall": round(c["recall"], 4), "support": c["support"]}
                    for c in per_class
                ],
            },
            "A class with low recall is effectively unusable; rebalance, relabel or add features.",
            rubric_cfg,
        )
    )

    # 6. robustness to feature noise
    gen = np.random.default_rng(seed)
    X_train = first["X_train"]
    X_test = first["X_test"]
    std_features = np.nanstd(X_train, axis=0)
    std_features[std_features == 0] = 1.0
    model_first = first["model"]
    baseline_acc = float(accuracy_score(first["y_true"], first["pred"]))
    noisy = X_test + gen.normal(0.0, 1.0, size=X_test.shape) * std_features * 0.5
    noisy_acc = float(accuracy_score(first["y_true"], model_first.predict(noisy)))
    drop = baseline_acc - noisy_acc
    checks.append(
        make_check(
            "robustness",
            "Robustness under feature noise",
            "model_correctness",
            score_lower_better(
                drop, ok=0.01, warn=float(t.robustness_drop_warn), fail=float(t.robustness_drop_fail)
            ),
            {
                "clean_accuracy": round(baseline_acc, 4),
                "noisy_accuracy_(0.5sd)": round(noisy_acc, 4),
                "accuracy_drop": round(drop, 4),
            },
            "Test with realistic noise; augment training data or prefer robust models.",
            rubric_cfg,
        )
    )

    # 7. prediction sanity
    proba = first["proba"]
    row_sums = proba.sum(axis=1)
    sane = bool(
        np.isfinite(proba).all()
        and np.allclose(row_sums, 1.0, atol=1e-6)
        and np.array_equal(model_first.predict(X_test), first["pred"])
    )
    checks.append(
        make_check(
            "prediction_sanity",
            "Prediction sanity & determinism",
            "model_correctness",
            1.0 if sane else 0.0,
            {
                "probabilities_finite": bool(np.isfinite(proba).all()),
                "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
                "min_probability": float(proba.min()),
                "deterministic_inference": bool(np.array_equal(model_first.predict(X_test), first["pred"])),
            },
            "Fix the prediction path: probabilities must be finite and sum to one, inference must be deterministic.",
            rubric_cfg,
        )
    )

    # 8. reproducibility with the same seed
    seed_everything(seed + 1000 * first["split"].repeat + first["split"].fold)
    X_tr, y_tr = X_train, first["y_train"]
    repro_model = _fit(model_name, X_tr, y_tr, seed + 1000 * first["split"].repeat + first["split"].fold,
                       params, image_shape, n_classes, X.shape[1])
    repro_pred = repro_model.predict(X_test)
    repro_acc = float(accuracy_score(first["y_true"], repro_pred))
    tolerance = float(t.repro_torch_tolerance if model_name in {"mlp", "cnn"} else t.repro_metric_tolerance)
    delta = abs(repro_acc - baseline_acc)
    checks.append(
        make_check(
            "reproducibility",
            "Reproducibility (same seed)",
            "model_correctness",
            1.0 if delta <= tolerance else 0.0,
            {
                "original_accuracy": round(baseline_acc, 6),
                "rerun_accuracy": round(repro_acc, 6),
                "abs_diff": float(delta),
                "tolerance": tolerance,
            },
            "Seed every RNG and avoid non-deterministic ops; see `mlbe repro`.",
            rubric_cfg,
        )
    )

    # 9. efficiency
    latency = float(np.mean(latencies))
    complexity = model_complexity(model_first)
    budget_ms = float(t.latency_budget_ms)
    budget_mb = float(t.size_budget_mb)
    eff_score = min(
        score_lower_better(latency, ok=1.0, warn=budget_ms, fail=budget_ms * 10),
        score_lower_better(complexity["model_size_mb"], ok=0.5, warn=budget_mb, fail=budget_mb * 10),
    )
    checks.append(
        make_check(
            "efficiency",
            "Efficiency budget",
            "model_correctness",
            eff_score,
            {
                "latency_p50_ms": round(latency, 3),
                "latency_budget_ms": budget_ms,
                **{k: round(v, 4) for k, v in complexity.items()},
            },
            "Compress, prune or choose a cheaper model if latency/size exceeds the budget.",
            rubric_cfg,
        )
    )

    report = build_report(
        checks,
        rubric_cfg.model_correctness.weights.to_dict(),
        "model_correctness",
        rubric_cfg,
    )
    report.update(
        {
            "dataset": dataset_name,
            "model": model_name,
            "confusion_matrix": confusion_matrix(first["y_true"], first["pred"], labels=labels).tolist(),
            "labels": [int(v) for v in labels],
        }
    )
    return report
