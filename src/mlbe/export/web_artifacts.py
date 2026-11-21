"""Export MLflow runs + analysis artifacts to static JSON for the web dashboard.

Every exported number is read from MLflow (benchmark/HPO) or from JSON artifacts
produced by earlier real runs (analysis/rubric). Each file carries provenance:
git hash, generation time, MLflow experiment and source run IDs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

from ..config import PROJECT_ROOT, dump_json
from ..data.registry import load_dataset
from ..env_info import env_info, git_hash
from ..models.registry import MODEL_LABELS
from ..rubrics.dataset_quality import (
    DatasetView,
    check_class_imbalance,
    check_conflicting_labels,
    check_constant_columns,
    check_duplicates,
    check_low_information,
    check_missing_values,
    check_sample_adequacy,
)
from ..stats.compare import (
    bootstrap_ci,
    compare_models_on_dataset,
    friedman_test,
    holm_correction,
    mcnemar,
    nemenyi_posthoc,
)
from ..stats.ranking import score_models
from ..tracking.queries import (
    benchmark_parents,
    experiment_id,
    fold_runs,
    hpo_studies,
    hpo_trials,
    load_json_artifact,
    load_oof,
)

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "roc_auc_ovr",
    "log_loss",
    "brier",
    "ece",
]


def _provenance(exp: str, run_ids: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "git_hash": git_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_experiment": exp,
        "source_run_ids": sorted(set(run_ids))[:500],
        "generated_by": "mlbe export",
    }
    if extra:
        data.update(extra)
    return data


def _rows(client: MlflowClient, exp_id: str, exp_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = benchmark_parents(client, exp_id)
    rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for p in parents:
        folds = fold_runs(client, p["run_id"])
        scores = [f["metrics"].get("accuracy", float("nan")) for f in folds]
        scores = [s for s in scores if np.isfinite(s)]
        if not scores:
            continue
        ci_low, ci_high = bootstrap_ci(scores, seed=42)
        m = p["metrics"]
        row = {
            "run_id": p["run_id"],
            "dataset": p["dataset"],
            "model": p["model"],
            "label": MODEL_LABELS.get(p["model"], p["model"]),
            "n_folds": len(scores),
            "accuracy": float(m.get("accuracy_mean", np.mean(scores))),
            "accuracy_std": float(m.get("accuracy_std", np.std(scores))),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "gap": float(m.get("gap_accuracy_mean", 0.0)),
            "latency_p50_ms": float(m.get("perf_latency_p50_ms_mean", 0.0)),
            "latency_p95_ms": float(m.get("perf_latency_p95_ms_mean", 0.0)),
            "model_size_mb": float(m.get("perf_model_size_mb_mean", 0.0)),
            "param_count": float(m.get("perf_param_count_mean", 0.0)),
            "train_time_s": float(m.get("perf_train_time_s_mean", 0.0)),
            "throughput_bs128_sps": float(m.get("perf_throughput_bs128_sps_mean", 0.0)),
            "fold_scores": [round(float(s), 6) for s in scores],
        }
        for key in METRIC_KEYS:
            row[key] = float(m.get(f"{key}_mean", float("nan")))
            row[f"{key}_std"] = float(m.get(f"{key}_std", 0.0))
        rows.append(row)
        for f in folds:
            fold_records.append(
                {
                    "run_id": f["run_id"],
                    "dataset": p["dataset"],
                    "model": p["model"],
                    "split_id": f["tags"].get("split_id"),
                    "repeat": int(f["tags"].get("repeat", 0)),
                    "fold": int(f["tags"].get("fold", 0)),
                    "seed": int(float(f["params"].get("seed", 0))),
                    "n_train": int(float(f["params"].get("n_train", 0))),
                    "n_test": int(float(f["params"].get("n_test", 0))),
                    "dataset_n_samples": int(float(f["params"].get("n_samples", 0))),
                    "metrics": {
                        k: float(v)
                        for k, v in f["metrics"].items()
                        if k in METRIC_KEYS or k in {"train_accuracy", "train_f1_macro"} or k.startswith("gap_")
                    },
                }
            )
    return rows, fold_records


def _overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    # Only average across models evaluated on the same set of datasets (the
    # tabular suite). The CNN runs on one image dataset and would otherwise
    # look better than models that average four datasets.
    max_datasets = max((len(items) for items in by_model.values()), default=0)
    groups = []
    for model, items in by_model.items():
        if len(items) != max_datasets:
            continue
        groups.append(
            {
                "id": model,
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "dataset": "all",
                "accuracy": float(np.mean([r["accuracy"] for r in items])),
                "gap": float(np.mean([r["gap"] for r in items])),
                "latency_p50_ms": float(np.mean([r["latency_p50_ms"] for r in items])),
                "model_size_mb": float(np.mean([r["model_size_mb"] for r in items])),
                "n_datasets": len(items),
                "datasets": [r["dataset"] for r in items],
            }
        )
    # bootstrap CIs over every fold score the model has across datasets
    for g in groups:
        pool = [s for r in by_model[g["model"]] for s in r["fold_scores"]]
        lo, hi = bootstrap_ci(pool, seed=42)
        g["ci_low"], g["ci_high"] = lo, hi
    return score_models(groups)


def _stats(rows: list[dict[str, Any]], folds: list[dict[str, Any]], client: MlflowClient, exp_name: str) -> dict[str, Any]:
    datasets = sorted({r["dataset"] for r in rows})
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in folds:
        scores[f["dataset"]][f["model"]].append(float(f["metrics"].get("accuracy", np.nan)))
    pairwise: dict[str, Any] = {}
    for ds in datasets:
        ds_scores = {m: v for m, v in scores[ds].items() if len(v) > 1}
        n_train = next((int(f["n_train"]) for f in folds if f["dataset"] == ds), 0)
        n_test = next((int(f["n_test"]) for f in folds if f["dataset"] == ds), 0)
        comparisons = compare_models_on_dataset(ds_scores, n_train, n_test)
        adjusted = holm_correction([c["t_p_value"] for c in comparisons])
        for c, adj in zip(comparisons, adjusted, strict=False):
            c["dataset"] = ds
            c["t_p_value_holm"] = adj
        pairwise[ds] = comparisons

    # McNemar on paired first-split predictions
    oof: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in rows:
        arrays = load_oof(client, row["run_id"])
        if arrays is not None:
            oof[(row["dataset"], row["model"])] = arrays
    mcnemar_rows = []
    for ds in datasets:
        models = sorted(m for (d, m) in oof if d == ds)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                ya, yb = oof[(ds, a)], oof[(ds, b)]
                if not np.array_equal(ya["test_idx"], yb["test_idx"]):
                    continue
                res = mcnemar(ya["y_true"], ya["y_pred"], yb["y_pred"])
                mcnemar_rows.append(
                    {"dataset": ds, "model_a": a, "model_b": b, **res}
                )

    # Friedman over models present on every *multi-model* dataset. Image-only
    # datasets (the CNN suite) are excluded because no common model set exists.
    tabular = [d for d in datasets if len(scores[d]) > 1]
    common = sorted(set.intersection(*[set(scores[d]) for d in tabular])) if tabular else []
    matrix = np.array([[float(np.mean(scores[d][m])) for m in common] for d in tabular]) if common else np.empty((0, 0))
    friedman = friedman_test(matrix) if matrix.size else {"statistic": 0.0, "p_value": 1.0, "ranks": []}
    nemenyi = nemenyi_posthoc(friedman["ranks"], len(tabular), common) if common else {"critical_difference": 0.0, "pairs": []}
    friedman["datasets"] = tabular
    friedman["models"] = common
    friedman["avg_ranks"] = friedman.get("ranks", [])
    friedman["critical_difference"] = nemenyi["critical_difference"]
    friedman["nemenyi_pairs"] = nemenyi["pairs"]
    friedman["n_datasets"] = len(tabular)
    return {
        "provenance": _provenance(exp_name, [r["run_id"] for r in rows]),
        "pairwise": pairwise,
        "mcnemar": mcnemar_rows,
        "friedman": friedman,
        "n_datasets": len(datasets),
        "alpha": 0.05,
        "tests": {
            "pairwise": "corrected resampled t-test (Nadeau-Bengio), Wilcoxon signed-rank, paired bootstrap",
            "multiple_comparison": "Holm-Bonferroni over pairwise t-tests within each dataset",
            "omnibus": "Friedman test with Nemenyi post-hoc critical differences",
        },
    }


def _pareto(rows: list[dict[str, Any]], exp_name: str) -> dict[str, Any]:
    from ..stats.ranking import pareto_front

    datasets = sorted({r["dataset"] for r in rows})
    out = []
    for ds in datasets:
        group = [r for r in rows if r["dataset"] == ds]
        points = [
            {
                "id": r["model"],
                "model": r["model"],
                "label": r["label"],
                "accuracy": r["accuracy"],
                "latency_p50_ms": max(r["latency_p50_ms"], 1e-4),
                "model_size_mb": max(r["model_size_mb"], 1e-6),
            }
            for r in group
        ]
        front = set(pareto_front(points))
        for p in points:
            p["pareto"] = p["id"] in front
        out.append({"dataset": ds, "points": points})
    return {"provenance": _provenance(exp_name, [r["run_id"] for r in rows]), "datasets": out,
            "objectives": {"maximize": ["accuracy"], "minimize": ["latency_p50_ms", "model_size_mb"]}}


def _hpo(client: MlflowClient, exp_id: str, exp_name: str) -> dict[str, Any]:
    studies = hpo_studies(client, exp_id)
    out = []
    for s in studies:
        payload = load_json_artifact(client, s["run_id"], "hpo_summary.json") or {}
        trials = hpo_trials(client, s["run_id"])
        out.append(
            {
                "study_run_id": s["run_id"],
                "model": s["model"],
                "dataset": s["dataset"],
                "best_params": payload.get("best_params", {}),
                "best_cv_accuracy": payload.get("best_cv_accuracy", s["metrics"].get("best_cv_accuracy")),
                "best_cv_ci": payload.get("best_cv_ci"),
                "default_cv_ci": payload.get("default_cv_ci"),
                "tuned_test_accuracy": payload.get("tuned_test_accuracy"),
                "default_test_accuracy": payload.get("default_test_accuracy"),
                "test_improvement": payload.get("test_improvement"),
                "paired_bootstrap_ci": payload.get("paired_bootstrap_ci"),
                "mcnemar": payload.get("mcnemar"),
                "importance": payload.get("importance", {}),
                "history": payload.get("history", []),
                "protocol": payload.get("protocol", ""),
                "seed": payload.get("seed"),
                "n_trials": payload.get("n_trials"),
                "n_completed": payload.get("n_completed"),
                "elapsed_s": payload.get("elapsed_s"),
                "trial_runs": [
                    {
                        "run_id": t["run_id"],
                        "trial": int(float(t["tags"].get("trial_number", 0))),
                        "metrics": t["metrics"],
                    }
                    for t in trials
                ],
            }
        )
    return {"provenance": _provenance(exp_name, [s["study_run_id"] for s in out]), "studies": out}


def _load_local(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _golden_vectors(cfg: Any) -> dict[str, Any]:
    """Golden vectors for the TypeScript port of the browser-side rubric checks."""
    from ..config import rubric_config

    rub = rubric_config()
    rng = np.random.default_rng(7)
    cases: list[dict[str, Any]] = []

    def add_case(name: str, X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        # train and test are the same rows for golden purposes: the checks only
        # read the training matrix, and this keeps the missing-rate definition
        # identical to the browser implementation (one dataset).
        view = DatasetView(X, y, X, y, names, seed=7)
        checks = {
            "missing_values": check_missing_values(view, rub),
            "constant_columns": check_constant_columns(view, rub),
            "duplicates": check_duplicates(view, rub),
            "conflicting_labels": check_conflicting_labels(view, rub),
            "class_imbalance": check_class_imbalance(view, rub),
            "low_information": check_low_information(view, rub),
            "sample_adequacy": check_sample_adequacy(view, rub),
        }
        cases.append(
            {
                "name": name,
                "feature_names": names,
                "X_train": _json_matrix(X),
                "y_train": y.tolist(),
                "expected": {
                    cid: {
                        "score": c["score"],
                        "severity": c["severity"],
                        "evidence": {
                            k: v
                            for k, v in c["evidence"].items()
                            if k
                            in {
                                "overall_missing_rate",
                                "n_constant",
                                "n_near_constant",
                                "share",
                                "duplicate_rate",
                                "n_duplicate_rows",
                                "rate",
                                "imbalance_ratio",
                                "minority_share",
                                "share_pairs_above_threshold",
                                "share_below_threshold",
                                "samples_per_feature",
                                "min_samples_per_class",
                            }
                        },
                    }
                    for cid, c in checks.items()
                },
            }
        )

    add_case("clean_balanced", rng.normal(size=(40, 3)), np.array([0, 1] * 20), ["a", "b", "c"])
    X2 = rng.normal(size=(40, 3))
    X2[0, 0] = np.nan
    X2[1:5, 2] = np.nan
    X2[:, 1] = 1.0
    add_case("missing_and_constant", X2, np.array([0, 1] * 20), ["a", "b", "c"])
    X3 = rng.normal(size=(30, 2))
    y3 = np.array([0, 1] * 15)
    X3 = np.vstack([X3, X3[:4]])
    y3 = np.concatenate([y3, [1, 0, 1, 0]])
    add_case("duplicates_conflicting", X3, y3, ["a", "b"])
    add_case(
        "imbalanced",
        rng.normal(size=(36, 2)),
        np.array([0] * 32 + [1] * 4),
        ["a", "b"],
    )
    X5 = rng.normal(size=(36, 3))
    X5[:, 2] = 0.5
    y5 = np.array([0] * 32 + [1] * 4)
    add_case("low_information", X5, y5, ["a", "b", "const"])
    return {
        "config": {
            "dataset_quality": {
                "thresholds": rub.dataset_quality.thresholds.to_dict(),
                "weights": rub.dataset_quality.weights.to_dict(),
            },
            "severity": rub.severity.to_dict(),
        },
        "cases": cases,
        "notes": "Scores must match src/mlbe/rubrics/dataset_quality.py for the browser-implemented checks.",
    }


def _json_matrix(X: np.ndarray) -> list[list[float | None]]:
    out = []
    for row in X:
        out.append([None if not np.isfinite(v) else float(v) for v in row])
    return out


def export_all(cfg: Any, artifacts_dir: str | Path | None = None, web_dir: str | Path | None = None) -> dict[str, Any]:
    client = MlflowClient()
    exp_name = str(cfg.paths.experiment)
    exp_id = experiment_id(client, exp_name)
    artifacts = Path(artifacts_dir) if artifacts_dir else PROJECT_ROOT / "artifacts"
    web = Path(web_dir) if web_dir else PROJECT_ROOT / cfg.paths.web_data
    web.mkdir(parents=True, exist_ok=True)
    (web.parent.parent / "tests" / "golden").mkdir(parents=True, exist_ok=True)

    rows, folds = _rows(client, exp_id, exp_name)
    if not rows:
        raise RuntimeError("no benchmark runs found; run `mlbe benchmark` first")
    overall = _overall(rows)
    leaderboard = {
        "provenance": _provenance(exp_name, [r["run_id"] for r in rows]),
        "datasets": sorted({r["dataset"] for r in rows}),
        "rows": rows,
        "overall": overall,
        "ranking_weights": overall[0]["weights"] if overall else {},
    }

    stats = _stats(rows, folds, client, exp_name)
    pareto = _pareto(rows, exp_name)
    hpo = _hpo(client, exp_id, exp_name)
    env = env_info()
    env_payload = {
        "provenance": _provenance(exp_name, [r["run_id"] for r in rows[:1]]),
        "environment": env,
        "datasets": {
            name: load_dataset(name, cfg.datasets[name], seed=int(cfg.seed)).meta()
            for name in cfg.datasets
        },
        "cv": {
            "n_splits": int(cfg.cv.n_splits),
            "n_repeats": int(cfg.cv.n_repeats),
            "scheme": "RepeatedStratifiedKFold, identical splits for every model",
            "seed": int(cfg.seed),
        },
    }
    runs_payload = {
        "provenance": _provenance(exp_name, [f["run_id"] for f in folds]),
        "metric_keys": METRIC_KEYS,
        "runs": folds,
    }

    analysis = _load_local(artifacts / "analysis" / "analysis.json")
    rubric_report = _load_local(artifacts / "rubric" / "rubric_report.json")
    repro = _load_local(artifacts / "repro" / "repro.json")
    if analysis is None or rubric_report is None:
        raise RuntimeError("missing analysis/rubric artifacts; run `mlbe analyze` and `mlbe rubric` first")

    sources = list(analysis.get("source_run_ids", [])) + list(rubric_report.get("source_run_ids", []))
    error_payload = {
        "provenance": _provenance(exp_name, sources),
        **{k: analysis[k] for k in ("results", "generalization") if k in analysis},
        "findings_markdown": analysis.get("findings_markdown", ""),
    }
    generalization = {
        "provenance": _provenance(exp_name, sources),
        **analysis.get("generalization", {}),
    }
    rubric_dataset = {
        "provenance": _provenance(exp_name, sources),
        **rubric_report["datasets"][0],
        "datasets": rubric_report["datasets"],
        "summary": rubric_report["summary"],
        "config": rubric_report["config"],
    }
    rubric_model = {
        "provenance": _provenance(exp_name, sources),
        "models": rubric_report["models"],
        "summary": rubric_report["summary"],
        "config": rubric_report["config"],
    }
    validation = {
        "provenance": _provenance(exp_name, sources),
        **rubric_report["validation"],
        "label_noise_plot": [r for r in rubric_report["validation"]["rows"] if "label_noise" in r],
    }
    repro_pass = (repro or {}).get("pass") or {}
    repro_ids = [repro_pass.get("source_run_id")] if repro_pass.get("source_run_id") else []
    repro_payload = (
        {"provenance": _provenance(exp_name, repro_ids), **repro} if repro else {}
    )

    payloads = {
        "leaderboard.json": leaderboard,
        "stats.json": stats,
        "pareto.json": pareto,
        "hpo.json": hpo,
        "error_analysis.json": error_payload,
        "generalization.json": generalization,
        "rubric_dataset.json": rubric_dataset,
        "rubric_model.json": rubric_model,
        "rubric_validation.json": validation,
        "rubric_config.json": __import__("yaml").safe_load(
            (PROJECT_ROOT / "configs" / "rubrics.yaml").read_text(encoding="utf-8")
        ),
        "env.json": env_payload,
        "runs.json": runs_payload,
        "repro.json": repro_payload,
    }
    written = {}
    for name, payload in payloads.items():
        path = web / name
        dump_json(payload, path)
        written[name] = {"path": str(path), "bytes": path.stat().st_size}
    golden = _golden_vectors(cfg)
    golden_path = web.parent.parent / "tests" / "golden" / "rubric_golden.json"
    dump_json(golden, golden_path)
    written["golden"] = {"path": str(golden_path), "bytes": golden_path.stat().st_size}
    total = sum(v["bytes"] for v in written.values())
    return {"files": written, "total_bytes": total, "web_dir": str(web)}
