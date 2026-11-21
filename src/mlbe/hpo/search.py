"""Optuna hyperparameter search with nested MLflow runs.

Protocol (documented in README): trials are scored with k-fold CV *inside* the
training split; the test split is used once for the final tuned-vs-default
comparison, so the reported improvement is not selection-biased.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..data.registry import Dataset, load_dataset
from ..env_info import env_info
from ..models.registry import DEFAULT_PARAMS, build_model
from ..seed import seed_everything
from ..stats.compare import bootstrap_ci, paired_bootstrap_difference
from ..tracking import mlflow_utils as mu

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:  # pragma: no cover
            raise ValueError(f"unknown space type {kind!r}")
    return params


def _cv_score(
    dataset: Dataset,
    model_name: str,
    params: dict[str, Any],
    n_folds: int,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[float]]:
    y = dataset.y
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    image_shape = dataset.image_shape if dataset.is_image else None
    for step, (tr, te) in enumerate(skf.split(dataset.X, y)):
        seed_everything(seed + step)
        model = build_model(
            model_name,
            seed=seed + step,
            params=params,
            n_classes=len(np.unique(y)),
            input_dim=dataset.n_features,
            image_shape=image_shape,
        )
        model.fit(dataset.X[tr], y[tr])
        pred = model.predict(dataset.X[te])
        score = float(np.mean(pred == y[te]))
        scores.append(score)
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), scores


def run_study(
    model_name: str,
    dataset_name: str,
    cfg: Any,
    hpo_cfg: Any,
    quick: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    seed = int(seed if seed is not None else cfg.seed)
    n_trials = int(hpo_cfg.quick.n_trials if quick else hpo_cfg.n_trials)
    n_folds = int(hpo_cfg.cv_folds)
    space = hpo_cfg.spaces[model_name].to_dict()
    dataset = load_dataset(dataset_name, cfg.datasets[dataset_name], seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.2, random_state=seed, stratify=dataset.y
    )
    train_ds = Dataset(
        name=dataset.name + "_train",
        X=X_train,
        y=y_train,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        image_shape=dataset.image_shape,
    )
    started = time.time()
    mu.init_tracking(cfg)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(hpo_cfg.pruner.n_startup_trials),
        n_warmup_steps=int(hpo_cfg.pruner.n_warmup_steps),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    trial_records: list[dict[str, Any]] = []

    with mu.run(
        f"hpo-{model_name}-{dataset_name}",
        tags={"kind": "hpo_study", "model": model_name, "dataset": dataset_name, "protocol": "cv-then-heldout"},
    ) as study_run:
        mu.log_env_tags(env_info())
        mu.log_params_safe({"seed": seed, "n_trials": n_trials, "cv_folds": n_folds, "model": model_name})

        def objective(trial: optuna.Trial) -> float:
            params = _suggest(trial, space)
            with mu.run(
                f"trial-{trial.number}",
                tags={"kind": "hpo_trial", "trial_number": trial.number, "model": model_name},
                nested=True,
            ):
                mu.log_params_safe(params)
                try:
                    score, folds = _cv_score(train_ds, model_name, params, n_folds, seed, trial)
                except optuna.TrialPruned:
                    mu.log_metrics({"cv_accuracy_mean": 0.0, "pruned": 1.0})
                    trial_records.append(
                        {"trial": trial.number, "state": "pruned", "params": params, "value": None}
                    )
                    raise
                mu.log_metrics(
                    {
                        "cv_accuracy_mean": score,
                        "cv_accuracy_std": float(np.std(folds)),
                        **{f"fold_{i}_accuracy": v for i, v in enumerate(folds)},
                    }
                )
                mu.log_metrics({"train_seconds": time.time() - started})
                trial_records.append(
                    {"trial": trial.number, "state": "complete", "params": params, "value": score}
                )
                return score

        study.optimize(objective, n_trials=n_trials)

        best_params = {**DEFAULT_PARAMS.get(model_name, {}), **study.best_params}
        # final unbiased comparison on the untouched test split
        def fit_eval(params_to_use: dict[str, Any], tag: str) -> tuple[float, Any, np.ndarray]:
            seed_everything(seed)
            model = build_model(
                model_name,
                seed=seed,
                params=params_to_use,
                n_classes=len(np.unique(y_train)),
                input_dim=dataset.n_features,
                image_shape=dataset.image_shape if dataset.is_image else None,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = float(np.mean(pred == y_test))
            with mu.run(f"final-{tag}", tags={"kind": "hpo_final", "variant": tag, "model": model_name}, nested=True):
                mu.log_params_safe(params_to_use)
                mu.log_metrics({"test_accuracy": acc})
                if tag == "tuned":
                    mu.log_model_with_signature(model, X_train[:5])

                    mu.log_json_artifact(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "best_params": best_params,
                            "test_accuracy": acc,
                            "protocol": "CV on 80% train, single evaluation on 20% held-out",
                            "seed": seed,
                        },
                        "best_params.json",
                    )
            return acc, model, pred

        tuned_acc, tuned_model, tuned_pred = fit_eval(best_params, "tuned")
        default_acc, _, default_pred = fit_eval(DEFAULT_PARAMS.get(model_name, {}), "default")
        # bootstrap over per-fold CV scores of the best configuration
        best_fold_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for step, (tr, te) in enumerate(skf.split(X_train, y_train)):
            seed_everything(seed + step)
            mdl = build_model(model_name, seed=seed + step, params=best_params,
                              n_classes=len(np.unique(y_train)), input_dim=dataset.n_features,
                              image_shape=dataset.image_shape if dataset.is_image else None)
            mdl.fit(X_train[tr], y_train[tr])
            best_fold_scores.append(float(np.mean(mdl.predict(X_train[te]) == y_train[te])))
        tuned_ci = bootstrap_ci(best_fold_scores, seed=seed)

        default_fold_scores = []
        for step, (tr, te) in enumerate(skf.split(X_train, y_train)):
            seed_everything(seed + step)
            mdl = build_model(model_name, seed=seed + step, params=DEFAULT_PARAMS.get(model_name, {}),
                              n_classes=len(np.unique(y_train)), input_dim=dataset.n_features,
                              image_shape=dataset.image_shape if dataset.is_image else None)
            mdl.fit(X_train[tr], y_train[tr])
            default_fold_scores.append(float(np.mean(mdl.predict(X_train[te]) == y_train[te])))
        default_ci = bootstrap_ci(default_fold_scores, seed=seed)
        paired = paired_bootstrap_difference(best_fold_scores, default_fold_scores, seed=seed)

        # McNemar on held-out predictions
        from ..stats.compare import mcnemar

        mcn = mcnemar(y_test, tuned_pred, default_pred)

        importances = {}
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:  # pragma: no cover - needs >=2 completed trials
            importances = {}
        history = [
            {
                "trial": t.number,
                "state": str(t.state.name),
                "value": float(t.value) if t.value is not None else None,
                "params": t.params,
            }
            for t in study.trials
        ]
        summary = {
            "model": model_name,
            "dataset": dataset_name,
            "study_run_id": study_run.info.run_id,
            "seed": seed,
            "n_trials": n_trials,
            "n_completed": sum(1 for t in study.trials if t.value is not None),
            "best_params": best_params,
            "best_cv_accuracy": float(study.best_value),
            "best_cv_ci": list(tuned_ci),
            "default_cv_ci": list(default_ci),
            "tuned_test_accuracy": tuned_acc,
            "default_test_accuracy": default_acc,
            "test_improvement": tuned_acc - default_acc,
            "paired_bootstrap_ci": [paired["ci_low"], paired["ci_high"]],
            "mcnemar": mcn,
            "importance": {k: float(v) for k, v in importances.items()},
            "history": history,
            "elapsed_s": time.time() - started,
            "protocol": "Trials use 3-fold CV on the 80% training split; the 20% test split is evaluated once.",
        }
        mu.log_metrics(
            {
                "best_cv_accuracy": summary["best_cv_accuracy"],
                "tuned_test_accuracy": tuned_acc,
                "default_test_accuracy": default_acc,
                "test_improvement": summary["test_improvement"],
                "elapsed_s": summary["elapsed_s"],
            }
        )
        mu.log_json_artifact(summary, "hpo_summary.json")
    return summary


def run_hpo(
    cfg: Any,
    hpo_cfg: Any,
    quick: bool = False,
    models: list[str] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    chosen = models or (list(hpo_cfg.quick.hpo_models) if quick else list(hpo_cfg.study_models))
    if dataset is None:
        dataset = str(hpo_cfg.quick.dataset) if quick else "breast_cancer"
    results = []
    for model in chosen:
        if model not in hpo_cfg.spaces._data:
            continue
        results.append(run_study(model, dataset, cfg, hpo_cfg, quick=quick))
    return {"studies": results, "dataset": dataset}
