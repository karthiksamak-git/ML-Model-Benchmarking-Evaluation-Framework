"""Hand-written dataset-quality rubric checks.

Thresholds and weights are read from ``configs/rubrics.yaml``. Each check
returns {id, name, category, score, severity, evidence, recommendation}; the
``evidence`` dict carries the raw numbers used for the score so every verdict is
auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..metrics import align_proba
from ..seed import seed_everything
from .scoring import build_report, make_check, score_higher_better, score_lower_better


@dataclass
class DatasetView:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    seed: int = 0

    @property
    def labels(self) -> np.ndarray:
        return np.unique(self.y_train)


def _finite_matrix(X: np.ndarray) -> np.ndarray:
    """Median-impute non-finite cells column-wise (avoids NaN -> 0 artefacts)."""
    arr = np.asarray(X, dtype=np.float64).copy()
    if not np.isnan(arr).any():
        return arr
    with np.errstate(all="ignore"):
        medians = np.nanmedian(arr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows, cols = np.where(~np.isfinite(arr))
    arr[rows, cols] = medians[cols]
    return arr


def _names(view: DatasetView, n_cols: int) -> list[str]:
    if len(view.feature_names) == n_cols:
        return view.feature_names
    return [f"x{i}" for i in range(n_cols)]


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_missing_values(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X_all = np.vstack([view.X_train, view.X_test])
    per_col = np.isnan(X_all).mean(axis=0)
    overall = float(np.isnan(X_all).mean())
    affected_rows = int(np.isnan(view.X_train).any(axis=1).sum())
    names = _names(view, X_all.shape[1])
    top = sorted(
        ({"column": names[i], "missing_rate": round(float(r), 5)} for i, r in enumerate(per_col) if r > 0),
        key=lambda d: -d["missing_rate"],
    )[:8]
    score = score_lower_better(
        overall, ok=0.0, warn=float(t.missing_rate_warn), fail=float(t.missing_rate_fail)
    )
    return make_check(
        "missing_values",
        "Missing values",
        "dataset_quality",
        score,
        {
            "overall_missing_rate": round(overall, 5),
            "columns_with_missing": int((per_col > 0).sum()),
            "n_columns": int(X_all.shape[1]),
            "affected_training_rows": affected_rows,
            "top_columns": top,
        },
        "Impute or drop missing cells; if missingness is informative, encode it explicitly.",
        cfg,
    )


def check_constant_columns(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = view.X_train
    names = _names(view, X.shape[1])
    constant, near = [], []
    for i in range(X.shape[1]):
        col = X[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            constant.append(names[i])
            continue
        values, counts = np.unique(col, return_counts=True)
        if values.size <= int(t.constant_unique_values):
            constant.append(names[i])
        elif counts.max() / col.size >= 1.0 - float(t.near_constant_share):
            near.append(names[i])
    share = (len(constant) + len(near)) / max(1, X.shape[1])
    score = score_lower_better(share, ok=0.0, warn=0.05, fail=0.20)
    return make_check(
        "constant_columns",
        "Constant / near-constant columns",
        "dataset_quality",
        score,
        {
            "n_constant": len(constant),
            "n_near_constant": len(near),
            "share": round(float(share), 4),
            "constant_columns": constant[:10],
            "near_constant_columns": near[:10],
        },
        "Drop constant columns; they carry no signal and inflate dimensionality.",
        cfg,
    )


def _row_hashes(X: np.ndarray) -> np.ndarray:
    return np.asarray([hash(row.tobytes()) for row in np.ascontiguousarray(X)], dtype=np.int64)


def check_duplicates(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = view.X_train
    y = view.y_train
    hashes = _row_hashes(X)
    unique, counts = np.unique(hashes, return_counts=True)
    dup_rows = int((counts - 1).sum())
    rate = dup_rows / max(1, len(X))
    # conflicting labels among identical feature rows
    conflicts = 0
    conflict_pairs: list[dict[str, Any]] = []
    for h in unique[counts > 1][:200]:
        idx = np.flatnonzero(hashes == h)
        labels = np.unique(y[idx])
        if labels.size > 1:
            conflicts += int(len(idx) - 1)
            if len(conflict_pairs) < 5:
                conflict_pairs.append(
                    {"indices": [int(i) for i in idx[:5]], "labels": [int(v) for v in labels]}
                )
    score = score_lower_better(rate, ok=0.0, warn=float(t.duplicate_rate_warn), fail=float(t.duplicate_rate_fail))
    return make_check(
        "duplicates",
        "Exact duplicate rows",
        "dataset_quality",
        score,
        {
            "n_rows": int(len(X)),
            "n_unique_rows": int(len(unique)),
            "n_duplicate_rows": dup_rows,
            "duplicate_rate": round(float(rate), 5),
            "n_conflicting_duplicates": conflicts,
            "examples": conflict_pairs,
        },
        "De-duplicate before splitting, otherwise identical rows leak across folds.",
        cfg,
    )


def check_conflicting_labels(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X, y = view.X_train, view.y_train
    hashes = _row_hashes(X)
    conflicts = 0
    groups = 0
    for h in np.unique(hashes):
        idx = np.flatnonzero(hashes == h)
        if len(idx) < 2:
            continue
        if np.unique(y[idx]).size > 1:
            groups += 1
            conflicts += int(len(idx) - 1)
    rate = conflicts / max(1, len(X))
    score = score_lower_better(
        rate, ok=0.0, warn=float(t.conflict_rate_warn), fail=float(t.conflict_rate_fail)
    )
    return make_check(
        "conflicting_labels",
        "Duplicate rows with conflicting labels",
        "dataset_quality",
        score,
        {
            "n_conflicting_groups": groups,
            "n_conflicting_rows": conflicts,
            "rate": round(float(rate), 5),
        },
        "Resolve conflicting labels (relabel or drop); they cap achievable accuracy.",
        cfg,
    )


def check_class_imbalance(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    counts = np.bincount(view.y_train.astype(int))
    nonzero = counts[counts > 0]
    ratio = float(nonzero.max() / max(1, nonzero.min()))
    minority_share = float(nonzero.min() / max(1, nonzero.sum()))
    score = min(
        score_lower_better(ratio, ok=1.0, warn=float(t.imbalance_ratio_warn), fail=float(t.imbalance_ratio_fail)),
        score_higher_better(minority_share, ok=0.25, warn=float(t.minority_share_warn), fail=float(t.minority_share_fail)),
    )
    return make_check(
        "class_imbalance",
        "Class imbalance",
        "dataset_quality",
        score,
        {
            "class_counts": {int(i): int(c) for i, c in enumerate(counts) if c > 0},
            "imbalance_ratio": round(ratio, 3),
            "minority_share": round(minority_share, 4),
        },
        "Use stratified splits and class weights; report balanced accuracy and per-class recall.",
        cfg,
    )


def oof_probabilities(view: DatasetView, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities used for label-noise detection.

    HistGradientBoosting is used because it is nonlinear, fast on tabular data
    and handles missing values natively (so the detector also works on the
    planted missing-value dataset).
    """
    seed = view.seed if seed is None else seed
    labels = view.labels
    counts = np.bincount(view.y_train.astype(int))
    n_splits = int(min(3, counts[counts > 0].min()))
    n_splits = max(2, n_splits)
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=100, learning_rate=0.1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, view.X_train, view.y_train, cv=skf, method="predict_proba")
    return align_proba(proba, labels, labels), labels


def label_noise_flags(view: DatasetView, cfg: Any, proba: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Confident-learning-style flags.

    Per predicted class, the confidence threshold is the mean predicted
    probability of samples assigned to that class. A sample is flagged when the
    predicted class disagrees with the given label and the prediction is at
    least that confident.
    """
    if proba is None:
        proba, _ = oof_probabilities(view)
    y = view.y_train
    labels = view.labels
    pred_idx = proba.argmax(axis=1)
    pred = labels[pred_idx]
    conf = proba.max(axis=1)
    thresholds = {}
    for j, c in enumerate(labels):
        mask = pred_idx == j
        thresholds[str(int(c))] = float(conf[mask].mean()) if mask.any() else 0.5
    flagged = np.zeros(len(y), dtype=bool)
    for j, c in enumerate(labels):
        t = thresholds[str(int(c))]
        flagged |= (pred_idx == j) & (pred != y) & (conf >= t)
    info = {
        "per_class_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "oof_accuracy": float(np.mean(pred == y)),
        "n_flagged": int(flagged.sum()),
        "flagged_rate": float(flagged.mean()),
    }
    return flagged, info


def check_label_noise(view: DatasetView, cfg: Any, proba: np.ndarray | None = None) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    flagged, info = label_noise_flags(view, cfg, proba=proba)
    rate = float(flagged.mean())
    score = score_lower_better(
        rate, ok=0.0, warn=float(t.label_noise_warn), fail=float(t.label_noise_fail)
    )
    return make_check(
        "label_noise",
        "Suspected label noise",
        "dataset_quality",
        score,
        {
            "estimated_noise_rate": round(rate, 4),
            "tolerance": float(t.label_noise_rate_tolerance),
            "flagged_indices": [int(i) for i in np.flatnonzero(flagged)],
            **info,
        },
        "Audit the flagged samples; relabel or remove confidently mislabelled points.",
        cfg,
    )


def check_leakage(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X_tr, X_te = view.X_train, view.X_test
    hashes_tr, hashes_te = _row_hashes(X_tr), _row_hashes(X_te)
    exact_overlap = int(np.isin(hashes_te, hashes_tr).sum())
    exact_rate = exact_overlap / max(1, len(X_te))

    scaler = StandardScaler().fit(_finite_matrix(X_tr))
    A = scaler.transform(_finite_matrix(X_tr))
    B = scaler.transform(_finite_matrix(X_te))
    nn = NearestNeighbors(n_neighbors=1).fit(A)
    dist, _ = nn.kneighbors(B)
    near_overlap = int((dist[:, 0] <= float(t.near_duplicate_distance)).sum())

    # single-feature predictive power (max one-vs-rest AUC)
    best_feature, best_auc = -1, 0.0
    Xf = _finite_matrix(X_tr)
    for j in range(Xf.shape[1]):
        col = Xf[:, j]
        if np.nanstd(col) == 0:
            continue
        try:
            if len(view.labels) == 2:
                auc = roc_auc_score(view.y_train, col)
                auc = max(auc, 1.0 - auc)
            else:
                auc = max(
                    roc_auc_score((view.y_train == c).astype(int), col) for c in view.labels
                )
                auc = max(auc, 1.0 - auc)
        except ValueError:
            continue
        if auc > best_auc:
            best_auc, best_feature = float(auc), j

    score = min(
        score_lower_better(exact_rate, ok=0.0, warn=float(t.leakage_overlap_warn), fail=float(t.leakage_overlap_fail)),
        score_lower_better(
            near_overlap / max(1, len(X_te)), ok=0.0, warn=float(t.leakage_overlap_warn), fail=float(t.leakage_overlap_fail)
        ),
        score_lower_better(
            best_auc,
            ok=float(t.leakage_single_feature_accuracy),
            warn=float(t.leakage_single_feature_accuracy) + 0.01,
            fail=1.0,
        ),
    )
    names = _names(view, X_tr.shape[1])
    return make_check(
        "leakage",
        "Train/test leakage",
        "dataset_quality",
        score,
        {
            "exact_overlap_rows": exact_overlap,
            "exact_overlap_rate": round(float(exact_rate), 5),
            "near_duplicate_rows": near_overlap,
            "best_single_feature": names[best_feature] if best_feature >= 0 else None,
            "best_single_feature_auc": round(float(best_auc), 4),
        },
        "Re-split after de-duplication and remove target-derived features.",
        cfg,
    )


def check_outliers(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = _finite_matrix(view.X_train)
    n, d = X.shape
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    factor = float(t.outlier_iqr_factor)
    with np.errstate(invalid="ignore"):
        iqr_mask = (X < q1 - factor * iqr) | (X > q3 + factor * iqr)
    iqr_flag = iqr_mask.any(axis=1)

    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1.0
    z = np.abs(X - median) / (1.4826 * mad)
    robust_flag = (z > float(t.outlier_robust_z)).any(axis=1)

    iso = IsolationForest(random_state=view.seed, n_estimators=100, contamination="auto")
    iso_flag = iso.fit_predict(X) == -1

    votes = iqr_flag.astype(int) + robust_flag.astype(int) + iso_flag.astype(int)
    flagged = votes >= 2
    rate = float(flagged.mean())
    score = score_lower_better(
        rate, ok=0.0, warn=float(t.outlier_rate_warn), fail=float(t.outlier_rate_fail)
    )
    return make_check(
        "outliers",
        "Outliers (IQR / robust z / Isolation Forest)",
        "dataset_quality",
        score,
        {
            "outlier_rate": round(rate, 4),
            "n_flagged": int(flagged.sum()),
            "iqr_rate": round(float(iqr_flag.mean()), 4),
            "robust_z_rate": round(float(robust_flag.mean()), 4),
            "isolation_forest_rate": round(float(iso_flag.mean()), 4),
            "agreement_rule": "flagged when >= 2 of 3 methods agree",
        },
        "Inspect flagged rows; winsorise or use models robust to heavy tails.",
        cfg,
    )


def check_covariate_shift(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = np.vstack([view.X_train, view.X_test]).astype(np.float64)
    domain = np.concatenate([np.zeros(len(view.X_train)), np.ones(len(view.X_test))])
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=view.seed)),
        ]
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=view.seed)
    proba = cross_val_predict(pipe, X, domain, cv=skf, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(domain, proba))
    auc = max(auc, 1.0 - auc)  # discriminability is direction-agnostic
    score = score_lower_better(auc, ok=0.5, warn=float(t.adversarial_auc_warn), fail=float(t.adversarial_auc_fail))
    return make_check(
        "covariate_shift",
        "Covariate shift (adversarial validation)",
        "dataset_quality",
        score,
        {"adversarial_auc": round(auc, 4), "n_train": int(len(view.X_train)), "n_test": int(len(view.X_test))},
        "Check collection or time period; consider importance weighting or domain adaptation.",
        cfg,
    )


def check_feature_redundancy(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = _finite_matrix(view.X_train)
    if X.shape[1] < 2:
        corr = np.eye(1)
    else:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu_indices_from(corr, k=1)
    pairs = np.abs(corr[upper])
    share = float((pairs > float(t.high_correlation)).mean()) if pairs.size else 0.0
    score = score_lower_better(share, ok=0.0, warn=float(t.high_correlation_share_warn), fail=0.5)
    names = _names(view, X.shape[1])
    top_pairs = []
    if pairs.size:
        order = np.argsort(-pairs)[:5]
        for k in order:
            i, j = upper[0][k], upper[1][k]
            top_pairs.append(
                {"a": names[i], "b": names[j], "correlation": round(float(corr[i, j]), 4)}
            )
    return make_check(
        "feature_redundancy",
        "Feature redundancy",
        "dataset_quality",
        score,
        {
            "share_pairs_above_threshold": round(share, 4),
            "threshold": float(t.high_correlation),
            "top_correlated_pairs": top_pairs,
        },
        "Remove or combine highly correlated features to reduce variance and cost.",
        cfg,
    )


def check_low_information(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    X = _finite_matrix(view.X_train)
    y = view.y_train
    scores = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            scores.append(0.0)
            continue
        if len(view.labels) == 2:
            r = abs(np.corrcoef(col, y)[0, 1])
        else:
            r = max(abs(np.corrcoef(col, (y == c).astype(float))[0, 1]) for c in view.labels)
        scores.append(float(0.0 if np.isnan(r) else r))
    scores_arr = np.asarray(scores)
    share = float((scores_arr < float(t.low_info_abs_corr)).mean())
    score = score_lower_better(share, ok=0.1, warn=float(t.low_info_share_warn), fail=0.8)
    names = _names(view, X.shape[1])
    low = [names[i] for i in np.flatnonzero(scores_arr < float(t.low_info_abs_corr))][:15]
    return make_check(
        "low_information",
        "Low-information features",
        "dataset_quality",
        score,
        {
            "share_below_threshold": round(share, 4),
            "threshold_abs_correlation": float(t.low_info_abs_corr),
            "low_information_features": low,
        },
        "Confirm with multivariate importance before dropping; univariate filters miss interactions.",
        cfg,
    )


def check_sample_adequacy(view: DatasetView, cfg: Any) -> dict[str, Any]:
    t = cfg.dataset_quality.thresholds
    n, d = view.X_train.shape
    per_feature = n / max(1, d)
    counts = np.bincount(view.y_train.astype(int))
    min_class = int(counts[counts > 0].min())
    score = min(
        score_higher_better(
            per_feature,
            ok=float(t.min_samples_per_feature),
            warn=float(t.min_samples_per_feature) / 2,
            fail=2.0,
        ),
        score_higher_better(
            min_class,
            ok=float(t.min_samples_per_class),
            warn=float(t.min_samples_per_class) / 2,
            fail=5.0,
        ),
    )
    return make_check(
        "sample_adequacy",
        "Sample-size adequacy",
        "dataset_quality",
        score,
        {
            "n_train": int(n),
            "n_features": int(d),
            "samples_per_feature": round(float(per_feature), 2),
            "min_samples_per_class": min_class,
        },
        "Collect more data (especially minority classes) or reduce dimensionality/regularise.",
        cfg,
    )


CHECKS = [
    check_missing_values,
    check_constant_columns,
    check_duplicates,
    check_conflicting_labels,
    check_class_imbalance,
    check_label_noise,
    check_leakage,
    check_outliers,
    check_covariate_shift,
    check_feature_redundancy,
    check_low_information,
    check_sample_adequacy,
]


def evaluate_dataset_quality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Any,
    feature_names: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    seed_everything(seed)
    view = DatasetView(
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        X_test=np.asarray(X_test),
        y_test=np.asarray(y_test),
        feature_names=list(feature_names or []),
        seed=seed,
    )
    proba, _ = oof_probabilities(view)
    checks = []
    for fn in CHECKS:
        if fn is check_label_noise:
            checks.append(fn(view, cfg, proba=proba))
        else:
            checks.append(fn(view, cfg))
    return build_report(
        checks,
        cfg.dataset_quality.weights.to_dict(),
        "dataset_quality",
        cfg,
    )


def evaluate_corrupted(data: Any, cfg: Any, seed: int = 0) -> dict[str, Any]:
    report = evaluate_dataset_quality(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        cfg,
        feature_names=data.feature_names,
        seed=seed,
    )
    report["dataset"] = data.name
    report["defects"] = data.manifest.get("defects", [])
    flags, _ = label_noise_flags(
        DatasetView(
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            feature_names=data.feature_names,
            seed=seed,
        ),
        cfg,
    )
    report["label_noise_flags"] = [int(i) for i in np.flatnonzero(flags)]
    return report
