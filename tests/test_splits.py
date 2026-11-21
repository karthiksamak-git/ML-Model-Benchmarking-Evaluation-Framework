from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlbe.data.registry import load_dataset
from mlbe.data.splits import (
    load_splits,
    make_splits,
    materialize,
    save_splits,
    split_checksum,
    validate_splits,
)


def _data():
    ds = load_dataset("breast_cancer", {"kind": "sklearn", "loader": "load_breast_cancer"})
    return ds.X, ds.y


def test_splits_are_identical_for_every_model():
    X, y = _data()
    a = make_splits(y, 3, 2, seed=7, dataset="bc")
    b = make_splits(y, 3, 2, seed=7, dataset="bc")
    assert split_checksum(a) == split_checksum(b)
    for sa, sb in zip(a, b, strict=True):
        assert sa.split_id == sb.split_id
        assert np.array_equal(sa.train_idx, sb.train_idx)
        assert np.array_equal(sa.test_idx, sb.test_idx)


def test_no_overlap_and_full_coverage_and_stratification():
    _, y = _data()
    splits = make_splits(y, 4, 2, seed=11)
    validate_splits(splits, y)  # raises on any violation
    for s in splits:
        assert np.intersect1d(s.train_idx, s.test_idx).size == 0
        assert len(np.union1d(s.train_idx, s.test_idx)) == len(y)
        for c in np.unique(y):
            assert np.any(y[s.train_idx] == c)
            assert np.any(y[s.test_idx] == c)


def test_splits_roundtrip(tmp_path):
    _, y = _data()
    splits = make_splits(y, 3, 1, seed=3, dataset="bc")
    path = save_splits(splits, tmp_path / "splits.json")
    loaded = load_splits(path)
    assert [s.split_id for s in loaded] == [s.split_id for s in splits]
    assert np.array_equal(loaded[0].test_idx, splits[0].test_idx)


def test_preprocessing_is_fitted_only_on_training_fold():
    X, y = _data()
    split = make_splits(y, 3, 1, seed=5)[0]
    X_train, X_test = X[split.train_idx], X[split.test_idx]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y[split.train_idx])
    scaler = pipe.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))
    # test features are transformed with training statistics
    assert np.allclose(scaler.transform(X_test)[0], (X_test[0] - X_train.mean(axis=0)) / X_train.std(axis=0))


def test_materialized_folds_match_indices():
    X, y = _data()
    splits = make_splits(y, 3, 1, seed=1)[:1]
    fold = materialize(splits, X, y)[0]
    assert np.array_equal(fold.X_train, X[splits[0].train_idx])
    assert np.array_equal(fold.y_test, y[splits[0].test_idx])
