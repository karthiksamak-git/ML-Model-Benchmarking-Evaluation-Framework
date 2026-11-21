from __future__ import annotations

import numpy as np

from mlbe.data.corruptions import (
    DEFECT_SPECS,
    add_missing_values,
    inject_covariate_shift,
    inject_duplicates,
    inject_label_noise,
    inject_leaky_feature,
    inject_outliers,
    make_planted_dataset,
)


def _toy(n=100, d=5, classes=3, seed=0):
    gen = np.random.default_rng(seed)
    X = gen.normal(size=(n, d))
    y = gen.integers(0, classes, size=n)
    return X, y


def test_label_noise_manifest_matches_injected_labels():
    X, y = _toy()
    Xc, yc, manifest = inject_label_noise(X, y, rate=0.20, seed=5)
    assert np.array_equal(Xc, X)
    idx = np.asarray(manifest["indices"])
    assert manifest["n_affected"] == len(idx) == int(round(0.20 * len(y)))
    assert np.array_equal(yc[idx], np.asarray(manifest["new_labels"]))
    assert np.array_equal(y[idx], np.asarray(manifest["original_labels"]))
    assert np.all(yc[idx] != y[idx])
    untouched = np.setdiff1d(np.arange(len(y)), idx)
    assert np.array_equal(yc[untouched], y[untouched])


def test_duplicates_manifest_matches_appended_rows():
    X, y = _toy()
    Xc, yc, manifest = inject_duplicates(X, y, rate=0.1, seed=2, conflict_rate=0.5)
    src = np.asarray(manifest["source_indices"])
    assert len(Xc) == len(X) + manifest["n_duplicated"]
    assert np.array_equal(Xc[len(X) :], X[src])
    assert manifest["n_duplicated"] == int(round(0.1 * len(X)))
    conflicts = manifest["conflicting_new_row_indices"]
    assert conflicts and all(yc[i] != y[i - len(X)] for i in conflicts)


def test_missing_values_manifest_counts_cells():
    X, _ = _toy()
    Xc, manifest = add_missing_values(X, rate=0.15, seed=1)
    assert int(np.isnan(Xc).sum()) == manifest["n_missing_cells"]
    assert manifest["n_missing_cells"] == int(round(0.15 * X.size))


def test_outliers_only_change_manifest_coordinates():
    X, y = _toy()
    Xc, yc, manifest = inject_outliers(X, y, rate=0.1, seed=4)
    idx = np.asarray(manifest["indices"])
    cols = np.asarray(manifest["columns"])
    mask = np.ones_like(X, dtype=bool)
    mask[idx, cols] = False
    assert np.array_equal(Xc[mask], X[mask])
    assert np.array_equal(yc, y)
    assert np.all(np.abs(Xc[idx, cols] - X[idx, cols]) > 1.0)


def test_covariate_shift_touches_only_test_features():
    X_tr, _ = _toy()
    X_te, _ = _toy(seed=9)
    shifted, manifest = inject_covariate_shift(X_tr, X_te, shift=2.0, seed=3)
    cols = manifest["columns"]
    changed = np.zeros(X_te.shape[1], dtype=bool)
    changed[cols] = True
    assert np.array_equal(shifted[:, ~changed], X_te[:, ~changed])
    assert not np.allclose(shifted[:, changed], X_te[:, changed])


def test_leaky_feature_is_almost_target():
    X, y = _toy()
    Xc, manifest = inject_leaky_feature(X, y, seed=0, noise=0.0)
    assert Xc.shape[1] == X.shape[1] + 1
    assert np.array_equal(Xc[:, -1], y.astype(float))
    assert manifest["feature_index"] == X.shape[1]


def test_injection_is_deterministic():
    X, y = _toy()
    a = inject_label_noise(X, y, 0.2, seed=8)
    b = inject_label_noise(X, y, 0.2, seed=8)
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_planted_datasets_cover_every_spec():
    for name in DEFECT_SPECS:
        data = make_planted_dataset(name)
        assert data.X_train.shape[0] == len(data.y_train)
        assert data.X_train.shape[1] == len(data.feature_names)
        if name != "clean":
            assert data.manifest["defects"]
