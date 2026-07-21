"""Tests for the feature-spec registry (Step 2 of the D023 generator-critic substrate).

Pure numpy + pytest, so they run in the local env. Exercises every seed family's
output shape / values, lowrank determinism, drift grouping on non-contiguous
request ordering, align edge cases, and build_features / default_seed_specs
validation. All synthetic arrays are constructed IN the test (unit test of the
transform, not a research result).
"""
import numpy as np
import pytest

from cas.autoresearch.features import build_features, default_seed_specs
from cas.autoresearch.types import FeatureSpec


def _meta(pairs):
    """meta_rows in the given (request_id, round_id) order."""
    return [{"request_id": r, "round_id": rd} for r, rd in pairs]


def test_raw_shape_and_values():
    n, d = 5, 4
    a = np.arange(n * d, dtype=np.float16).reshape(n, d)
    b = a + 100
    acts = {6: a, 12: b}
    meta = _meta([("a", i) for i in range(n)])
    X = build_features(FeatureSpec("raw2", "raw", (6, 12)), acts, meta)
    assert X.shape == (n, 2 * d)
    assert np.allclose(X[:, :d], a.astype(np.float32))
    assert np.allclose(X[:, d:], b.astype(np.float32))


def test_lowrank_shape_and_determinism():
    n, d, k = 6, 8, 3
    rng = np.random.default_rng(123)
    acts = {6: rng.standard_normal((n, d)).astype(np.float16)}
    meta = _meta([("a", i) for i in range(n)])
    s0 = FeatureSpec("lr", "lowrank", (6,), {"k": k, "seed": 0})
    X0 = build_features(s0, acts, meta)
    assert X0.shape == (n, k)
    assert np.array_equal(X0, build_features(s0, acts, meta))          # deterministic
    X1 = build_features(FeatureSpec("lr1", "lowrank", (6,), {"k": k, "seed": 1}),
                        acts, meta)
    assert not np.array_equal(X0, X1)                                  # seed matters


def test_lowrank_requires_valid_k():
    acts = {6: np.zeros((3, 4), dtype=np.float16)}
    meta = _meta([("a", i) for i in range(3)])
    with pytest.raises(ValueError):
        build_features(FeatureSpec("x", "lowrank", (6,)), acts, meta)      # no k
    with pytest.raises(ValueError):
        build_features(FeatureSpec("x", "lowrank", (6,), {"k": 0}), acts, meta)


def test_drift_missing_flag_and_diff_noncontiguous():
    d = 3
    # interleaved requests so artifact order is NOT per-request contiguous
    meta = _meta([("A", 0), ("B", 0), ("A", 1), ("B", 1)])
    h = np.array([[1, 1, 1],    # A r0 (idx0)
                  [5, 5, 5],    # B r0 (idx1)
                  [4, 4, 4],    # A r1 (idx2)
                  [9, 9, 9]],   # B r1 (idx3)
                 dtype=np.float16)
    X = build_features(FeatureSpec("d", "drift", (6,), {"order": 1}), {6: h}, meta)
    assert X.shape == (4, d + 1)                       # [diff(d), missing_flag]
    assert np.allclose(X[0, :d], 0) and X[0, d] == 1.0     # A r0: no predecessor
    assert np.allclose(X[1, :d], 0) and X[1, d] == 1.0     # B r0: no predecessor
    assert np.allclose(X[2, :d], [3, 3, 3]) and X[2, d] == 0.0   # A r1 - A r0
    assert np.allclose(X[3, :d], [4, 4, 4]) and X[3, d] == 0.0   # B r1 - B r0


def test_drift_order_two():
    meta = _meta([("A", 0), ("A", 1), ("A", 2)])
    h = np.array([[0, 0], [10, 10], [30, 30]], dtype=np.float16)
    X = build_features(FeatureSpec("d2", "drift", (6,), {"order": 2}), {6: h}, meta)
    assert X[0, -1] == 1.0 and X[1, -1] == 1.0          # first two imputed
    assert np.allclose(X[2, :2], [30, 30]) and X[2, -1] == 0.0   # r2 - r0


def test_align_cosine_aligned_and_zero_row():
    a = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=np.float16)
    acts = {0: a, 1: 2 * a}                              # b = 2a -> perfectly aligned
    meta = _meta([("a", i) for i in range(4)])
    X = build_features(FeatureSpec("al", "align", (0, 1)), acts, meta)
    assert X.shape == (4, 2)
    assert np.allclose(X[:3, 0], 1.0, atol=1e-3)        # aligned rows: cosine ~1
    assert X[3, 0] == 0.0                                # zero-norm row guarded
    assert not np.isnan(X).any()


def test_align_orthogonal():
    acts = {0: np.array([[1, 0]], dtype=np.float16),
            1: np.array([[0, 1]], dtype=np.float16)}
    X = build_features(FeatureSpec("al", "align", (0, 1)), acts, _meta([("a", 0)]))
    assert np.allclose(X[0, 0], 0.0, atol=1e-3)


def test_align_requires_two_layers():
    acts = {0: np.ones((2, 3), dtype=np.float16)}
    with pytest.raises(ValueError):
        build_features(FeatureSpec("al", "align", (0,)), acts,
                       _meta([("a", 0), ("a", 1)]))


def test_norm_values():
    h = np.array([[3, 4, 0]], dtype=np.float16)          # l2=5, mean=7/3
    X = build_features(FeatureSpec("nm", "norm", (6,)), {6: h}, _meta([("a", 0)]))
    assert X.shape == (1, 3)
    assert np.allclose(X[0, 0], 5.0, atol=1e-3)          # l2 norm
    assert np.allclose(X[0, 1], 7.0 / 3.0, atol=1e-3)    # mean
    assert np.allclose(X[0, 2], np.std([3.0, 4.0, 0.0]), atol=1e-3)  # std (ddof=0)


def test_missing_layer_raises():
    acts = {6: np.zeros((2, 4), dtype=np.float16)}
    with pytest.raises(ValueError):
        build_features(FeatureSpec("r", "raw", (6, 99)), acts,
                       _meta([("a", 0), ("a", 1)]))


def test_row_misalignment_raises():
    acts = {6: np.zeros((3, 4), dtype=np.float16)}       # 3 act rows
    with pytest.raises(ValueError):
        build_features(FeatureSpec("r", "raw", (6,)), acts,
                       _meta([("a", 0), ("a", 1)]))       # but 2 meta rows


def test_default_seed_specs_spans_families():
    specs = default_seed_specs((6, 12, 18, 24))
    assert {s.family for s in specs} == {"raw", "lowrank", "drift", "norm", "align"}
    names = [s.name for s in specs]
    assert len(names) == len(set(names))                 # stable unique keys
    assert "align" not in {s.family for s in default_seed_specs((6,))}  # needs 2 layers
    with pytest.raises(ValueError):
        default_seed_specs(())
