"""Unit tests for the deployed-cost probe (I23/G3, D023).

These tests exercise the numpy timing path and the verdict logic; they assert
only structural / ordering properties of measured timings (never a specific
timing value -- AGENTS.md forbids fabricated numbers). The torch path is
importorskip-guarded so the suite passes without torch / CUDA.
"""
import numpy as np
import pytest

from cas.autoresearch.cost import bench_feature, bench_feature_torch, cost_verdict


def _fixed_matmul(d_in: int = 64, k: int = 16):
    """A trivial fixed-projection transform, the canonical cheap pre-round op."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d_in, k)).astype(np.float32)
    return lambda x: x @ W


def test_bench_feature_returns_sane_summary():
    d_in = 64
    transform = _fixed_matmul(d_in=d_in)
    sample = np.random.default_rng(1).standard_normal((d_in,)).astype(np.float32)

    b = bench_feature(transform, sample, repeats=50, warmup=5)

    assert b["backend"] == "numpy"
    assert b["repeats"] == 50
    # every reported duration is positive and finite
    for key in ("mean_ns", "median_ns", "p90_ns", "min_ns"):
        assert np.isfinite(b[key])
        assert b[key] > 0
    # sane ordering: p90 >= median >= min (percentile is nearest-rank)
    assert b["p90_ns"] >= b["median_ns"] >= b["min_ns"]
    # mean is at least the minimum observed sample (always true)
    assert b["mean_ns"] >= b["min_ns"]


def test_bench_feature_batched_sample():
    d_in = 32
    transform = _fixed_matmul(d_in=d_in, k=8)
    sample = np.random.default_rng(2).standard_normal((4, d_in)).astype(np.float32)
    b = bench_feature(transform, sample, repeats=20, warmup=2)
    assert b["p90_ns"] >= b["median_ns"] >= b["min_ns"] > 0


def test_bench_feature_rejects_bad_args():
    transform = _fixed_matmul()
    sample = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError):
        bench_feature(transform, sample, repeats=0)
    with pytest.raises(ValueError):
        bench_feature(transform, sample, warmup=-1)


def test_cost_verdict_flags_near_zero_true():
    # median far below 5% of the (caller-supplied) reference -> near-zero
    bench = {"median_ns": 1_000.0}          # 1 us
    reference_ns = 24_000_000.0             # 24 ms, a MEASURED cost passed in
    v = cost_verdict(bench, reference_ns=reference_ns)
    assert v["is_near_zero"] is True
    assert v["ratio"] == pytest.approx(1_000.0 / 24_000_000.0)
    assert v["reference_ns"] == reference_ns


def test_cost_verdict_flags_near_zero_false():
    # median above 5% of the reference -> NOT near-zero (fails its premise)
    bench = {"median_ns": 5_000_000.0}      # 5 ms
    reference_ns = 8_000_000.0              # 8 ms realistic serving draft
    v = cost_verdict(bench, reference_ns=reference_ns)
    assert v["is_near_zero"] is False
    assert v["ratio"] == pytest.approx(5_000_000.0 / 8_000_000.0)


def test_cost_verdict_boundary_is_inclusive():
    # exactly at near_zero_frac * reference counts as near-zero (<=)
    reference_ns = 1_000_000.0
    bench = {"median_ns": 0.05 * reference_ns}
    v = cost_verdict(bench, reference_ns=reference_ns, near_zero_frac=0.05)
    assert v["is_near_zero"] is True


def test_cost_verdict_rejects_nonpositive_reference():
    with pytest.raises(ValueError):
        cost_verdict({"median_ns": 1.0}, reference_ns=0.0)
    with pytest.raises(ValueError):
        cost_verdict({"median_ns": 1.0}, reference_ns=-5.0)
    with pytest.raises(ValueError):
        cost_verdict({"median_ns": 1.0}, reference_ns=1.0, near_zero_frac=0.0)


def test_bench_feature_torch_path():
    torch = pytest.importorskip("torch")
    d_in = 64
    W = torch.randn(d_in, 16)
    sample = torch.randn(d_in)
    b = bench_feature_torch(lambda x: x @ W, sample, repeats=20, warmup=2)
    assert b["backend"] == "torch"
    assert "device" in b
    assert b["p90_ns"] >= b["median_ns"] >= b["min_ns"] > 0
    assert b["repeats"] == 20
