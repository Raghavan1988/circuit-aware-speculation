"""Deployed-cost probe for candidate pre-round feature transforms (I23/G3, D023).

The whole premise of a *pre-round* signal is that it is computable at near-zero
marginal cost: it reads a vector that is already cached (the target frontier
residual at the last-committed position) and applies a cheap statistic before
round r spends any draft compute. A candidate that predicts acceptance well but
costs a draft forward pass has failed its own premise. This module lets the
generator-critic loop *prove* near-zero cost by measuring it, rather than
asserting it (AGENTS.md: never fabricate timing numbers).

Conventions mirror ``cas.timing`` (issue I04): all durations are integer/float
nanoseconds; the torch path synchronizes CUDA at the timing boundaries so the
asynchronous kernel launches are actually captured (wall-clock alone is a lie
for GPU work). Cost timing belongs in spirit to the compiled/fast path (D021):
here we only supply the harness -- a pure timer of a transform -- not an engine.

The verdict compares the measured median against a *caller-supplied* reference
cost (a measured per-round draft/verify cost). This module never hardcodes a
draft cost: for reference, the ledger records eager draft ~24 ms/token
(launch-bound) and a realistic serving draft ~2-8 ms/token, but those are
inputs the caller measures and passes in, not constants baked in here.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np


def _summarize_ns(samples_ns: list[int], repeats: int, backend: str) -> dict:
    """Reduce a list of per-call nanosecond timings to the reported summary.

    p90 is the inclusive-method nearest-rank percentile so it is always one of
    the observed samples and satisfies p90 >= median >= min for any input.
    """
    if not samples_ns:
        raise ValueError("no timing samples collected (repeats must be >= 1)")
    ordered = sorted(samples_ns)
    n = len(ordered)
    # nearest-rank: ceil(0.90 * n), 1-indexed, clamped into range.
    rank = max(1, min(n, int(-(-90 * n // 100))))
    return {
        "mean_ns": float(statistics.fmean(ordered)),
        "median_ns": float(statistics.median(ordered)),
        "p90_ns": float(ordered[rank - 1]),
        "min_ns": int(ordered[0]),
        "repeats": int(n),
        "backend": backend,
    }


def bench_feature(
    transform: Callable[[np.ndarray], np.ndarray],
    sample: np.ndarray,
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """Time a numpy feature transform with ``time.perf_counter_ns``.

    transform: Callable[[np.ndarray], np.ndarray] -- the candidate pre-round
        feature transform (e.g. a fixed projection matmul on one cached vector).
    sample: a representative frontier rep ndarray, shape ``(d_model,)`` or
        ``(n, d_model)``.
    repeats: number of timed iterations (>= 1).
    warmup: untimed iterations run first to page in code paths / caches.

    Returns ``{mean_ns, median_ns, p90_ns, min_ns, repeats, backend:'numpy'}``.
    All durations are floats/ints in nanoseconds. No global state.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        transform(sample)

    samples_ns: list[int] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        transform(sample)
        samples_ns.append(time.perf_counter_ns() - t0)

    return _summarize_ns(samples_ns, repeats, backend="numpy")


def bench_feature_torch(
    fn: Callable,
    sample,
    repeats: int = 100,
    warmup: int = 10,
    synchronize: bool = True,
) -> dict:
    """Time a torch transform on the sample's own device.

    Mirrors ``cas.timing``: when ``synchronize`` is set and the sample is on
    CUDA, ``torch.cuda.synchronize()`` is called at the timing boundaries so the
    asynchronous kernel launches are included in the measurement (otherwise the
    numbers would be non-authoritative wall-clock, per timing.py's warning).

    torch is imported lazily inside this function so the module imports without
    torch installed.

    fn: Callable applied to ``sample`` each iteration (a torch transform).
    sample: a torch.Tensor whose ``.device`` determines where timing happens.
    Returns the standard summary dict with ``backend:'torch'`` and ``device``.
    """
    import torch  # lazy: module must import without torch

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    device = getattr(sample, "device", torch.device("cpu"))
    use_cuda = synchronize and getattr(device, "type", "cpu") == "cuda" and torch.cuda.is_available()

    def _sync() -> None:
        if use_cuda:
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        fn(sample)
    _sync()

    samples_ns: list[int] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter_ns()
        fn(sample)
        _sync()
        samples_ns.append(time.perf_counter_ns() - t0)

    out = _summarize_ns(samples_ns, repeats, backend="torch")
    out["device"] = str(device)
    return out


def cost_verdict(
    bench: dict,
    reference_ns: float,
    near_zero_frac: float = 0.05,
) -> dict:
    """Judge a benchmark against a caller-supplied reference cost.

    bench: a dict returned by ``bench_feature`` / ``bench_feature_torch`` (must
        contain ``median_ns``).
    reference_ns: a MEASURED per-round draft/verify cost supplied by the caller.
        This function NEVER hardcodes a draft cost -- the deployed reference is
        an input, so the same verdict logic works for the eager (~24 ms) and the
        realistic serving (~2-8 ms) regimes without code changes.
    near_zero_frac: the fraction of the reference below which the transform
        counts as "near-zero cost" (default 5%).

    Returns ``{is_near_zero, ratio, median_ns, reference_ns, near_zero_frac}``
    where ``ratio = median_ns / reference_ns`` and
    ``is_near_zero = median_ns <= near_zero_frac * reference_ns``.
    """
    if reference_ns <= 0:
        raise ValueError("reference_ns must be positive (a measured cost)")
    if not (0 < near_zero_frac <= 1):
        raise ValueError("near_zero_frac must be in (0, 1]")

    median_ns = float(bench["median_ns"])
    ratio = median_ns / reference_ns
    return {
        "is_near_zero": bool(median_ns <= near_zero_frac * reference_ns),
        "ratio": ratio,
        "median_ns": median_ns,
        "reference_ns": float(reference_ns),
        "near_zero_frac": float(near_zero_frac),
    }
