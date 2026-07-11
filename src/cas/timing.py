"""Synchronized latency instrumentation (issue I04).

Wall-clock is a lie for GPU work because kernel launches are asynchronous, so we
time with CUDA events and synchronize at boundaries. On CPU (no CUDA) we fall
back to perf_counter so the engine still runs for logic checks, but such numbers
are explicitly marked non-authoritative.

Every component the contract requires is timed separately: prefill, draft,
verify, controller, tracing. `timer.overhead_ns` characterizes the timer itself
so we can confirm the instrumentation is negligible.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class LatencyAccumulator:
    """Sums per-component nanoseconds over a request; all fields authoritative
    only when `cuda` is True."""

    cuda: bool
    components_ns: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, ns: int) -> None:
        self.components_ns[name] = self.components_ns.get(name, 0) + ns

    def total_ns(self) -> int:
        return sum(self.components_ns.values())


class Stopwatch:
    """Times labeled code blocks. Uses CUDA events when available.

    Usage:
        sw = Stopwatch(device)
        with sw.measure("draft"):
            ...
        sw.acc.components_ns["draft"]  # nanoseconds
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.acc = LatencyAccumulator(cuda=self.cuda)

    @contextmanager
    def measure(self, name: str):
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)  # milliseconds, device-synchronized
                self.acc.add(name, int(ms * 1e6))
        else:
            t0 = time.perf_counter_ns()
            try:
                yield
            finally:
                self.acc.add(name, time.perf_counter_ns() - t0)

    def measure_overhead_ns(self, iters: int = 1000) -> int:
        """Mean cost of an empty measured block, to confirm timer overhead is
        negligible relative to draft/verify. Recorded per run."""
        tmp = Stopwatch(self.device)
        for _ in range(iters):
            with tmp.measure("noop"):
                pass
        return tmp.acc.components_ns.get("noop", 0) // max(iters, 1)
