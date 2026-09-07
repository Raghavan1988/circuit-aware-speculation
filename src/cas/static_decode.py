"""Static-KV, graph-captured draft decode step (issue I26, Tier-1 / D021 scope).

Why this exists
---------------
T3.4/T3.4b (CLAIMS_LEDGER 2026-07-12) established that the 0.5B draft forward is
**launch-bound** under eager execution: ~24 ms/token, invariant to sequence
length (1 vs 5 tokens cost the same), so the cost is per-forward kernel dispatch,
not compute or HBM. On that harness the M3 routing headroom collapses to ~2-5%
(best fixed action = skip); the SAME sealed acceptance labels imply 26-46%
headroom at a serving-realistic draft cost of 2-8 ms/token. To make an honest
M3/G3 decision we must measure the *deployed-regime* draft cost.

The fix T3.4 identified: neither `torch.compile(reduce-overhead)` nor `default`
helps on the HF ``DynamicCache`` decode path -- reduce-overhead (CUDA graphs)
errors because the dynamic cache's KV tensors grow and are overwritten between
graph replays, and default recompiles every step because the sequence length
varies. A **fixed-shape ``StaticCache`` single-token step** removes both
obstacles at once: one compilable shape and stable output buffers for CUDA-graph
replay. That is what this module provides.

Scope (D021): this is a **latency-characterization** path, not a scientific
(lossless) decode path and not serving-engine integration (Tier-2 / G4, deferred
by D009/D010). A timing-only characterization does not require token-identity
re-verification, but the *forward step itself* is CPU-tested to be token-
identical to the eager ``DynamicCache`` step for a fresh (no-rollback) draft
(``tests/test_static_cache_equiv.py``).

Known Tier-2 obstacle (measured 2026-09-06, I26): a correct **rollback** -- the
per-round cache truncation the speculative loop needs after rejection -- is NOT
a simple ``StaticCache`` write-pointer reset. Rewriting from an earlier
``cache_position`` leaks the stale (rejected) K/V of higher slots into the
attention read; an explicit 2D valid-length ``attention_mask`` did not suppress
it on transformers 5.13. Getting lossless static-cache rollback right is
version-sensitive KV-mask engineering -- serving-grade plumbing, and part of why
a correct engine integration is Tier-2, not a flag flip. This module therefore
does NOT wire a static path into ``SpeculativeDecoder.generate``; it powers the
microbench only, where per-forward cost is what is measured and stale slots do
not affect the timing number.

Runtime note: the canonical run environment pins ``transformers==4.46.3``;
``make_static_cache`` is written to construct across the 4.46 and 5.x
signatures. ``torch.compile`` / CUDA-graph replay is exercised on Modal/GPU; the
local CPU suite runs the eager static step only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


def make_static_cache(model, max_cache_len: int):
    """Allocate a ``StaticCache`` sized to ``max_cache_len``, robust to the
    transformers 4.46 vs 5.x constructor signatures.

    4.46: ``StaticCache(config, max_batch_size, max_cache_len, device, dtype)``.
    5.x:  ``StaticCache(config, max_cache_len, ...)`` (device/dtype inferred).
    """
    from transformers import StaticCache

    try:  # transformers 4.46 style (the pinned run environment)
        return StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=model.dtype,
        )
    except TypeError:  # transformers 5.x style
        return StaticCache(config=model.config, max_cache_len=max_cache_len)


def static_forward(model, input_ids: torch.Tensor, cache, cache_position: torch.Tensor):
    """One forward writing into ``cache`` at ``cache_position``.

    Fixed input shape (call with a constant ``input_ids`` length -- 1 for the
    per-token decode step, or the prompt length for the one-shot prefill) so the
    compiled step has a single trace and a CUDA graph can replay it. Returns the
    logits tensor ``[1, seq, vocab]``.
    """
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )
    return out.logits


@dataclass
class StaticDraftStepper:
    """Runs a warm-prefill + fixed-shape per-token draft on a ``StaticCache``.

    Latency-characterization use (I26): construct once, ``prefill`` the context,
    then call ``step`` in a loop to time the graph-captured single-token decode.
    ``compile_mode`` (e.g. ``"reduce-overhead"``) wraps the step function with
    ``torch.compile``; leave it ``None`` for the eager static path (the only path
    the CPU suite exercises).

    This does not implement rollback (see module docstring): reuse across timed
    iterations resets ``pos`` to the post-prefill length, which is sufficient for
    a per-forward cost measurement and deliberately not a lossless continuation.
    """

    model: Any
    max_cache_len: int
    compile_mode: str | None = None

    def __post_init__(self) -> None:
        self.cache = make_static_cache(self.model, self.max_cache_len)
        self.device = self.model.device
        self._prefill_len = 0
        self.pos = 0
        # Pre-allocated STATIC input buffers. CUDA-graph replay (reduce-overhead)
        # requires the graph's inputs to live at fixed addresses: we mutate these
        # in place each step (never re-allocate), so the captured graph reads the
        # new token id / cache_position on replay. Freshly-allocated per-step
        # inputs are what trip the "tensor overwritten by a subsequent run" error
        # (CLAIMS_LEDGER T3.4).
        self._ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._cp = torch.zeros((1,), dtype=torch.long, device=self.device)
        step: Callable = static_forward
        if self.compile_mode:
            # Compile the bare forward; the step's shape is constant [1,1] so this
            # traces once. reduce-overhead additionally captures a CUDA graph.
            step = torch.compile(static_forward, mode=self.compile_mode)
        self._step = step

    def prefill(self, prompt_ids: list[int]) -> torch.Tensor:
        """Process the prompt in one (non-graph) eager forward; returns last-token
        logits (cloned). Resets the write pointer to the end of the prompt."""
        ids = torch.tensor([prompt_ids], device=self.device)
        cp = torch.arange(0, ids.shape[1], device=self.device)
        logits = static_forward(self.model, ids, self.cache, cp)
        self._prefill_len = ids.shape[1]
        self.pos = ids.shape[1]
        return logits[0, -1].clone()

    def step(self, token) -> torch.Tensor:
        """Advance one token through the fixed-shape (graph-captured) path.

        ``token`` may be a python int (CPU tests) or a 0-dim device tensor (the
        deployment-realistic path: the previous step's on-device argmax, written
        device-to-device with NO host sync so the timing is comparable to the
        eager no-sync baseline). Returns the new last-token logits (cloned, so the
        caller keeps a stable copy across the next CUDA-graph replay).
        """
        if isinstance(token, torch.Tensor):
            self._ids[0, 0].copy_(token.reshape(()))  # device->device, no sync
        else:
            self._ids[0, 0] = int(token)
        self._cp[0] = self.pos
        logits = self._step(self.model, self._ids, self.cache, self._cp)
        self.pos += 1
        return logits[0, -1].clone()

    def rewind_to_prefill(self) -> None:
        """Reset the write pointer to the post-prefill position for the next
        timed draft. Timing-only: stale slots are not cleared (see docstring)."""
        self.pos = self._prefill_len
