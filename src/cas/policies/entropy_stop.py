"""Entropy-based per-token stopping rule for issue I08.

The threshold is a frozen constructor value selected on development data.  The
rule stops only when entropy *exceeds* that value; equality continues drafting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StopContext:
    """Cheap signals available before proposing one draft token."""

    draft_index: int
    cur_entropy: float
    cur_margin: float
    proposed_so_far: tuple[int, ...]


class StopRule(Protocol):
    """Engine seam for deciding whether to stop the current draft."""

    def __call__(self, context: StopContext) -> bool: ...

    def reset(self) -> None: ...


class EntropyStopRule:
    """Stop before proposing a token whose entropy exceeds ``threshold``."""

    def __init__(self, threshold: float) -> None:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        self.threshold = float(threshold)

    def __call__(self, context: StopContext) -> bool:
        if context.draft_index < 0:
            raise ValueError("draft_index must be non-negative")
        if len(context.proposed_so_far) != context.draft_index:
            raise ValueError("proposed_so_far length must equal draft_index")
        if not math.isfinite(context.cur_entropy) or context.cur_entropy < 0:
            raise ValueError("cur_entropy must be finite and non-negative")
        return context.cur_entropy > self.threshold

    def reset(self) -> None:
        """Reset request state (the v1 rule intentionally has none)."""
