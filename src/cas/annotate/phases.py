"""Generation-phase labels (issue I11).

Phases are absolute position bins over the generated token stream (0-indexed
output position). Absolute bins keep the pure per-token signature streaming-
friendly (no need for final length at annotate time). Version bumps when
thresholds change.

When a full sequence is available, :func:`annotate_phase_relative` can assign
relative tertiles; the primary path used by the trace writer is absolute.
"""
from __future__ import annotations

PHASE_SET_VERSION = "v1.0.0"

# Inclusive-exclusive absolute bins over output position (0-indexed).
# prefix: first 32 generated tokens; mid: next 96; late: remainder.
PREFIX_END = 32
MID_END = 128

KNOWN_PHASES: frozenset[str] = frozenset({"prefix", "mid", "late"})


def annotate_phase(position: int) -> str:
    """Return absolute generation-phase label for an output position."""
    if position < 0:
        raise ValueError(f"position must be >= 0, got {position}")
    if position < PREFIX_END:
        return "prefix"
    if position < MID_END:
        return "mid"
    return "late"


def annotate_phase_relative(position: int, total_generated: int) -> str:
    """Return relative tertile phase when total generation length is known.

    Used by stratified analysis and offline re-annotation. Positions are
    0-indexed; ``total_generated`` must be > 0.
    """
    if total_generated <= 0:
        raise ValueError("total_generated must be > 0")
    if position < 0 or position >= total_generated:
        raise ValueError(
            f"position {position} out of range for total_generated={total_generated}"
        )
    if total_generated == 1:
        return "prefix"
    if total_generated == 2:
        return "prefix" if position == 0 else "late"
    t1 = total_generated // 3
    t2 = (2 * total_generated) // 3
    # Ensure non-empty mid when length >= 3: t1 at least 1, t2 > t1.
    t1 = max(t1, 1)
    t2 = max(t2, t1 + 1)
    t2 = min(t2, total_generated - 1) if total_generated > 2 else t2
    if position < t1:
        return "prefix"
    if position < t2:
        return "mid"
    return "late"
