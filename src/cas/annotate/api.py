"""Public annotation API (issue I11).

Torch-free pure functions of the token stream. Claude's trace writer (I06)
should call :func:`annotate_token` per token once the schema fields land.
"""
from __future__ import annotations

from dataclasses import dataclass

from cas.annotate.categories import CATEGORY_SET_VERSION, annotate_categories
from cas.annotate.phases import PHASE_SET_VERSION, annotate_phase, annotate_phase_relative


@dataclass(frozen=True)
class AnnotatedToken:
    """Per-token annotation record for the token-trace seam."""

    categories: frozenset[str]
    phase: str
    category_set_version: str
    phase_set_version: str = PHASE_SET_VERSION

    def categories_sorted(self) -> list[str]:
        """Stable list form for Parquet/JSON serialization."""
        return sorted(self.categories)


def annotate_token(
    token_id: int,
    piece: str,
    position: int,
    context_pieces: list[str],
) -> AnnotatedToken:
    """Annotate one token from its id, decoded piece, and prior pieces.

    Parameters
    ----------
    token_id:
        Vocabulary id (informational; used only for special-id heuristics).
    piece:
        Decoded string for this token (tokenizer-dependent surface form).
    position:
        0-indexed position in the *generated* (or proposed) stream.
    context_pieces:
        Decoded pieces strictly before this token in the same stream
        (prompt tokens may be included when available; used for
        ``repeated_span`` and future context features).

    Returns
    -------
    AnnotatedToken
        Overlapping category set + absolute generation phase + versions.
    """
    cats = annotate_categories(
        piece, token_id=token_id, context_pieces=context_pieces
    )
    phase = annotate_phase(position)
    return AnnotatedToken(
        categories=cats,
        phase=phase,
        category_set_version=CATEGORY_SET_VERSION,
        phase_set_version=PHASE_SET_VERSION,
    )


def annotate_sequence(
    pieces: list[str],
    token_ids: list[int] | None = None,
    *,
    relative_phase: bool = False,
) -> list[AnnotatedToken]:
    """Annotate a full piece stream left-to-right.

    If ``relative_phase`` is True, use tertile phases over the sequence length
    (offline analysis). Default matches the streaming path (absolute bins).
    """
    if token_ids is None:
        token_ids = [0] * len(pieces)
    if len(token_ids) != len(pieces):
        raise ValueError("token_ids and pieces must have the same length")

    out: list[AnnotatedToken] = []
    for i, (tid, piece) in enumerate(zip(token_ids, pieces)):
        ctx = pieces[:i]
        cats = annotate_categories(piece, token_id=tid, context_pieces=ctx)
        if relative_phase:
            phase = annotate_phase_relative(i, len(pieces))
        else:
            phase = annotate_phase(i)
        out.append(
            AnnotatedToken(
                categories=cats,
                phase=phase,
                category_set_version=CATEGORY_SET_VERSION,
                phase_set_version=PHASE_SET_VERSION,
            )
        )
    return out
