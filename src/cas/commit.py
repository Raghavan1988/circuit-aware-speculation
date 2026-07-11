"""Pure greedy speculative-decoding verification logic (stdlib only).

This module is deliberately free of torch / numpy / transformers so the
correctness of the accept/commit/rollback bookkeeping — the part most prone to
silent bugs — can be unit-tested anywhere, including CPU-only machines and CI.

Nomenclature (greedy, exact-match; see docs/EXPERIMENT_CONTRACT.md "Decoding"):

    context C of length n is already committed.
    the draft proposes tokens d_1 .. d_L   (L = action length; L = 0 is `skip`).
    the target is run once over [C, d_1, .., d_L] and yields, at each position,
    its own greedy argmax prediction of the *next* token:

        t_1 = argmax target(C)                     -> the token after C
        t_2 = argmax target(C, d_1)                -> the token after d_1
        ...
        t_{L+1} = argmax target(C, d_1, .., d_L)   -> the bonus token

    so target_argmax has length L + 1, draft has length L.

Acceptance rule: accept d_i while d_i == t_i, stopping at the first mismatch.
Let k be the number accepted. The round emits d_1 .. d_k followed by t_{k+1}
(the target's own token at the divergence point, or the bonus token if k = L).

That emitted run is *exactly* what target-only greedy decoding would produce,
which is why exact-verification speculative decoding is lossless: the draft
changes only speed, never output. `verify_and_commit` returns everything the
engine and the trace schema need to record and to roll the KV caches back.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommitResult:
    """Outcome of verifying one draft/verify round.

    Attributes:
        drafted:     L, the number of tokens the draft proposed this round.
        accepted:    k, the length of the accepted draft prefix (0 <= k <= L).
        emitted_ids: the k+1 tokens committed this round (accepted prefix plus
                     the target's correction/bonus token). Always non-empty, so
                     every round makes at least one token of forward progress.
        cache_commit_len: the number of *draft-proposed* positions that remain
                     valid in the KV caches after this round, i.e. k. Both the
                     target and draft caches must be rolled back to
                     (context_len + k); the emitted bonus token t_{k+1} is not
                     yet in either cache and is (re)processed next round.
        first_rejection: index (1-based, into the draft) of the first rejected
                     token, or None if the whole draft was accepted. Recorded
                     for the trace / error taxonomy.
    """

    drafted: int
    accepted: int
    emitted_ids: tuple[int, ...]
    cache_commit_len: int
    first_rejection: int | None


def verify_and_commit(
    draft_ids: list[int] | tuple[int, ...],
    target_argmax_ids: list[int] | tuple[int, ...],
) -> CommitResult:
    """Apply the greedy accept rule to one round.

    Args:
        draft_ids: d_1 .. d_L proposed by the draft (empty tuple/list for skip).
        target_argmax_ids: t_1 .. t_{L+1}, the target's greedy next-token
            argmax at each position; must have exactly len(draft_ids) + 1 items.

    Returns:
        CommitResult with the accepted prefix length, emitted tokens, and the
        cache rollback length.

    Raises:
        ValueError: if the length invariant target = draft + 1 is violated, or
            if target_argmax is empty (the target must always predict at least
            the next token, even for skip).
    """
    L = len(draft_ids)
    if len(target_argmax_ids) != L + 1:
        raise ValueError(
            f"target_argmax must have len(draft)+1 = {L + 1} items, "
            f"got {len(target_argmax_ids)}"
        )

    k = 0
    first_rejection: int | None = None
    for i in range(L):
        if draft_ids[i] == target_argmax_ids[i]:
            k += 1
        else:
            first_rejection = i + 1  # 1-based index into the draft
            break

    # emitted = accepted draft prefix + the target's own token at position k
    emitted = tuple(draft_ids[:k]) + (target_argmax_ids[k],)

    return CommitResult(
        drafted=L,
        accepted=k,
        emitted_ids=emitted,
        cache_commit_len=k,
        first_rejection=first_rejection,
    )


def reference_next_token(target_argmax_first: int) -> int:
    """The token target-only greedy decoding would emit next.

    Equivalent to `verify_and_commit([], [t]).emitted_ids[0]`; provided so the
    equivalence oracle can be written without constructing a CommitResult.
    """
    return target_argmax_first
