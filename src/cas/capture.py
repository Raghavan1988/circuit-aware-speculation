"""Selected-layer draft activation capture (I10).

Teacher-forced capture: for one decode round we feed the draft the exact
committed prefix followed by the SEALED proposed tokens in a single forward with
`output_hidden_states=True`, then read the residual stream at the position that
*generated* each proposal. Because the fed sequence is the immutable sealed
proposal sequence, token-to-activation alignment is exact by construction (no
bf16 trajectory drift), and the sealed per-position acceptance labels
(`target_match`, D018.3) apply directly — the activations are fresh, the labels
are the frozen ground truth.

Pure/torch-free helpers here so the position arithmetic is unit-testable; the
model forward + volume IO live in the Modal `capture_activations` function.
"""
from __future__ import annotations

# Early / middle / late / final hidden_states indices for Qwen2.5-0.5B (24
# transformer layers -> hidden_states has 25 entries, 0 = embeddings). The
# backlog (I10) asks for declared early/middle/late points; we add final.
DEFAULT_LAYERS = (6, 12, 18, 24)


def generating_positions(prefix_len: int, n_proposals: int) -> list[int]:
    """Absolute positions whose residual stream *produces* each proposal.

    In one forward over [prefix (len P) | proposals], the hidden state at
    position P-1 produces proposal 0, and position P-1+i produces proposal i
    (the state after consuming proposal i-1). Returns the P-1+i list.
    """
    if prefix_len < 1:
        raise ValueError("prefix must be non-empty (the draft needs a context)")
    return [prefix_len - 1 + i for i in range(n_proposals)]


def frontier_position(prefix_len: int) -> int:
    """Absolute index of the verified-context FRONTIER (last committed) position.

    Given a committed prefix of length ``prefix_len`` (the tokens already emitted
    and verified BEFORE a decode round drafts), the target residual stream at the
    last-committed position -- index ``prefix_len - 1`` -- is the representation
    that conditions the next drafted round. A probe on that vector predicts the
    round's acceptance before any draft compute is spent (I23/C10). Torch-free so
    the arithmetic stays unit-testable next to ``generating_positions``.

    Raises ValueError if ``prefix_len < 1`` (no committed context to condition on).
    """
    if prefix_len < 1:
        raise ValueError("prefix must be non-empty (need a committed frontier position)")
    return prefix_len - 1


def committed_prefixes(rounds: list[dict], prompt_len: int):
    """Yield (round, prefix_len) walking rounds in order, where prefix_len is
    prompt_len plus all tokens emitted by earlier rounds.

    `rounds` are dicts from rounds.parquet (need `emitted_token_ids`,
    `start_output_pos`); they must be for a single request. The running emitted
    count is validated against the recorded `start_output_pos` so a mismatch
    (schema drift, wrong ordering) is caught rather than silently misaligning.
    """
    rounds = sorted(rounds, key=lambda r: r["round_id"])
    emitted = 0
    for r in rounds:
        if r["start_output_pos"] != emitted:
            raise ValueError(
                f"round {r['round_id']} start_output_pos={r['start_output_pos']} "
                f"but running emitted={emitted}; ordering/emit mismatch")
        yield r, prompt_len + emitted
        emitted += len(r["emitted_token_ids"] or ())
