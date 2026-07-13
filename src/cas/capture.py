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
