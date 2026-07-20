"""Pure (torch-free) tests for the I23/C10 target-frontier capture helpers.

Covers the ``frontier_position`` arithmetic and the pre-round alignment invariant
that ``capture_frontier_activations`` relies on: one frontier row per round that
has proposals, positioned at the last-committed index BEFORE the round drafts,
labelled by that round's own realized acceptance. The Modal capture itself needs
torch/GPU/volumes and cannot run here, so we exercise a pure reference walk that
mirrors its per-round logic exactly.
"""
import pytest

from cas.capture import frontier_position


def test_frontier_position_values():
    # frontier == last-committed index == prefix_len - 1
    assert frontier_position(1) == 0
    assert frontier_position(5) == 4
    assert frontier_position(200) == 199


def test_frontier_position_needs_context():
    with pytest.raises(ValueError):
        frontier_position(0)
    with pytest.raises(ValueError):
        frontier_position(-3)


def _frontier_rows(prompt_len, rounds):
    """Reference for capture_frontier_activations' pure per-round walk.

    Mirrors the Modal function: prefix starts at prompt_len + 1 (the committed
    prefill token before round 0), each round appends its emissions, and rounds
    WITHOUT proposals are skipped. Emits one row per proposal-bearing round with
    (round_id, frontier_pos, accept, accepted_len). Raises on emit/pos mismatch,
    matching the engine's start_output_pos invariant.
    """
    prefix_len = prompt_len + 1  # prompt + committed prefill token (generated[0])
    rows = []
    for r in sorted(rounds, key=lambda x: x["round_id"]):
        if r["start_output_pos"] != prefix_len - prompt_len:
            raise ValueError(
                f"r{r['round_id']} emit/pos mismatch "
                f"({r['start_output_pos']} vs {prefix_len - prompt_len})")
        proposals = list(r.get("proposed_token_ids") or ())
        emitted = list(r.get("emitted_token_ids") or ())
        if proposals:
            rows.append({
                "round_id": r["round_id"],
                "frontier_pos": frontier_position(prefix_len),
                "accept": int(r["accepted_prefix_len"] >= 1),
                "accepted_len": int(r["accepted_prefix_len"]),
            })
        prefix_len += len(emitted)
    return rows


def _fake_rounds():
    # prompt_len = 3, generated[0] = prefill token, so round 0 starts at pos 1.
    return [
        # accepted round (accept=1)
        {"round_id": 0, "start_output_pos": 1, "proposed_token_ids": (10, 11),
         "emitted_token_ids": (10, 20), "accepted_prefix_len": 1},
        # skip round: no proposals -> no frontier row, but still advances prefix
        {"round_id": 1, "start_output_pos": 3, "proposed_token_ids": (),
         "emitted_token_ids": (30,), "accepted_prefix_len": 0},
        # rejected round (accept=0)
        {"round_id": 2, "start_output_pos": 4, "proposed_token_ids": (40, 41, 42),
         "emitted_token_ids": (50,), "accepted_prefix_len": 0},
    ]


def test_row_count_equals_rounds_with_proposals():
    rounds = _fake_rounds()
    rows = _frontier_rows(prompt_len=3, rounds=rounds)
    n_with_proposals = sum(1 for r in rounds if r["proposed_token_ids"])
    assert len(rows) == n_with_proposals == 2


def test_frontier_positions_are_last_committed_before_round():
    rows = _frontier_rows(prompt_len=3, rounds=_fake_rounds())
    # round 0: prefix = prompt(3) + prefill(1) = 4 -> frontier 3
    # round 2: prefix = 4 + emitted r0(2) + emitted r1(1) = 7 -> frontier 6
    assert [(r["round_id"], r["frontier_pos"]) for r in rows] == [(0, 3), (2, 6)]


def test_label_derivation_from_round_acceptance():
    rows = _frontier_rows(prompt_len=3, rounds=_fake_rounds())
    by_id = {r["round_id"]: r for r in rows}
    assert (by_id[0]["accept"], by_id[0]["accepted_len"]) == (1, 1)
    assert (by_id[2]["accept"], by_id[2]["accepted_len"]) == (0, 0)


def test_emit_pos_mismatch_raises():
    bad = [
        {"round_id": 0, "start_output_pos": 1, "proposed_token_ids": (10,),
         "emitted_token_ids": (10, 20), "accepted_prefix_len": 1},
        # start_output_pos should be 3 after r0 emits 2 tokens; 9 is wrong.
        {"round_id": 1, "start_output_pos": 9, "proposed_token_ids": (11,),
         "emitted_token_ids": (11,), "accepted_prefix_len": 1},
    ]
    with pytest.raises(ValueError):
        _frontier_rows(prompt_len=3, rounds=bad)
