"""Unit tests for the I10 capture position/alignment helpers (pure, no torch)."""
import pytest

from cas.capture import committed_prefixes, generating_positions


def test_generating_positions():
    # prefix of length P; proposal i produced by residual state at P-1+i
    assert generating_positions(10, 3) == [9, 10, 11]
    assert generating_positions(1, 1) == [0]


def test_generating_positions_needs_context():
    with pytest.raises(ValueError):
        generating_positions(0, 4)


def test_committed_prefixes_walks_emissions():
    rounds = [
        {"round_id": 0, "start_output_pos": 0, "emitted_token_ids": (5, 6)},
        {"round_id": 1, "start_output_pos": 2, "emitted_token_ids": (7,)},
        {"round_id": 2, "start_output_pos": 3, "emitted_token_ids": (8, 9)},
    ]
    got = [(r["round_id"], pl) for r, pl in committed_prefixes(rounds, prompt_len=4)]
    # prefix_len = prompt_len(4) + tokens emitted by earlier rounds
    assert got == [(0, 4), (1, 6), (2, 7)]


def test_committed_prefixes_detects_misalignment():
    rounds = [
        {"round_id": 0, "start_output_pos": 0, "emitted_token_ids": (5, 6)},
        {"round_id": 1, "start_output_pos": 5, "emitted_token_ids": (7,)},  # wrong
    ]
    with pytest.raises(ValueError):
        list(committed_prefixes(rounds, prompt_len=4))
