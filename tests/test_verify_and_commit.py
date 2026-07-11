"""I03 (partial): unit tests for the pure greedy accept/commit logic.

These run anywhere (stdlib only) and cover the seven scenarios the contract
requires at the algorithm level: all-accepted, first-token rejection, middle
rejection, zero drafting (skip), maximum drafting, and the cache-rollback
length bookkeeping. The model-level bit-identity test lives in
test_equivalence_gpu.py and runs on Modal.
"""
import pytest

from cas.commit import CommitResult, verify_and_commit


def test_all_accepted_emits_bonus_token():
    # draft d1..d4 all match target; target's 5th prediction is the bonus token
    r = verify_and_commit([11, 12, 13, 14], [11, 12, 13, 14, 99])
    assert r.accepted == 4
    assert r.drafted == 4
    assert r.emitted_ids == (11, 12, 13, 14, 99)
    assert r.cache_commit_len == 4
    assert r.first_rejection is None


def test_first_token_rejection():
    # d1 mismatches immediately; emit only the target's correction t_1
    r = verify_and_commit([50, 51, 52], [7, 51, 52, 53])
    assert r.accepted == 0
    assert r.emitted_ids == (7,)
    assert r.cache_commit_len == 0
    assert r.first_rejection == 1


def test_middle_rejection():
    # accept d1,d2; d3 mismatches; emit d1,d2,t_3
    r = verify_and_commit([20, 21, 22, 23], [20, 21, 88, 0, 0])
    assert r.accepted == 2
    assert r.emitted_ids == (20, 21, 88)
    assert r.cache_commit_len == 2
    assert r.first_rejection == 3


def test_skip_action_is_pure_autoregressive():
    # L = 0: no draft, target predicts one token, emit it
    r = verify_and_commit([], [42])
    assert r.drafted == 0
    assert r.accepted == 0
    assert r.emitted_ids == (42,)
    assert r.cache_commit_len == 0
    assert r.first_rejection is None


def test_maximum_drafting_all_accepted():
    draft = [1, 2, 3, 4, 5, 6, 7, 8]
    target = draft + [9]
    r = verify_and_commit(draft, target)
    assert r.accepted == 8
    assert r.emitted_ids == tuple(draft) + (9,)
    assert r.cache_commit_len == 8


def test_last_token_rejection():
    # accept all but the final drafted token
    r = verify_and_commit([1, 2, 3], [1, 2, 77, 5])
    assert r.accepted == 2
    assert r.emitted_ids == (1, 2, 77)
    assert r.first_rejection == 3


def test_every_round_makes_forward_progress():
    # emitted is always non-empty regardless of rejection point
    for draft, target in [
        ([], [1]),
        ([5], [9, 0]),
        ([5], [5, 0]),
        ([1, 2, 3], [1, 9, 0, 0]),
    ]:
        r = verify_and_commit(draft, target)
        assert len(r.emitted_ids) == r.accepted + 1
        assert len(r.emitted_ids) >= 1


def test_length_invariant_violation_raises():
    with pytest.raises(ValueError):
        verify_and_commit([1, 2], [1, 2])  # target too short (need len 3)
    with pytest.raises(ValueError):
        verify_and_commit([1, 2], [1, 2, 3, 4])  # target too long


def test_emitted_matches_target_greedy_sequence():
    # Property: concatenating emitted tokens across rounds reproduces exactly
    # what target-only greedy decoding produces. Simulate with an oracle target.
    # oracle: target's argmax after any prefix is prefix_len (a strictly
    # increasing deterministic sequence 0,1,2,3,...). A perfect draft proposes
    # the same; a bad draft proposes wrong tokens but must never change output.
    def target_argmax_after(prefix_len: int) -> int:
        return prefix_len  # deterministic ground-truth greedy stream

    for draft_guess in ([], [999], [0, 999], [0, 1, 2]):
        # context length starts at 5 for this check
        ctx = 5
        target = [target_argmax_after(ctx + i) for i in range(len(draft_guess) + 1)]
        r = verify_and_commit(draft_guess, target)
        # every emitted token must equal the ground-truth greedy token at its
        # position, whether it came from the draft (accepted) or the target
        for j, tok in enumerate(r.emitted_ids):
            assert tok == target_argmax_after(ctx + j)
