from dataclasses import dataclass

import pytest

from cas.policies import EntropyStopRule, RecentAcceptancePolicy, StopContext, UCBSpecPolicy


@dataclass(frozen=True)
class FakeRound:
    round_id: int
    requested_action: int
    realized_draft_len: int
    accepted_prefix_len: int


@dataclass(frozen=True)
class FakeContext:
    round_id: int
    generated_so_far: int = 0
    last_round: FakeRound | None = None


def test_entropy_stop_threshold_boundary_and_validation():
    rule = EntropyStopRule(1.5)
    assert not rule(StopContext(0, 1.5, 0.2, ()))
    assert rule(StopContext(1, 1.50001, 0.2, (7,)))
    with pytest.raises(ValueError, match="proposed_so_far"):
        rule(StopContext(1, 1.0, 0.2, ()))


def test_entropy_stop_reset_preserves_frozen_threshold():
    rule = EntropyStopRule(1.5)
    rule.reset()
    assert rule.threshold == 1.5


def test_recent_acceptance_boundary_skip_and_request_reset():
    policy = RecentAcceptancePolicy(
        ((0.0, 0), (0.5, 4), (0.75, 8)), initial_action=2, window_size=2
    )
    assert policy(FakeContext(0)) == 2
    assert policy(FakeContext(1, last_round=FakeRound(0, 2, 2, 1))) == 4
    assert policy.acceptance_rate == 0.5
    assert policy(FakeContext(2, last_round=FakeRound(1, 0, 0, 0))) == 4
    assert policy.acceptance_rate == 0.5
    assert policy(FakeContext(0)) == 2
    assert policy.acceptance_rate is None


def test_recent_acceptance_uses_rolling_window():
    policy = RecentAcceptancePolicy(((0.0, 1), (0.75, 8)), initial_action=1, window_size=2)
    policy(FakeContext(0))
    policy(FakeContext(1, last_round=FakeRound(0, 1, 1, 0)))
    policy(FakeContext(2, last_round=FakeRound(1, 1, 1, 1)))
    assert policy.acceptance_rate == 0.5
    assert policy(FakeContext(3, last_round=FakeRound(2, 1, 1, 1))) == 8


def test_ucbspec_cold_start_round_robin_and_rewards():
    policy = UCBSpecPolicy(actions=(0, 2, 4), delta=0.1)
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 2
    assert policy(FakeContext(2, last_round=FakeRound(1, 2, 2, 1))) == 4
    assert policy.counts == {0: 1, 2: 1, 4: 0}
    assert policy.reward_sums == {0: 1.0, 2: 2.0, 4: 0.0}


def test_ucbspec_steady_state_uses_ucb_and_reset_starts_new_request():
    policy = UCBSpecPolicy(actions=(0, 2), delta=0.1)
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 2
    chosen = policy(FakeContext(2, last_round=FakeRound(1, 2, 2, 2)))
    assert chosen == 2
    assert policy.counts == {0: 1, 2: 1}
    assert policy(FakeContext(0)) == 0
    assert policy.counts == {0: 0, 2: 0}


def test_ucbspec_rejects_mismatched_engine_feedback():
    policy = UCBSpecPolicy(actions=(1, 2))
    assert policy(FakeContext(0)) == 1
    with pytest.raises(ValueError, match="pending arm"):
        policy(FakeContext(1, last_round=FakeRound(0, 2, 2, 1)))
