from dataclasses import dataclass

import pytest

from cas.policies import (
    EntropyStopRule,
    RecentAcceptancePolicy,
    StopContext,
    UCBSpecNaive,
    UCBSpecPolicy,
)


@dataclass(frozen=True)
class FakeRound:
    round_id: int
    requested_action: int
    realized_draft_len: int
    accepted_prefix_len: int
    emitted_token_ids: tuple[int, ...] | None = None


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


def test_ucbspec_naive_cold_start_round_robin_and_rewards():
    policy = UCBSpecNaive(actions=(0, 2, 4), delta=0.1)
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 2
    assert policy(FakeContext(2, last_round=FakeRound(1, 2, 2, 1))) == 4
    assert policy.counts == {0: 1, 2: 1, 4: 0}
    assert policy.reward_sums == {0: 1.0, 2: 2.0, 4: 0.0}


def test_ucbspec_naive_steady_state_uses_ucb_and_reset_starts_new_request():
    policy = UCBSpecNaive(actions=(0, 2), delta=0.1)
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 2
    chosen = policy(FakeContext(2, last_round=FakeRound(1, 2, 2, 2)))
    assert chosen == 2
    assert policy.counts == {0: 1, 2: 1}
    assert policy(FakeContext(0)) == 0
    assert policy.counts == {0: 0, 2: 0}


def test_ucbspec_naive_rejects_mismatched_engine_feedback():
    policy = UCBSpecNaive(actions=(1, 2))
    assert policy(FakeContext(0)) == 1
    with pytest.raises(ValueError, match="pending arm"):
        policy(FakeContext(1, last_round=FakeRound(0, 2, 2, 1)))


def test_ucbspec_latency_reward_uses_emitted_tokens_per_frozen_cost():
    policy = UCBSpecPolicy({0: 2.0, 2: 8.0}, actions=(0, 2), delta=0.1)
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(
        1,
        last_round=FakeRound(0, 0, 0, 0, emitted_token_ids=(11,)),
    )) == 2
    policy(FakeContext(
        2,
        last_round=FakeRound(1, 2, 2, 1, emitted_token_ids=(12, 13)),
    ))
    assert policy.reward_sums == {0: 0.5, 2: 0.25}
    assert policy.mean_reward(0) == 0.5
    assert policy.mean_reward(2) == 0.25


@pytest.mark.parametrize("bad_cost", [0.0, -1.0, float("nan"), float("inf")])
def test_ucbspec_latency_requires_complete_positive_cost_profile(bad_cost):
    with pytest.raises(ValueError, match="missing action 2"):
        UCBSpecPolicy({0: 1.0}, actions=(0, 2))
    with pytest.raises(ValueError, match="finite and positive"):
        UCBSpecPolicy({0: 1.0, 2: bad_cost}, actions=(0, 2))


def test_ucbspec_naive_allows_skip_only_action():
    policy = UCBSpecNaive(actions=(0,))
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 0


def test_ucbspec_latency_cost_unit_scaling_preserves_choices_and_freezes_input():
    source_costs = {0: 2.0, 2: 5.0}
    base = UCBSpecPolicy(source_costs, actions=(0, 2), delta=0.1)
    scaled = UCBSpecPolicy({0: 2000.0, 2: 5000.0}, actions=(0, 2), delta=0.1)
    source_costs[0] = 999.0

    assert base.cost_profile[0] == 2.0
    assert _run_bandit(base, 100) == _run_bandit(scaled, 100)


def test_ucbspec_latency_uses_requested_arm_cost_when_draft_stops_early():
    policy = UCBSpecPolicy({0: 1.0, 8: 20.0}, actions=(0, 8))
    assert policy(FakeContext(0)) == 0
    assert policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0))) == 8
    policy(FakeContext(2, last_round=FakeRound(1, 8, 0, 0)))

    assert policy.mean_reward(8) == 1.0 / 20.0


def test_ucbspec_reset_preserves_frozen_cost_profile():
    policy = UCBSpecPolicy({0: 2.0, 2: 5.0}, actions=(0, 2))
    policy(FakeContext(0))
    policy(FakeContext(1, last_round=FakeRound(0, 0, 0, 0)))
    policy.reset()

    assert policy.counts == {0: 0, 2: 0}
    assert dict(policy.cost_profile) == {0: 2.0, 2: 5.0}


def _run_bandit(policy, rounds: int) -> list[int]:
    """Synthetic policy-unit-test environment, not an experimental result."""

    last_round = None
    choices: list[int] = []
    for round_id in range(rounds):
        action = policy(FakeContext(round_id, last_round=last_round))
        choices.append(action)
        # L=8 gains one extra emitted token, but at deliberately much higher
        # cost.  The raw-token baseline should prefer it; the latency-aware
        # policy should learn that skip has better emitted-token throughput.
        accepted = 0 if action == 0 else 1
        emitted = 1 if action == 0 else 2
        last_round = FakeRound(
            round_id,
            action,
            action,
            accepted,
            emitted_token_ids=tuple(range(emitted)),
        )
    return choices


def test_latency_aware_bandit_converges_away_from_expensive_aggressive_arm():
    latency_aware = UCBSpecPolicy({0: 1.0, 8: 100.0}, actions=(0, 8), delta=0.1)
    naive = UCBSpecNaive(actions=(0, 8), delta=0.1)

    repaired_tail = _run_bandit(latency_aware, 600)[-100:]
    naive_tail = _run_bandit(naive, 600)[-100:]

    assert repaired_tail.count(0) >= 90
    assert naive_tail.count(8) >= 80
