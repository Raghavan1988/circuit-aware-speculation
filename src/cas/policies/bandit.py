"""UCBSpec-style draft-length bandits for issue I09.

``UCBSpecNaive`` preserves the round-1 reproduction of UCBSpec
(arXiv:2505.15141), Algorithm 4/8: round-robin cold start, reward equal to the
number of emitted tokens, empirical means, and the published self-normalized
confidence radius.  It is retained as a disclosed naive baseline because an
emitted-token reward does not price the latency of an action.

``UCBSpecPolicy`` is the D018 repair used for controller comparisons.  Its
reward is emitted tokens per frozen measured round cost.  The cost profile is
mandatory constructor input and must be estimated on development traces; this
module deliberately provides no default or invented timing values.

Documented deviations from arXiv:2505.15141 shared by both variants:

* Arms are draft lengths ``(skip, 1, 2, 3, 4, 6, 8)``, rather than alternative
  speculative-decoding hyperparameter specifications under a common maximum L.
* Greedy exact-match rounds supply deterministic realized rewards conditional
  on a prefix; the paper also treats sampling and broader hyperparameters.
* The finite generation cap is enforced by the surrounding engine, not by this
  pure selector.  These resettable per-request policies do not claim the
  paper's stopping-time guarantee for the changed arm definition.
* Ties are resolved by configured arm order for reproducibility.

Additional deviation in ``UCBSpecPolicy``: the empirical mean and confidence
radius are expressed in emitted-tokens-per-cost units.  The radius is scaled by
the largest feasible per-action throughput ``(L + 1) / cost(L)``; maximizing
this reward targets measured speed rather than raw accepted length.

The policies consume the previous ``RoundTrace`` on the next call, avoiding an
engine callback or any edit to the Claude-owned decoding loop.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from cas.config import ACTION_LENGTHS


def _normalize_actions(actions: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(int(operator.index(action)) for action in actions)
    except TypeError as error:
        raise ValueError("actions must be exact integers") from error
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("actions must be non-empty and unique")
    if any(action < 0 for action in normalized):
        raise ValueError("actions must be non-negative")
    return normalized


def _emitted_tokens(round_trace, accepted: int, realized: int) -> int:
    """Return the authoritative emitted count, with a legacy-test fallback."""

    emitted_ids = getattr(round_trace, "emitted_token_ids", None)
    emitted = accepted + 1 if emitted_ids is None else len(emitted_ids)
    if emitted != accepted + 1:
        raise ValueError("emitted-token count must equal accepted_prefix_len + 1")
    return emitted


class _UCBSpecBase:
    """Shared reset, observation, and UCB selection mechanics."""

    def __init__(
        self,
        *,
        actions: Sequence[int],
        delta: float,
        max_speculation_length: int | None,
        confidence_reward_bound: float,
    ) -> None:
        normalized = _normalize_actions(actions)
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if max_speculation_length is None:
            maximum = max(normalized)
        else:
            try:
                maximum = int(operator.index(max_speculation_length))
            except TypeError as error:
                raise ValueError("max_speculation_length must be an integer") from error
        if maximum < max(normalized):
            raise ValueError("max_speculation_length cannot be below an action")
        if not math.isfinite(confidence_reward_bound) or confidence_reward_bound < 0:
            raise ValueError("confidence reward bound must be finite and non-negative")

        self.actions = normalized
        self.delta = float(delta)
        self.max_speculation_length = maximum
        self._confidence_reward_bound = float(confidence_reward_bound)
        self.reset()

    def reset(self) -> None:
        self.counts = {action: 0 for action in self.actions}
        self.reward_sums = {action: 0.0 for action in self.actions}
        self._pending_action: int | None = None
        self._last_observed_round: int | None = None

    def _reward(self, requested: int, emitted: int) -> float:
        raise NotImplementedError

    def _observe(self, round_trace) -> None:
        if self._pending_action is None:
            raise ValueError("last_round supplied before this policy selected an action")
        try:
            requested = int(operator.index(round_trace.requested_action))
            accepted = int(operator.index(round_trace.accepted_prefix_len))
            realized = int(operator.index(round_trace.realized_draft_len))
        except TypeError as error:
            raise ValueError("round feedback counts must be exact integers") from error
        if requested != self._pending_action:
            raise ValueError("last_round action does not match the policy's pending arm")
        if realized < 0 or realized > requested or not 0 <= accepted <= realized:
            raise ValueError("invalid acceptance counts in last_round")

        emitted = _emitted_tokens(round_trace, accepted, realized)
        reward = self._reward(requested, emitted)
        if not math.isfinite(reward) or reward < 0:
            raise ValueError("bandit reward must be finite and non-negative")
        self.counts[requested] += 1
        self.reward_sums[requested] += reward
        self._last_observed_round = int(round_trace.round_id)
        self._pending_action = None

    def mean_reward(self, action: int) -> float | None:
        """Return an arm's empirical mean, or ``None`` before it is observed."""

        try:
            action = int(operator.index(action))
        except TypeError as error:
            raise ValueError("action must be an exact integer") from error
        if action not in self.counts:
            raise ValueError(f"unknown action {action}")
        n = self.counts[action]
        return self.reward_sums[action] / n if n else None

    def _ucb(self, action: int, pulls: int) -> float:
        n = self.counts[action]
        if n == 0:
            return math.inf
        mean = self.reward_sums[action] / n
        k = len(self.actions)
        log_arg = k * pulls**2 * math.sqrt(1 + n) / self.delta
        radius = (self._confidence_reward_bound / 2) * math.sqrt(
            ((1 + n) / n**2) * (1 + 2 * math.log(log_arg))
        )
        return mean + radius

    def __call__(self, context) -> int:
        if context.round_id == 0:
            self.reset()
        last = context.last_round
        if last is not None and self._last_observed_round != last.round_id:
            self._observe(last)

        total_pulls = sum(self.counts.values())
        if total_pulls < len(self.actions):
            action = self.actions[total_pulls]
        else:
            action = max(self.actions, key=lambda arm: self._ucb(arm, total_pulls))
        self._pending_action = action
        return action


class UCBSpecNaive(_UCBSpecBase):
    """Disclosed raw-emitted-token reproduction; not speed-optimal control."""

    def __init__(
        self,
        *,
        actions: Sequence[int] = ACTION_LENGTHS,
        delta: float = 0.05,
        max_speculation_length: int | None = None,
    ) -> None:
        normalized = _normalize_actions(actions)
        if max_speculation_length is None:
            maximum = max(normalized)
        else:
            try:
                maximum = int(operator.index(max_speculation_length))
            except TypeError as error:
                raise ValueError("max_speculation_length must be an integer") from error
        super().__init__(
            actions=normalized,
            delta=delta,
            max_speculation_length=maximum,
            # Preserve the round-1 confidence-radius scale exactly.
            confidence_reward_bound=float(maximum),
        )

    def _reward(self, requested: int, emitted: int) -> float:
        del requested
        return float(emitted)


class UCBSpecPolicy(_UCBSpecBase):
    """Latency-aware length bandit using a frozen development cost profile."""

    def __init__(
        self,
        cost_profile: Mapping[int, float],
        *,
        actions: Sequence[int] = ACTION_LENGTHS,
        delta: float = 0.05,
        max_speculation_length: int | None = None,
    ) -> None:
        normalized = _normalize_actions(actions)
        costs: dict[int, float] = {}
        for action in normalized:
            if action not in cost_profile:
                raise ValueError(f"cost_profile is missing action {action}")
            cost = float(cost_profile[action])
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("cost_profile values must be finite and positive")
            costs[action] = cost

        reward_bound = max((action + 1) / costs[action] for action in normalized)
        super().__init__(
            actions=normalized,
            delta=delta,
            max_speculation_length=max_speculation_length,
            confidence_reward_bound=reward_bound,
        )
        # Prevent accidental online mutation of what must be a frozen dev fit.
        self.cost_profile = MappingProxyType(costs)

    def _reward(self, requested: int, emitted: int) -> float:
        return emitted / self.cost_profile[requested]
