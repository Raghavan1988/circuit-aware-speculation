"""UCBSpec-style draft-length bandit for issue I09.

The implementation follows UCBSpec (arXiv:2505.15141), Algorithm 4/8, in its
round-robin cold start, reward ``Y = len(accepted + bonus output)``, empirical
means, and self-normalized confidence radius.  Documented deviations:

* Arms are draft lengths ``(skip, 1, 2, 3, 4, 6, 8)``, rather than alternative
  speculative-decoding hyperparameter specifications under a common maximum L.
* Greedy exact-match rounds supply deterministic realized rewards conditional
  on a prefix; the paper also treats sampling and broader hyperparameters.
* The finite generation cap is enforced by the surrounding engine, not by this
  pure selector.  This module exposes resettable per-request state and does not
  claim the paper's stopping-time guarantee for the changed arm definition.
* Ties are resolved by configured arm order for reproducibility.

The policy consumes the previous ``RoundTrace`` on the next call, avoiding an
engine callback or any edit to the Claude-owned decoding loop.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from cas.config import ACTION_LENGTHS


class UCBSpecPolicy:
    """Stateful callable implementation of the stochastic UCBSpec selector."""

    def __init__(
        self,
        *,
        actions: Sequence[int] = ACTION_LENGTHS,
        delta: float = 0.05,
        max_speculation_length: int | None = None,
    ) -> None:
        normalized = tuple(int(action) for action in actions)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("actions must be non-empty and unique")
        if any(action < 0 for action in normalized):
            raise ValueError("actions must be non-negative")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        maximum = max(normalized) if max_speculation_length is None else int(max_speculation_length)
        if maximum < max(normalized):
            raise ValueError("max_speculation_length cannot be below an action")
        self.actions = normalized
        self.delta = float(delta)
        self.max_speculation_length = maximum
        self.reset()

    def reset(self) -> None:
        self.counts = {action: 0 for action in self.actions}
        self.reward_sums = {action: 0.0 for action in self.actions}
        self._pending_action: int | None = None
        self._last_observed_round: int | None = None

    def _observe(self, round_trace) -> None:
        if self._pending_action is None:
            raise ValueError("last_round supplied before this policy selected an action")
        requested = int(round_trace.requested_action)
        if requested != self._pending_action:
            raise ValueError("last_round action does not match the policy's pending arm")
        accepted = int(round_trace.accepted_prefix_len)
        realized = int(round_trace.realized_draft_len)
        if realized < 0 or not 0 <= accepted <= realized:
            raise ValueError("invalid acceptance counts in last_round")
        reward = accepted + 1
        self.counts[requested] += 1
        self.reward_sums[requested] += reward
        self._last_observed_round = int(round_trace.round_id)
        self._pending_action = None

    def _ucb(self, action: int, pulls: int) -> float:
        n = self.counts[action]
        if n == 0:
            return math.inf
        mean = self.reward_sums[action] / n
        k = len(self.actions)
        log_arg = k * pulls**2 * math.sqrt(1 + n) / self.delta
        radius = (self.max_speculation_length / 2) * math.sqrt(
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
