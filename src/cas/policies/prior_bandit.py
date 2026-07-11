"""Probe-as-prior UCB interface (round-2 C3 stretch; design only).

This module does not fit a probe and is not wired into the engine.  It defines
how a calibrated predicted accepted-length distribution can act as contextual
pseudo-observations for a seven-action UCB selector.  If the prior's declared
confidence is below a frozen threshold, its pseudo-count is exactly zero and
selection falls back to pure realized-history UCB.

The realized update is latency-aware: nominal emitted tokens divided by a
mandatory development-measured action cost.  Terminal/capped rounds whose
actual yield is shorter than ``accepted + 1`` must be excluded or handled by a
future explicit terminal-aware interface; they are never silently relabeled.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from cas.config import ACTION_LENGTHS


def _normalize_actions(actions: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(sorted(int(operator.index(action)) for action in actions))
    except TypeError as error:
        raise ValueError("actions must be exact integers") from error
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("actions must be non-empty and unique")
    if any(action < 0 for action in normalized):
        raise ValueError("actions must be non-negative")
    return normalized


@dataclass(frozen=True)
class AcceptedLengthPrior:
    """Calibrated probabilities for accepted lengths ``0..max_length``."""

    probabilities: tuple[float, ...]
    confidence: float

    def __post_init__(self) -> None:
        probabilities = tuple(float(value) for value in self.probabilities)
        if not probabilities:
            raise ValueError("prior probabilities must be non-empty")
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("prior probabilities must be finite and non-negative")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("prior probabilities must sum to one")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("prior confidence must be finite and in [0, 1]")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "confidence", confidence)

    @property
    def max_length(self) -> int:
        return len(self.probabilities) - 1

    def expected_emitted(self, action: int) -> float:
        try:
            action = int(operator.index(action))
        except TypeError as error:
            raise ValueError("action must be an exact integer") from error
        if not 0 <= action <= self.max_length:
            raise ValueError("prior does not cover requested action")
        return sum(
            probability * (1 + min(accepted, action))
            for accepted, probability in enumerate(self.probabilities)
        )


@dataclass(frozen=True)
class PriorBanditDecision:
    action: int
    used_prior: bool
    prior_weight: float
    scores: Mapping[int, float]


class ProbePriorUCB:
    """UCB selector with confidence-gated contextual pseudo-observations."""

    def __init__(
        self,
        cost_profile: Mapping[int, float],
        *,
        actions: Sequence[int] = ACTION_LENGTHS,
        confidence_threshold: float = 0.5,
        prior_strength: float = 2.0,
        exploration: float = 2.0,
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
        threshold = float(confidence_threshold)
        strength = float(prior_strength)
        exploration_value = float(exploration)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("prior_strength must be finite and positive")
        if not math.isfinite(exploration_value) or exploration_value < 0:
            raise ValueError("exploration must be finite and non-negative")

        self.actions = normalized
        self.cost_profile = MappingProxyType(costs)
        self.confidence_threshold = threshold
        self.prior_strength = strength
        self.exploration = exploration_value
        self._reward_bound = max((action + 1) / costs[action] for action in normalized)
        self.reset()

    def reset(self) -> None:
        self.counts = {action: 0 for action in self.actions}
        self.reward_sums = {action: 0.0 for action in self.actions}

    def update(
        self,
        action: int,
        *,
        accepted_prefix_len: int,
        realized_draft_len: int,
        emitted_tokens: int | None = None,
    ) -> None:
        try:
            action = int(operator.index(action))
            accepted = int(operator.index(accepted_prefix_len))
            realized = int(operator.index(realized_draft_len))
        except TypeError as error:
            raise ValueError("realized feedback must use exact integers") from error
        if action not in self.counts:
            raise ValueError(f"unknown action {action}")
        if not 0 <= accepted <= realized <= action:
            raise ValueError("invalid realized acceptance feedback")
        nominal_emitted = accepted + 1
        if emitted_tokens is not None:
            try:
                emitted = int(operator.index(emitted_tokens))
            except TypeError as error:
                raise ValueError("emitted_tokens must be an exact integer") from error
            if emitted != nominal_emitted:
                raise ValueError(
                    "terminal/capped feedback requires an explicit future interface"
                )
        reward = nominal_emitted / self.cost_profile[action]
        self.counts[action] += 1
        self.reward_sums[action] += reward

    def evaluate(self, prior: AcceptedLengthPrior | None = None) -> PriorBanditDecision:
        use_prior = prior is not None and prior.confidence >= self.confidence_threshold
        if prior is not None and prior.max_length < max(self.actions):
            raise ValueError("accepted-length prior does not cover every action")
        prior_weight = self.prior_strength * prior.confidence if use_prior else 0.0
        total_effective = sum(self.counts.values()) + prior_weight * len(self.actions)
        log_term = math.log(max(2.0, total_effective + 1.0))

        scores: dict[int, float] = {}
        for action in self.actions:
            count = self.counts[action]
            if prior_weight:
                prior_reward = prior.expected_emitted(action) / self.cost_profile[action]
            else:
                prior_reward = 0.0
            effective_count = count + prior_weight
            if effective_count == 0:
                score = math.inf
            else:
                mean = (
                    self.reward_sums[action] + prior_weight * prior_reward
                ) / effective_count
                radius = self._reward_bound * math.sqrt(
                    self.exploration * log_term / effective_count
                )
                score = mean + radius
            scores[action] = score

        # Configured actions are normalized ascending, so max keeps the shortest
        # action on an exact score tie (the calibrate-and-abstain safe default).
        action = max(self.actions, key=scores.__getitem__)
        return PriorBanditDecision(
            action=action,
            used_prior=use_prior,
            prior_weight=prior_weight,
            scores=MappingProxyType(scores),
        )

    def select_action(self, prior: AcceptedLengthPrior | None = None) -> int:
        return self.evaluate(prior).action
