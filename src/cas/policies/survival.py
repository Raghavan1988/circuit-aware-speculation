"""Pure-stdlib survival/hazard controller scaffolding ratified by D018.3.

Let ``A`` be the accepted-prefix length for one draft/verify round.  We use

``q_j(x) = P(A >= j | A >= j - 1, x)``

for positions ``j = 1, ..., 8``.  D018 uses "hazard" as shorthand, but ``q_j``
is mathematically a conditional *continuation* probability; the rejection
hazard is ``h_j = 1 - q_j``.  The survival probability is therefore

``S_0 = 1`` and ``S_k = product(q_j, j=1..k)``.

For action ``L``, nominal round yield is ``N_L = 1 + min(A, L)`` and

``E[N_L | x] = 1 + sum(S_k, k=1..L)``.

The controller chooses the shortest action maximizing
``E[N_L | x] / cost(L)``.  Action zero is ordinary target decoding, with
expected yield one.  Costs are mandatory frozen constructor inputs measured on
development traces; no default timings live here.

Counterfactual labels are valid only at the same pre-round state.  A row with
eight proposals labels all locked actions by truncating the first all-True
draft/target match prefix.  An observed rejection also labels every longer
action because each would stop at that rejection; only a shorter all-accepted
row is right-censored.  Labels must never be composed into a counterfactual
multi-round trajectory.

Terminal rounds need separate treatment in downstream fitting: EOS or the
generation cap can make actual emitted tokens shorter than nominal
``accepted + 1``.  ``CounterfactualLabels.from_token_ids`` therefore requires
an explicit terminal flag and rejects terminal/capped rows.

Every feature must exist before the current action is selected.  Canonical
same-round outcome and trace-field names are rejected; callers should use
explicit pre-round names (for example, ``previous_target_entropy_frontier``).
This timing check is conservative and does not make arbitrary renamed features
safe: dataset builders remain responsible for prompt-grouped, leakage-free
feature construction.

Grouped Platt calibration expects raw base-model scores produced without prompt
leakage (out-of-fold or from a disjoint development calibration split).  Grouped
calibration keeps prompt groups intact but cannot repair leakage already present
in the scores.

Reproduce with::

    PYTHONPATH=src python -m pytest tests/test_survival.py -q
"""
from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cas.config import ACTION_LENGTHS


FeatureMap = Mapping[str, float]
ScoreHead = Callable[[FeatureMap], float]

# Canonical trace fields that do not exist until after the current action has
# been chosen.  Pre-round variants should use explicit names such as
# ``previous_target_entropy_frontier`` or ``history_acceptance_ema``.
_POST_ACTION_FEATURE_NAMES = frozenset(
    {
        "accepted_prefix_len",
        "draft_entropy",
        "draft_top1_margin",
        "emitted_token_ids",
        "first_rejection_pos",
        "latency_ns",
        "proposed_token_ids",
        "realized_draft_len",
        "requested_action",
        "target_argmax_ids",
    }
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _exact_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        return int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _binary_labels(labels: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for label in labels:
        try:
            value = int(operator.index(label))
        except TypeError as error:
            raise ValueError("labels must be exact binary integers") from error
        if value not in (0, 1):
            raise ValueError("labels must be binary")
        normalized.append(value)
    return tuple(normalized)


def validate_pre_action_features(features: FeatureMap) -> None:
    """Reject non-float or canonically post-action controller features."""

    for name, raw_value in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("feature names must be non-empty strings")
        if name in _POST_ACTION_FEATURE_NAMES:
            raise ValueError(f"feature {name!r} is unavailable before the action")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"feature {name!r} must be finite")


def _normalize_actions(actions: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(
        sorted(_exact_int(action, name="action") for action in actions)
    )
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("actions must be non-empty and unique")
    if any(action < 0 for action in normalized):
        raise ValueError("actions must be non-negative")
    return normalized


def _freeze_cost_profile(
    cost_profile: Mapping[int, float], actions: Sequence[int]
) -> Mapping[int, float]:
    costs: dict[int, float] = {}
    for action in actions:
        if action not in cost_profile:
            raise ValueError(f"cost_profile is missing action {action}")
        cost = float(cost_profile[action])
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("cost_profile values must be finite and positive")
        costs[action] = cost
    return MappingProxyType(costs)


@dataclass(frozen=True)
class ConditionalTarget:
    """One at-risk binary target for a 1-indexed continuation head."""

    position: int
    label: int

    def __post_init__(self) -> None:
        if self.position < 1:
            raise ValueError("conditional target position must be >= 1")
        if self.label not in (0, 1):
            raise ValueError("conditional target label must be binary")


@dataclass(frozen=True)
class CounterfactualLabels:
    """Per-round first-prefix labels derived from D018 trace fields."""

    matches: tuple[bool, ...]
    observed_length: int
    accepted_prefix_len: int

    def __post_init__(self) -> None:
        observed_length = _exact_int(self.observed_length, name="observed_length")
        accepted_prefix_len = _exact_int(
            self.accepted_prefix_len, name="accepted_prefix_len"
        )
        if any(not isinstance(match, bool) for match in self.matches):
            raise ValueError("matches must contain booleans")
        if observed_length != len(self.matches):
            raise ValueError("observed_length must equal the number of matches")
        accepted = 0
        for match in self.matches:
            if not match:
                break
            accepted += 1
        if accepted_prefix_len != accepted:
            raise ValueError("accepted_prefix_len must equal the first-True prefix")

    @classmethod
    def from_token_ids(
        cls,
        proposed_token_ids: Sequence[int],
        target_argmax_ids: Sequence[int],
        *,
        is_terminal: bool | None = None,
    ) -> "CounterfactualLabels":
        if is_terminal is None:
            raise ValueError("is_terminal must be supplied explicitly")
        if not isinstance(is_terminal, bool):
            raise ValueError("is_terminal must be boolean")
        if is_terminal:
            raise ValueError(
                "terminal/capped rounds are excluded from nominal-yield fitting"
            )
        proposed = tuple(
            _exact_int(token, name="proposed token id")
            for token in proposed_token_ids
        )
        target = tuple(
            _exact_int(token, name="target token id")
            for token in target_argmax_ids
        )
        if len(target) != len(proposed) + 1:
            raise ValueError(
                "target_argmax_ids must contain one entry per proposal plus the bonus"
            )

        matches = tuple(draft == target[i] for i, draft in enumerate(proposed))
        accepted = 0
        for match in matches:
            if not match:
                break
            accepted += 1
        return cls(matches, len(proposed), accepted)

    def accepted_for(self, action: int) -> int:
        action = _exact_int(action, name="action")
        if action < 0:
            raise ValueError("action must be non-negative")
        if (
            action > self.observed_length
            and self.accepted_prefix_len == self.observed_length
        ):
            raise ValueError(
                f"action {action} is censored by observed length {self.observed_length}"
            )
        return min(self.accepted_prefix_len, action)

    def accepted_by_action(
        self, actions: Sequence[int] = ACTION_LENGTHS
    ) -> dict[int, int]:
        normalized = _normalize_actions(actions)
        return {action: self.accepted_for(action) for action in normalized}

    def conditional_targets(self) -> tuple[ConditionalTarget, ...]:
        """Return at-risk labels only; ignore matches after first rejection."""

        targets = [ConditionalTarget(position, 1)
                   for position in range(1, self.accepted_prefix_len + 1)]
        if self.accepted_prefix_len < self.observed_length:
            targets.append(ConditionalTarget(self.accepted_prefix_len + 1, 0))
        return tuple(targets)


def survival_curve(conditional_probabilities: Sequence[float]) -> tuple[float, ...]:
    """Return ``(S_0, ..., S_m)`` from continuation probabilities."""

    survival = [1.0]
    current = 1.0
    for probability in conditional_probabilities:
        q = float(probability)
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("conditional probabilities must be finite and in [0, 1]")
        current *= q
        survival.append(current)
    return tuple(survival)


def expected_emitted_tokens(survival: Sequence[float], action: int) -> float:
    """Compute nominal ``E[1 + min(A, L)]`` for one action."""

    action = _exact_int(action, name="action")
    values = tuple(float(value) for value in survival)
    if action < 0:
        raise ValueError("action must be non-negative")
    if not values or action >= len(values):
        raise ValueError("survival curve does not cover the requested action")
    if not math.isclose(values[0], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("survival curve must begin at S_0 = 1")
    previous = values[0]
    for value in values:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("survival values must be finite and in [0, 1]")
        if value > previous + 1e-12:
            raise ValueError("survival curve must be non-increasing")
        previous = value
    return 1.0 + sum(values[1:action + 1])


def expected_by_action(
    survival: Sequence[float], actions: Sequence[int] = ACTION_LENGTHS
) -> dict[int, float]:
    normalized = _normalize_actions(actions)
    return {action: expected_emitted_tokens(survival, action) for action in normalized}


def utility_by_action(
    survival: Sequence[float],
    cost_profile: Mapping[int, float],
    actions: Sequence[int] = ACTION_LENGTHS,
) -> dict[int, float]:
    normalized = _normalize_actions(actions)
    costs = _freeze_cost_profile(cost_profile, normalized)
    expected = expected_by_action(survival, normalized)
    return {action: expected[action] / costs[action] for action in normalized}


def select_utility_action(utilities: Mapping[int, float]) -> int:
    """Select maximum utility, breaking exact ties toward the shortest action."""

    if not utilities:
        raise ValueError("utilities must be non-empty")
    normalized: dict[int, float] = {}
    for action, utility in utilities.items():
        action_int = _exact_int(action, name="action")
        value = float(utility)
        if action_int < 0 or not math.isfinite(value):
            raise ValueError("utilities require non-negative actions and finite values")
        normalized[action_int] = value
    return max(sorted(normalized), key=normalized.__getitem__)


@dataclass(frozen=True)
class LinearScoreHead:
    """Feature-name-based linear score; required features are explicit."""

    weights: Mapping[str, float]
    intercept: float = 0.0

    def __post_init__(self) -> None:
        frozen: dict[str, float] = {}
        for name, weight in self.weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            value = float(weight)
            if not math.isfinite(value):
                raise ValueError("feature weights must be finite")
            frozen[name] = value
        intercept = float(self.intercept)
        if not math.isfinite(intercept):
            raise ValueError("intercept must be finite")
        object.__setattr__(self, "weights", MappingProxyType(frozen))
        object.__setattr__(self, "intercept", intercept)

    def __call__(self, features: FeatureMap) -> float:
        validate_pre_action_features(features)
        missing = [name for name in self.weights if name not in features]
        if missing:
            raise ValueError(f"missing required features: {', '.join(sorted(missing))}")
        score = self.intercept
        for name, weight in self.weights.items():
            value = float(features[name])
            if not math.isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")
            score += weight * value
        return score


@dataclass(frozen=True)
class PlattScaler:
    """Two-parameter sigmoid calibrator over a raw scalar score."""

    slope: float = 1.0
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.slope) or not math.isfinite(self.intercept):
            raise ValueError("Platt parameters must be finite")

    def __call__(self, score: float) -> float:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("calibration score must be finite")
        return _sigmoid(self.slope * score + self.intercept)

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "PlattScaler":
        return cls(float(values["slope"]), float(values["intercept"]))


class ConditionalSurvivalModel:
    """Position-specific raw score heads followed by Platt calibration."""

    def __init__(
        self,
        heads: Sequence[ScoreHead],
        calibrators: Sequence[PlattScaler] | None = None,
    ) -> None:
        normalized_heads = tuple(heads)
        if not normalized_heads:
            raise ValueError("at least one continuation head is required")
        if not all(callable(head) for head in normalized_heads):
            raise ValueError("every continuation head must be callable")
        if calibrators is None:
            normalized_calibrators = tuple(PlattScaler() for _ in normalized_heads)
        else:
            normalized_calibrators = tuple(calibrators)
        if len(normalized_calibrators) != len(normalized_heads):
            raise ValueError("one calibrator is required per continuation head")
        self.heads = normalized_heads
        self.calibrators = normalized_calibrators

    @property
    def max_length(self) -> int:
        return len(self.heads)

    def conditional_probabilities(self, features: FeatureMap) -> tuple[float, ...]:
        validate_pre_action_features(features)
        return tuple(
            calibrator(head(features))
            for head, calibrator in zip(self.heads, self.calibrators)
        )

    def survival(self, features: FeatureMap) -> tuple[float, ...]:
        return survival_curve(self.conditional_probabilities(features))


@dataclass(frozen=True)
class SurvivalDecision:
    action: int
    conditional_probabilities: tuple[float, ...]
    survival: tuple[float, ...]
    expected_emitted_by_action: Mapping[int, float]
    utility_by_action: Mapping[int, float]


class SurvivalController:
    """Feature-agnostic utility controller; no engine or trace dependency."""

    def __init__(
        self,
        model: ConditionalSurvivalModel,
        cost_profile: Mapping[int, float],
        *,
        actions: Sequence[int] = ACTION_LENGTHS,
    ) -> None:
        normalized = _normalize_actions(actions)
        if max(normalized) > model.max_length:
            raise ValueError("survival model does not cover every requested action")
        self.model = model
        self.actions = normalized
        self.cost_profile = _freeze_cost_profile(cost_profile, normalized)

    def evaluate(self, features: FeatureMap) -> SurvivalDecision:
        conditional = self.model.conditional_probabilities(features)
        survival = survival_curve(conditional)
        expected = expected_by_action(survival, self.actions)
        utilities = {
            action: expected[action] / self.cost_profile[action]
            for action in self.actions
        }
        action = select_utility_action(utilities)
        return SurvivalDecision(
            action=action,
            conditional_probabilities=conditional,
            survival=survival,
            expected_emitted_by_action=MappingProxyType(expected),
            utility_by_action=MappingProxyType(utilities),
        )

    def select_action(self, features: FeatureMap) -> int:
        return self.evaluate(features).action

    def __call__(self, features: FeatureMap) -> int:
        return self.select_action(features)


def _platt_objective(
    scores: Sequence[float], labels: Sequence[int], slope: float, intercept: float, l2: float
) -> float:
    total = 0.5 * l2 * (slope * slope + intercept * intercept)
    for score, label in zip(scores, labels):
        logit = slope * score + intercept
        total += max(logit, 0.0) - label * logit + math.log1p(math.exp(-abs(logit)))
    return total


def fit_platt_scaler(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    l2: float = 1e-6,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> PlattScaler:
    """Fit a two-parameter calibrator with damped Newton updates."""

    x = tuple(float(score) for score in scores)
    y = _binary_labels(labels)
    if not x or len(x) != len(y):
        raise ValueError("scores and labels must be non-empty and equally sized")
    if any(not math.isfinite(score) for score in x):
        raise ValueError("scores must be finite")
    if any(label not in (0, 1) for label in y):
        raise ValueError("labels must be binary")
    if len(set(y)) != 2:
        raise ValueError("Platt fitting requires both label classes")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("l2 must be finite and positive")
    if max_iter < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid optimizer controls")

    positives = sum(y)
    negatives = len(y) - positives
    slope = 0.0
    intercept = math.log((positives + 1.0) / (negatives + 1.0))

    for _ in range(max_iter):
        grad_s = l2 * slope
        grad_i = l2 * intercept
        h_ss = l2
        h_si = 0.0
        h_ii = l2
        for score, label in zip(x, y):
            probability = _sigmoid(slope * score + intercept)
            residual = probability - label
            weight = probability * (1.0 - probability)
            grad_s += residual * score
            grad_i += residual
            h_ss += weight * score * score
            h_si += weight * score
            h_ii += weight
        if max(abs(grad_s), abs(grad_i)) <= tolerance:
            break

        determinant = h_ss * h_ii - h_si * h_si
        if determinant <= 0 or not math.isfinite(determinant):
            raise ValueError("Platt optimizer encountered a singular Hessian")
        step_s = (h_ii * grad_s - h_si * grad_i) / determinant
        step_i = (-h_si * grad_s + h_ss * grad_i) / determinant
        old_objective = _platt_objective(x, y, slope, intercept, l2)

        scale = 1.0
        accepted_step = False
        for _ in range(30):
            candidate_s = slope - scale * step_s
            candidate_i = intercept - scale * step_i
            candidate_objective = _platt_objective(
                x, y, candidate_s, candidate_i, l2
            )
            if candidate_objective <= old_objective:
                slope, intercept = candidate_s, candidate_i
                accepted_step = True
                break
            scale *= 0.5
        if not accepted_step:
            break
        if max(abs(scale * step_s), abs(scale * step_i)) <= tolerance:
            break
    return PlattScaler(slope, intercept)


def grouped_fold_ids(
    group_ids: Sequence[Any],
    *,
    labels: Sequence[int] | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[int, ...]:
    """Assign whole groups to deterministic, stratified balanced folds.

    When ``labels`` are omitted, balance only the number of groups.  Calibration
    fitting always supplies labels so class and row counts are balanced without
    splitting a prompt group.
    """

    groups = tuple(group_ids)
    if not groups:
        raise ValueError("group_ids must be non-empty")
    try:
        unique = list(dict.fromkeys(groups))
    except TypeError as error:
        raise ValueError("group ids must be hashable") from error
    if n_folds < 2 or n_folds > len(unique):
        raise ValueError("n_folds must be between 2 and the number of groups")

    def order_key(group: Any) -> bytes:
        payload = f"{seed}:{group!r}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    if labels is None:
        ordered = sorted(unique, key=order_key)
        assignment = {
            group: index % n_folds for index, group in enumerate(ordered)
        }
        return tuple(assignment[group] for group in groups)

    y = _binary_labels(labels)
    if len(y) != len(groups):
        raise ValueError("labels and group_ids must be equally sized")
    stats = {group: [0, 0, 0] for group in unique}  # rows, positives, negatives
    for group, label in zip(groups, y):
        stats[group][0] += 1
        stats[group][1] += label
        stats[group][2] += 1 - label

    ordered = sorted(
        unique,
        key=lambda group: (
            -max(stats[group][1], stats[group][2]),
            -stats[group][0],
            order_key(group),
        ),
    )
    target_rows = len(groups) / n_folds
    target_positives = sum(y) / n_folds
    target_negatives = (len(y) - sum(y)) / n_folds
    fold_stats = [[0, 0, 0] for _ in range(n_folds)]
    fold_group_counts = [0] * n_folds
    assignment: dict[Any, int] = {}

    def balance_score(candidate_fold: int, group: Any) -> float:
        score = 0.0
        for fold, current in enumerate(fold_stats):
            add = stats[group] if fold == candidate_fold else (0, 0, 0)
            rows = current[0] + add[0]
            positives = current[1] + add[1]
            negatives = current[2] + add[2]
            score += 0.25 * ((rows - target_rows) / max(target_rows, 1.0)) ** 2
            score += (
                (positives - target_positives) / max(target_positives, 1.0)
            ) ** 2
            score += (
                (negatives - target_negatives) / max(target_negatives, 1.0)
            ) ** 2
        return score

    for group in ordered:
        fold = min(
            range(n_folds),
            key=lambda candidate: (
                balance_score(candidate, group),
                fold_group_counts[candidate],
                candidate,
            ),
        )
        assignment[group] = fold
        fold_group_counts[fold] += 1
        for index in range(3):
            fold_stats[fold][index] += stats[group][index]
    return tuple(assignment[group] for group in groups)


@dataclass(frozen=True)
class GroupedPlattResult:
    """Deployment scaler plus prompt-grouped out-of-fold calibration outputs."""

    final_scaler: PlattScaler
    oof_probabilities: tuple[float, ...]
    fold_ids: tuple[int, ...]
    fold_scalers: tuple[PlattScaler, ...]


def fit_grouped_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    group_ids: Sequence[Any],
    *,
    n_folds: int = 5,
    seed: int = 0,
    l2: float = 1e-6,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> GroupedPlattResult:
    """Fit grouped-fold OOF scalers and one all-development deployment scaler."""

    x = tuple(float(score) for score in scores)
    y = _binary_labels(labels)
    groups = tuple(group_ids)
    if not x or len(x) != len(y) or len(x) != len(groups):
        raise ValueError("scores, labels, and group_ids must be equally sized")
    folds = grouped_fold_ids(groups, labels=y, n_folds=n_folds, seed=seed)

    oof: list[float | None] = [None] * len(x)
    fold_scalers: list[PlattScaler] = []
    for fold in range(n_folds):
        train_indices = [index for index, fold_id in enumerate(folds) if fold_id != fold]
        held_indices = [index for index, fold_id in enumerate(folds) if fold_id == fold]
        if set(y[index] for index in train_indices) != {0, 1}:
            raise ValueError(
                "each grouped calibration training fold requires both classes; "
                "reduce n_folds or supply more prompt groups per class"
            )
        scaler = fit_platt_scaler(
            [x[index] for index in train_indices],
            [y[index] for index in train_indices],
            l2=l2,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        fold_scalers.append(scaler)
        for index in held_indices:
            oof[index] = scaler(x[index])

    if any(value is None for value in oof):
        raise RuntimeError("grouped calibration failed to cover every example")
    final_scaler = fit_platt_scaler(
        x, y, l2=l2, max_iter=max_iter, tolerance=tolerance
    )
    return GroupedPlattResult(
        final_scaler=final_scaler,
        oof_probabilities=tuple(float(value) for value in oof),
        fold_ids=folds,
        fold_scalers=tuple(fold_scalers),
    )
