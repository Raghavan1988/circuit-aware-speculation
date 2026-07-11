import math

import pytest

from cas.policies import (
    AcceptedLengthPrior,
    ConditionalSurvivalModel,
    CounterfactualLabels,
    LinearScoreHead,
    PlattScaler,
    ProbePriorUCB,
    SurvivalController,
    expected_emitted_tokens,
    fit_grouped_platt,
    fit_platt_scaler,
    grouped_fold_ids,
    select_utility_action,
    survival_curve,
    utility_by_action,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _constant_model(*probabilities: float) -> ConditionalSurvivalModel:
    heads = [LinearScoreHead({}, _logit(probability)) for probability in probabilities]
    return ConditionalSurvivalModel(heads)


def test_counterfactual_labels_stop_at_first_rejection_and_cover_shorter_actions():
    proposed = tuple(range(1, 9))
    target = (1, 2, 99, 4, 5, 6, 7, 8, 1234)
    labels = CounterfactualLabels.from_token_ids(
        proposed, target, is_terminal=False
    )

    assert labels.matches == (True, True, False, True, True, True, True, True)
    assert labels.accepted_prefix_len == 2
    assert labels.accepted_by_action() == {
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        6: 2,
        8: 2,
    }
    assert [(target.position, target.label) for target in labels.conditional_targets()] == [
        (1, 1),
        (2, 1),
        (3, 0),
    ]


def test_counterfactual_labels_reject_censored_longer_action_and_bad_bonus_shape():
    labels = CounterfactualLabels.from_token_ids(
        (1, 2), (1, 2, 3), is_terminal=False
    )
    with pytest.raises(ValueError, match="censored"):
        labels.accepted_for(3)
    with pytest.raises(ValueError, match="plus the bonus"):
        CounterfactualLabels.from_token_ids(
            (1, 2), (1, 2), is_terminal=False
        )
    with pytest.raises(ValueError, match="first-True prefix"):
        CounterfactualLabels((True, False), observed_length=2, accepted_prefix_len=2)


def test_observed_rejection_labels_longer_actions_but_terminal_rows_are_rejected():
    labels = CounterfactualLabels.from_token_ids(
        (1, 2), (1, 99, 7), is_terminal=False
    )
    assert labels.accepted_for(8) == 1
    with pytest.raises(ValueError, match="supplied explicitly"):
        CounterfactualLabels.from_token_ids((1,), (1, 2))
    with pytest.raises(ValueError, match="terminal/capped"):
        CounterfactualLabels.from_token_ids((1,), (1, 2), is_terminal=True)
    with pytest.raises(ValueError, match="must be an integer"):
        CounterfactualLabels.from_token_ids((1.5,), (1, 2), is_terminal=False)


def test_survival_curve_is_monotone_and_expected_yield_matches_formula():
    survival = survival_curve((0.8, 0.5, 0.25))
    assert survival == pytest.approx((1.0, 0.8, 0.4, 0.1))
    assert all(left >= right for left, right in zip(survival, survival[1:]))
    assert expected_emitted_tokens(survival, 0) == 1.0
    assert expected_emitted_tokens(survival, 3) == pytest.approx(2.3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        survival_curve((float("nan"),))


def test_high_rejection_hazard_selects_skip_and_costs_are_frozen():
    # q=0 means rejection hazard h=1 at every position.
    model = ConditionalSurvivalModel([LinearScoreHead({}, -1000.0) for _ in range(8)])
    source_costs = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 6: 7.0, 8: 9.0}
    controller = SurvivalController(model, source_costs)
    source_costs[0] = 999.0

    decision = controller.evaluate({})
    assert decision.action == 0
    assert decision.expected_emitted_by_action[0] == 1.0
    assert controller.cost_profile[0] == 1.0


def test_utility_threshold_tie_prefers_skip_and_crossing_selects_draft():
    costs = {0: 1.0, 1: 1.5}
    tie = SurvivalController(_constant_model(0.5), costs, actions=(0, 1))
    above = SurvivalController(_constant_model(0.5001), costs, actions=(0, 1))
    below = SurvivalController(_constant_model(0.4999), costs, actions=(0, 1))

    assert tie.select_action({}) == 0
    assert above.select_action({}) == 1
    assert below.select_action({}) == 0
    assert select_utility_action({1: 1.0, 0: 1.0}) == 0


def test_feature_named_heads_allow_extra_features_but_require_declared_inputs():
    head = LinearScoreHead({"entropy": -1.0, "probe": 2.0}, intercept=0.25)
    assert head({"entropy": 0.5, "probe": 0.25, "unused": 999.0}) == 0.25
    with pytest.raises(ValueError, match="probe"):
        head({"entropy": 0.5})
    with pytest.raises(ValueError, match="unavailable before the action"):
        head({"entropy": 0.5, "probe": 0.25, "accepted_prefix_len": 1.0})


def test_controller_requires_positional_head_coverage_for_action_eight():
    model = ConditionalSurvivalModel([LinearScoreHead({}, 0.0) for _ in range(7)])
    with pytest.raises(ValueError, match="does not cover"):
        SurvivalController(
            model,
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 6: 1.0, 8: 1.0},
        )


def test_utility_helpers_validate_costs_and_survival_coverage():
    survival = survival_curve((0.5,))
    assert utility_by_action(survival, {0: 1.0, 1: 2.0}, (0, 1)) == {
        0: 1.0,
        1: 0.75,
    }
    with pytest.raises(ValueError, match="missing action 1"):
        utility_by_action(survival, {0: 1.0}, (0, 1))
    with pytest.raises(ValueError, match="does not cover"):
        expected_emitted_tokens(survival, 2)


def test_platt_calibration_round_trip_and_extreme_score_stability():
    fitted = fit_platt_scaler((-2.0, -1.0, 1.0, 2.0), (0, 0, 1, 1), l2=0.1)
    restored = PlattScaler.from_dict(fitted.to_dict())

    assert restored(-2.0) == pytest.approx(fitted(-2.0))
    assert restored(2.0) == pytest.approx(fitted(2.0))
    assert fitted(-2.0) < fitted(2.0)
    assert 0.0 <= PlattScaler()(1000.0) <= 1.0
    assert 0.0 <= PlattScaler()(-1000.0) <= 1.0
    with pytest.raises(ValueError, match="exact binary integers"):
        fit_platt_scaler((-1.0, 1.0), (0.9, 1.9))


def test_grouped_platt_keeps_prompt_groups_intact_and_is_deterministic():
    scores = (-2.0, 2.0, -1.5, 1.5, -1.0, 1.0, -0.5, 0.5)
    labels = (0, 1, 0, 1, 0, 1, 0, 1)
    groups = ("p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3")

    fold_ids = grouped_fold_ids(groups, n_folds=2, seed=7)
    assert fold_ids[0] == fold_ids[1]
    assert fold_ids[2] == fold_ids[3]
    assert fold_ids == grouped_fold_ids(groups, n_folds=2, seed=7)

    result = fit_grouped_platt(scores, labels, groups, n_folds=2, seed=7, l2=0.1)
    assert result.fold_ids == fold_ids
    assert len(result.oof_probabilities) == len(scores)
    assert len(result.fold_scalers) == 2
    assert all(0.0 <= probability <= 1.0 for probability in result.oof_probabilities)
    restored = PlattScaler.from_dict(result.final_scaler.to_dict())
    assert restored(0.25) == pytest.approx(result.final_scaler(0.25))


def test_grouped_platt_stratifies_pure_class_prompt_groups():
    scores = (2.0, 1.0, -2.0, -1.0)
    labels = (1, 1, 0, 0)
    groups = ("p0", "p1", "n0", "n1")

    result = fit_grouped_platt(
        scores, labels, groups, n_folds=2, seed=0, l2=0.1
    )
    for fold in range(2):
        training_labels = {
            label
            for label, fold_id in zip(labels, result.fold_ids)
            if fold_id != fold
        }
        assert training_labels == {0, 1}


def test_probe_prior_ucb_uses_confident_prior_and_abstains_below_threshold():
    costs = {0: 1.0, 8: 1.0}
    policy = ProbePriorUCB(
        costs,
        actions=(0, 8),
        confidence_threshold=0.6,
        prior_strength=10.0,
        exploration=0.0,
    )
    long_prior = AcceptedLengthPrior((0.0,) * 8 + (1.0,), confidence=0.9)
    low_confidence = AcceptedLengthPrior((0.0,) * 8 + (1.0,), confidence=0.59)

    confident = policy.evaluate(long_prior)
    assert confident.used_prior
    assert confident.action == 8
    abstained = policy.evaluate(low_confidence)
    assert not abstained.used_prior
    assert abstained.action == 0


def test_probe_prior_ucb_low_confidence_fallback_matches_pure_history():
    costs = {0: 1.0, 2: 4.0}
    with_prior = ProbePriorUCB(costs, actions=(0, 2), confidence_threshold=0.5)
    history_only = ProbePriorUCB(costs, actions=(0, 2), confidence_threshold=0.5)
    low_prior = AcceptedLengthPrior((0.0, 0.0, 1.0), confidence=0.1)

    for _ in range(8):
        action_with_prior = with_prior.select_action(low_prior)
        action_history = history_only.select_action()
        assert action_with_prior == action_history
        accepted = 0 if action_with_prior == 0 else 1
        with_prior.update(
            action_with_prior,
            accepted_prefix_len=accepted,
            realized_draft_len=action_with_prior,
        )
        history_only.update(
            action_history,
            accepted_prefix_len=accepted,
            realized_draft_len=action_history,
        )


def test_probe_prior_ucb_updates_realized_history_and_validates_prior():
    policy = ProbePriorUCB({0: 1.0, 2: 5.0}, actions=(0, 2))
    assert policy.select_action() == 0
    policy.update(0, accepted_prefix_len=0, realized_draft_len=0, emitted_tokens=1)
    assert policy.counts[0] == 1
    assert policy.reward_sums[0] == 1.0
    with pytest.raises(ValueError, match="sum to one"):
        AcceptedLengthPrior((0.2, 0.2), confidence=1.0)
    with pytest.raises(ValueError, match="terminal/capped"):
        policy.update(2, accepted_prefix_len=1, realized_draft_len=2, emitted_tokens=1)
