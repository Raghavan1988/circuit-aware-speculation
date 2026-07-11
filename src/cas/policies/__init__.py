"""Pure-Python adaptive length policies for speculative decoding.

Reproduce with::

    PYTHONPATH=src python -m pytest tests/test_policies.py tests/test_survival.py -q
"""

from .bandit import UCBSpecNaive, UCBSpecPolicy
from .entropy_stop import EntropyStopRule, StopContext, StopRule
from .prior_bandit import AcceptedLengthPrior, PriorBanditDecision, ProbePriorUCB
from .recent_acceptance import RecentAcceptancePolicy
from .survival import (
    ConditionalSurvivalModel,
    ConditionalTarget,
    CounterfactualLabels,
    GroupedPlattResult,
    LinearScoreHead,
    PlattScaler,
    SurvivalController,
    SurvivalDecision,
    expected_by_action,
    expected_emitted_tokens,
    fit_grouped_platt,
    fit_platt_scaler,
    grouped_fold_ids,
    select_utility_action,
    survival_curve,
    utility_by_action,
)

__all__ = [
    "AcceptedLengthPrior",
    "ConditionalSurvivalModel",
    "ConditionalTarget",
    "CounterfactualLabels",
    "EntropyStopRule",
    "GroupedPlattResult",
    "LinearScoreHead",
    "PlattScaler",
    "PriorBanditDecision",
    "ProbePriorUCB",
    "RecentAcceptancePolicy",
    "StopContext",
    "StopRule",
    "SurvivalController",
    "SurvivalDecision",
    "UCBSpecNaive",
    "UCBSpecPolicy",
    "expected_by_action",
    "expected_emitted_tokens",
    "fit_grouped_platt",
    "fit_platt_scaler",
    "grouped_fold_ids",
    "select_utility_action",
    "survival_curve",
    "utility_by_action",
]
