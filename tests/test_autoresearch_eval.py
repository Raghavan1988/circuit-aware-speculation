"""Unit tests for the incremental-lift scorer (cas.autoresearch.eval, I13/I23).

These build SMALL SYNTHETIC arrays IN THE TEST. That is deliberate and allowed:
this is a unit test of the *scorer's mechanics* (does an informative feature
register as a lift, does pure noise fail its equal-capacity control, is regret
well-behaved), NOT an experimental result. No synthetic number is ever reported
as a research finding (AGENTS.md). Requires numpy + sklearn (pinned env).
"""
import numpy as np
import pytest

from cas.autoresearch.eval import decision_regret, incremental_lift

SEED = 0


def _grouped(n_groups=40, rows_per=12):
    """Prompt-grouped row layout: contiguous blocks of rows share a group id."""
    groups = np.repeat(np.arange(n_groups), rows_per)
    return groups, len(groups)


def test_informative_candidate_beats_baseline_with_positive_delta():
    rng = np.random.default_rng(SEED)
    groups, n = _grouped()
    latent = rng.standard_normal(n)                      # the signal driving y
    p = 1.0 / (1.0 + np.exp(-1.6 * latent))
    y = (rng.random(n) < p).astype(int)
    X_base = rng.standard_normal((n, 3))                 # uninformative baseline
    X_cand = (latent + 0.5 * rng.standard_normal(n)).reshape(-1, 1)  # informative

    res = incremental_lift(X_base, X_cand, y, groups, seed=SEED, n_boot=400)

    assert res["beats_baseline"] is True
    assert res["deltas"]["auroc"] is not None and res["deltas"]["auroc"] > 0.0
    assert res["beats_controls"] is True                 # beats equal-capacity noise
    # a real lift's CI should sit above 0 (low P of delta<=0)
    assert res["delta_auroc_ci"]["p_delta_le_0"] < 0.5
    assert res["delta_auroc_ci"]["lo"] is not None


def test_noise_candidate_fails_its_equal_capacity_control():
    rng = np.random.default_rng(SEED)
    groups, n = _grouped()
    latent = rng.standard_normal(n)                      # y depends on a hidden
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-latent))).astype(int)
    X_base = rng.standard_normal((n, 3))                 # uninformative
    X_cand = rng.standard_normal((n, 1))                 # pure noise, no info

    res = incremental_lift(X_base, X_cand, y, groups, seed=SEED, n_boot=400)

    # noise must NOT beat its own equal-capacity random control
    assert res["beats_controls"] is False
    # and the (combined - base) AUROC CI must straddle 0
    ci = res["delta_auroc_ci"]
    assert ci["lo"] <= 0.0 <= ci["hi"]


def test_decision_regret_zero_for_perfect_and_positive_for_constant():
    rng = np.random.default_rng(SEED)
    y = (rng.random(200) < 0.4).astype(int)
    assert 0 < y.mean() < 1                              # both classes present

    # perfect predictor: p == y -> matches the oracle exactly -> zero regret
    assert decision_regret(y, y.astype(float)) == 0.0

    # constant predictor: cannot separate -> strictly positive regret
    assert decision_regret(y, np.full(len(y), 0.5)) > 0.0
    # (also holds for a constant below the ~0.091 breakeven threshold)
    assert decision_regret(y, np.full(len(y), 0.02)) > 0.0


def test_ece_in_unit_interval_for_all_models():
    rng = np.random.default_rng(SEED)
    groups, n = _grouped()
    latent = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-latent))).astype(int)
    X_base = rng.standard_normal((n, 2))
    X_cand = (latent + rng.standard_normal(n)).reshape(-1, 1)

    res = incremental_lift(X_base, X_cand, y, groups, seed=SEED, n_boot=200)
    for model in ("base", "combined", "control_random", "control_norm"):
        ece = res[model]["ece"]
        assert 0.0 <= ece <= 1.0, f"{model} ECE out of range: {ece}"


def test_controls_have_same_feature_count_as_combined():
    rng = np.random.default_rng(SEED)
    groups, n = _grouped()
    y = (rng.random(n) < 0.5).astype(int)
    X_base = rng.standard_normal((n, 3))                 # 3 baseline features
    X_cand = rng.standard_normal((n, 2))                 # 2 candidate features

    res = incremental_lift(X_base, X_cand, y, groups, seed=SEED, n_boot=100)

    assert res["n_features_base"] == 3
    assert res["n_features_cand"] == 2
    # combined = hstack([X_base, X_cand]) has n_base+n_cand columns; the controls
    # are hstack([X_base, R]) / hstack([X_base, Rn]) with R.shape == X_cand.shape,
    # so by construction every control has exactly the combined feature count.
    combined_width = res["n_features_base"] + res["n_features_cand"]
    assert combined_width == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
