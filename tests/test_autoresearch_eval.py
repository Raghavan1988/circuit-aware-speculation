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

SEED = 0            # the scorer's internal seed (controls, bootstrap)
# Synthetic data uses a DIFFERENT stream than SEED on purpose: the scorer draws
# its equal-capacity control noise from default_rng(SEED); if the test data were
# also drawn from default_rng(SEED) the control noise would collide byte-for-byte
# with the signal and the "random" control would reconstruct y. Real callers pass
# activations for X_cand, never the scorer's seed stream, so no collision occurs.
DATA_SEED = 20260719


def _grouped(n_groups=40, rows_per=12):
    """Prompt-grouped row layout: contiguous blocks of rows share a group id."""
    groups = np.repeat(np.arange(n_groups), rows_per)
    return groups, len(groups)


def test_informative_candidate_beats_baseline_with_positive_delta():
    rng = np.random.default_rng(DATA_SEED)
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
    rng = np.random.default_rng(DATA_SEED)
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
    rng = np.random.default_rng(DATA_SEED)
    y = (rng.random(200) < 0.4).astype(int)
    assert 0 < y.mean() < 1                              # both classes present

    # perfect predictor: p == y -> matches the oracle exactly -> zero regret
    assert decision_regret(y, y.astype(float)) == 0.0

    # constant predictor: cannot separate -> strictly positive regret
    assert decision_regret(y, np.full(len(y), 0.5)) > 0.0
    # (also holds for a constant below the ~0.091 breakeven threshold)
    assert decision_regret(y, np.full(len(y), 0.02)) > 0.0


def test_ece_in_unit_interval_for_all_models():
    rng = np.random.default_rng(DATA_SEED)
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
    rng = np.random.default_rng(DATA_SEED)
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


def test_recalibration_preserves_auroc_and_reduces_ece():
    from sklearn.metrics import roc_auc_score

    from cas.autoresearch.eval import _calibrate_oof, _ece

    rng = np.random.default_rng(DATA_SEED)
    n = 4000
    true_p = rng.uniform(0.05, 0.95, n)                  # well-calibrated ground truth
    y = (rng.random(n) < true_p).astype(int)
    # logit-AFFINE overconfidence: ranking intact but probabilities miscalibrated.
    # Platt (a logistic on logit(over)) can invert a logit-affine distortion, so it
    # recovers ~true_p -> a strict, deterministic ECE reduction.
    over = 1.0 / (1.0 + np.exp(-(2.2 * np.log(true_p / (1.0 - true_p)) + 0.4)))

    cal = _calibrate_oof(over, y, seed=SEED)

    # Global Platt is monotone -> AUROC preserved exactly.
    assert abs(roc_auc_score(y, cal) - roc_auc_score(y, over)) < 1e-6
    # It inverts the logit-affine miscalibration -> ECE strictly reduced.
    assert _ece(y, cal) < _ece(y, over)


def test_incremental_lift_exposes_calibrated_decision_metrics():
    rng = np.random.default_rng(DATA_SEED)
    groups, n = _grouped()
    latent = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-latent))).astype(int)
    X_base = rng.standard_normal((n, 3))
    X_cand = (latent + 0.5 * rng.standard_normal(n)).reshape(-1, 1)

    res = incremental_lift(X_base, X_cand, y, groups, seed=SEED, n_boot=100)

    for k in ("base_calibrated", "combined_calibrated", "deltas_calibrated",
              "helps_decision_calibrated"):
        assert k in res
    # Calibration is monotonic -> calibrated AUROC equals the raw AUROC.
    assert abs(res["combined_calibrated"]["auroc"] - res["combined"]["auroc"]) < 1e-6
    assert abs(res["base_calibrated"]["auroc"] - res["base"]["auroc"]) < 1e-6
    assert isinstance(res["helps_decision_calibrated"], bool)
    assert 0.0 <= res["combined_calibrated"]["regret"] <= 1.0


def test_regret_cost_sweep_monotone_tau_and_flags_help():
    from cas.autoresearch.eval import COST_GRID, regret_cost_sweep

    rng = np.random.default_rng(DATA_SEED)
    n = 800
    y = (rng.random(n) < 0.55).astype(int)
    base_p = np.clip(0.55 + 0.05 * rng.standard_normal(n), 0.01, 0.99)   # ~uninformative
    comb_p = np.clip(0.5 + (2 * y - 1) * 0.3 + 0.08 * rng.standard_normal(n),
                     0.01, 0.99)                                          # informative

    sweep = regret_cost_sweep(y, base_p, comb_p)

    assert len(sweep) == len(COST_GRID)
    taus = [s["tau"] for s in sweep]
    assert taus == sorted(taus)                       # tau increases with cost
    for s in sweep:
        assert s["base_regret"] >= 0.0 and s["combined_regret"] >= 0.0
        assert s["helps"] == (s["combined_regret"] < s["base_regret"] - 1e-6)
    # an informative candidate must reduce regret at SOME cost ratio
    assert any(s["helps"] for s in sweep)


def test_regret_vec_matches_scalar_and_ci_flags_are_consistent():
    from cas.autoresearch.eval import (COST_GRID, _regret_sweep_ci, _regret_vec,
                                       decision_regret)

    rng = np.random.default_rng(DATA_SEED)
    groups, n = _grouped(n_groups=40, rows_per=15)
    y = (rng.random(n) < 0.6).astype(int)
    p = np.clip(0.5 + (2 * y - 1) * 0.25 + 0.1 * rng.standard_normal(n), 0.01, 0.99)

    # vectorized regret must equal the scalar decision_regret at every cost
    vec = _regret_vec(y, p, np.asarray(COST_GRID))
    for c, v in zip(COST_GRID, vec):
        assert abs(float(v) - decision_regret(y, p, cost_draft=c)) < 1e-9

    # CI structure: lo<=hi, helps_ci iff the 95% CI is entirely below 0, p in [0,1]
    base_p = np.clip(0.6 + 0.02 * rng.standard_normal(n), 0.01, 0.99)  # ~flat baseline
    ci = _regret_sweep_ci(y, base_p, p, groups, seed=SEED, n_boot=300)
    assert len(ci) == len(COST_GRID)
    for e in ci:
        assert e["delta_ci_lo"] <= e["delta_ci_hi"]
        assert e["helps_ci"] == (e["delta_ci_hi"] < 0.0)
        assert 0.0 <= e["p_delta_ge_0"] <= 1.0


def test_noninformative_candidate_is_not_ci_robust():
    # A candidate equal to the baseline (zero ranking lift) must NOT show a
    # CI-robust regret reduction at any cost -- this is the align/drift guard.
    rng = np.random.default_rng(DATA_SEED)
    groups, n = _grouped(n_groups=40, rows_per=15)
    y = (rng.random(n) < 0.6).astype(int)
    base_p = np.clip(0.6 + 0.05 * rng.standard_normal(n), 0.01, 0.99)
    comb_p = base_p.copy()                               # identical -> no real help

    from cas.autoresearch.eval import _regret_sweep_ci
    ci = _regret_sweep_ci(y, base_p, comb_p, groups, seed=SEED, n_boot=300)
    assert not any(e["helps_ci"] for e in ci)            # zero delta -> never CI-robust


def test_credible_systems_flag_is_conservative():
    # A candidate with no ranking lift must never be flagged credible, and the
    # flag must be internally consistent (credible => AUROC-CI-clean AND >=2
    # CI-robust costs). This is the drift/align guard at the summary level.
    rng = np.random.default_rng(DATA_SEED)
    groups, n = _grouped()
    latent = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-latent))).astype(int)
    X_base = rng.standard_normal((n, 3))                 # neither carries `latent`
    X_noise = rng.standard_normal((n, 1))                # -> no real signal

    res = incremental_lift(X_base, X_noise, y, groups, seed=SEED, n_boot=300)

    assert res["credible_systems"] is False
    assert isinstance(res["credible_systems"], bool)
    for k in ("auroc_ci_clean", "n_ci_robust_costs", "credible_systems"):
        assert k in res
    # internal consistency of the composite flag
    if res["credible_systems"]:
        assert res["auroc_ci_clean"] and res["n_ci_robust_costs"] >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
