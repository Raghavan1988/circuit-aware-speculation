"""Tests for the I15 intervention harness pure logic (cas.autoresearch.interventions).

Pure numpy + pytest, local. Covers direction derivation, norm-matched/shuffled
controls, dose-response, beats-controls, and the two decisive entropy-mediation
cases (fully-entropy-mediated -> NOT beyond entropy; direct effect -> beyond
entropy). Synthetic data built in-test (unit test of the harness, not a result).
"""
import numpy as np

from cas.autoresearch.interventions import (acceptance_direction, beats_controls,
                                            dose_response, entropy_mediation,
                                            entropy_stratified_effect, g2_verdict,
                                            random_control, shuffled_control)


def test_acceptance_direction_recovers_planted():
    rng = np.random.default_rng(0)
    d = 32
    u = rng.standard_normal(d); u /= np.linalg.norm(u)          # planted direction
    n = 400
    accept = (rng.random(n) < 0.5).astype(int)
    states = rng.standard_normal((n, d)) + 3.0 * accept[:, None] * u  # accepted shifted +u
    dhat = acceptance_direction(states, accept)
    assert abs(np.linalg.norm(dhat) - 1.0) < 1e-9                # unit norm
    assert float(dhat @ u) > 0.9                                # aligned with planted


def test_controls_preserve_norm():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(64) * 2.5
    rc = random_control(d, seed=3)
    sc = shuffled_control(d, seed=3)
    assert abs(np.linalg.norm(rc) - np.linalg.norm(d)) < 1e-9   # norm-matched
    assert abs(np.linalg.norm(sc) - np.linalg.norm(d)) < 1e-9   # norm-preserving
    assert sorted(sc.tolist()) == sorted(d.tolist())            # a permutation of d


def test_dose_response_and_beats_controls():
    alphas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    real = np.array([0.30, 0.40, 0.50, 0.60, 0.70])            # monotone increasing
    dr = dose_response(alphas, real)
    assert dr["monotone"] and dr["slope"] > 0 and dr["effect"] > 0.3
    # a flat control has ~zero effect -> real beats it
    ctrl = dose_response(alphas, np.array([0.50, 0.49, 0.50, 0.51, 0.50]))
    assert beats_controls(dr["effect"], [ctrl["effect"]])


def test_entropy_mediation_fully_mediated_is_not_beyond():
    # alpha -> entropy -> accept : acceptance flows ENTIRELY through entropy, so the
    # direct alpha effect vanishes. Robust within-bin check is the primary verdict;
    # LPM agrees here because accept is ~linear in entropy (well-specified).
    rng = np.random.default_rng(2)
    n = 6000
    alpha = rng.uniform(-2, 2, n)
    entropy = 0.3 * alpha + rng.standard_normal(n)            # weak coupling -> separable
    prob = np.clip(0.5 + 0.15 * entropy, 0.02, 0.98)          # accept LINEAR in entropy
    accept = (rng.random(n) < prob).astype(int)               # depends on entropy only
    # within entropy bins (controlling residual entropy), alpha has no effect
    assert entropy_stratified_effect(alpha, accept, entropy)["beyond_entropy"] is False


def test_entropy_mediation_direct_effect_is_beyond():
    # alpha -> accept directly, entropy independent noise -> beyond entropy (both
    # the robust within-bin check and the LPM agree).
    rng = np.random.default_rng(3)
    n = 4000
    alpha = rng.uniform(-2, 2, n)
    accept = ((2.0 * alpha + rng.standard_normal(n)) > 0).astype(int)
    entropy = rng.standard_normal(n)                           # unrelated to accept
    assert entropy_stratified_effect(alpha, accept, entropy)["beyond_entropy"] is True
    m = entropy_mediation(alpha, accept, entropy)
    assert m["beyond_entropy"] is True and m["c_total"] > 0 and m["c_direct"] > 0


def test_g2_verdict_requires_all_conditions():
    alphas = np.array([-2.0, 0.0, 2.0])
    real = dose_response(alphas, np.array([0.3, 0.5, 0.7]))     # monotone, effect 0.4
    ctrls = [dose_response(alphas, np.array([0.5, 0.5, 0.5]))]  # flat controls
    good_med = {"beyond_entropy": True}
    bad_med = {"beyond_entropy": False}
    assert g2_verdict(real, ctrls, good_med)["causal_beyond_entropy"] is True
    assert g2_verdict(real, ctrls, bad_med)["causal_beyond_entropy"] is False   # entropy-mediated
    # verdict never emits mechanistic language
    assert "diagnostic signal" in g2_verdict(real, ctrls, good_med)["language"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
