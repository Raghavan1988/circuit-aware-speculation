"""Unit tests for the strict dev->test transfer scorer (frozen_transfer_lift)
and the dev-frozen baseline statistics (I13/I23, D025 frozen test pass).

Small SYNTHETIC arrays built IN THE TEST — a unit test of the scorer's
mechanics (does a real signal transfer, does noise fail its control, is the
pipeline deterministic and truly dev-frozen), NOT an experimental result. No
synthetic number is ever reported as a research finding (AGENTS.md). Requires
numpy + sklearn (pinned env).
"""
import numpy as np
import pytest

from cas.autoresearch.eval import (_apply_platt, _calibrate_oof, _fit_platt,
                                   frozen_transfer_lift)
from scripts.fit_autoresearch import _baseline_design, _baseline_stats

SEED = 0            # the scorer's internal seed (controls, bootstrap)
# Different stream than SEED so the scorer's control noise never collides with
# the synthetic signal (see tests/test_autoresearch_eval.py for the rationale).
DATA_SEED = 20260722


def _make_split(rng, n_groups, rows_per, group_offset, informative):
    """One split (dev or test) drawn from a shared generative process."""
    n = n_groups * rows_per
    groups = np.repeat(np.arange(n_groups) + group_offset, rows_per)
    latent = rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.6 * latent))).astype(int)
    X_base = rng.standard_normal((n, 3))                 # uninformative baseline
    if informative:
        X_cand = (latent + 0.5 * rng.standard_normal(n)).reshape(-1, 1)
    else:
        X_cand = rng.standard_normal((n, 1))             # pure noise
    return X_base, X_cand, y, groups


def test_informative_candidate_transfers_dev_to_test():
    rng = np.random.default_rng(DATA_SEED)
    Xb_d, Xc_d, y_d, g_d = _make_split(rng, 40, 12, 0, informative=True)
    Xb_t, Xc_t, y_t, g_t = _make_split(rng, 40, 12, 100, informative=True)

    res = frozen_transfer_lift(Xb_d, Xc_d, y_d, g_d, Xb_t, Xc_t, y_t, g_t,
                               seed=SEED, n_boot=400)

    assert res["protocol"] == "frozen_dev_to_test"
    assert res["n"] == len(y_t) and res["n_dev"] == len(y_d)
    assert res["beats_baseline"] is True
    assert res["deltas"]["auroc"] is not None and res["deltas"]["auroc"] > 0.0
    assert res["beats_controls"] is True
    assert res["delta_auroc_ci"]["p_delta_le_0"] < 0.5
    # dev-reference OOF numbers exist and show the same direction
    dr = res["dev_reference"]
    assert dr["combined_auroc_oof"] > dr["base_auroc_oof"]


def test_noise_candidate_fails_equal_capacity_control_frozen():
    rng = np.random.default_rng(DATA_SEED)
    Xb_d, Xc_d, y_d, g_d = _make_split(rng, 40, 12, 0, informative=False)
    Xb_t, Xc_t, y_t, g_t = _make_split(rng, 40, 12, 100, informative=False)

    res = frozen_transfer_lift(Xb_d, Xc_d, y_d, g_d, Xb_t, Xc_t, y_t, g_t,
                               seed=SEED, n_boot=400)

    # a dev-fit probe on noise must not register a transferable lift
    ci = res["delta_auroc_ci"]
    assert ci["lo"] <= 0.0 <= ci["hi"]
    assert res["auroc_ci_clean"] is False
    assert res["credible_systems"] is False


def test_frozen_transfer_deterministic():
    rng = np.random.default_rng(DATA_SEED)
    Xb_d, Xc_d, y_d, g_d = _make_split(rng, 25, 8, 0, informative=True)
    Xb_t, Xc_t, y_t, g_t = _make_split(rng, 25, 8, 100, informative=True)

    r1 = frozen_transfer_lift(Xb_d, Xc_d, y_d, g_d, Xb_t, Xc_t, y_t, g_t,
                              seed=SEED, n_boot=200)
    r2 = frozen_transfer_lift(Xb_d, Xc_d, y_d, g_d, Xb_t, Xc_t, y_t, g_t,
                              seed=SEED, n_boot=200)

    assert r1["deltas"]["auroc"] == r2["deltas"]["auroc"]
    assert r1["delta_auroc_ci"] == r2["delta_auroc_ci"]
    assert r1["combined"]["auroc"] == r2["combined"]["auroc"]
    assert ([s["delta_regret"] for s in r1["regret_cost_sweep"]]
            == [s["delta_regret"] for s in r2["regret_cost_sweep"]])


def test_platt_split_helpers_match_calibrate_oof():
    rng = np.random.default_rng(DATA_SEED)
    y = (rng.random(300) < 0.6).astype(int)
    scores = np.clip(rng.random(300), 0.01, 0.99)

    via_split = _apply_platt(_fit_platt(scores, y, seed=SEED), scores)
    via_oof = _calibrate_oof(scores, y, seed=SEED)
    assert np.allclose(via_split, via_oof)

    # single-class fallback: None map -> constant prior
    y1 = np.ones(50, dtype=int)
    assert _fit_platt(scores[:50], y1, seed=SEED) is None
    out = _apply_platt(None, scores[:50], prior=0.25)
    assert np.all(out == 0.25)


def test_baseline_design_frozen_stats_impute_from_dev():
    base_by_key = {
        ("r1", 0): {"history_ema": 0.2, "prev_target_entropy": 1.0,
                    "prev_target_margin": 0.5},
        ("r1", 1): {"history_ema": 0.4, "prev_target_entropy": 3.0,
                    "prev_target_margin": 0.7},
        # ("r2", 0) intentionally missing -> imputed
    }
    meta_dev = [{"request_id": "r1", "round_id": 0, "domain": "chat"},
                {"request_id": "r1", "round_id": 1, "domain": "code"}]
    meta_test = [{"request_id": "r2", "round_id": 0, "domain": "chat"}]

    stats = _baseline_stats(meta_dev, base_by_key, include_domain=True)
    assert stats["means"]["prev_target_entropy"] == pytest.approx(2.0)
    assert stats["domains"] == ["chat", "code"]          # dev vocabulary only

    X_test = _baseline_design(meta_test, base_by_key, include_domain=True,
                              stats=stats)
    # missing round imputed with the DEV mean (2.0), missing flag set
    assert X_test[0, 2] == pytest.approx(2.0)            # prev_target_entropy
    assert X_test[0, 3] == 1.0                           # its missing flag
    # domain one-hot uses the dev vocabulary ordering
    assert X_test[0, 6] == 1.0 and X_test[0, 7] == 0.0   # chat=1, code=0

    # stats=None reproduces the original within-split behavior exactly
    explicit = _baseline_design(meta_dev, base_by_key, include_domain=True,
                                stats=stats)
    default = _baseline_design(meta_dev, base_by_key, include_domain=True)
    assert np.array_equal(explicit, default)
