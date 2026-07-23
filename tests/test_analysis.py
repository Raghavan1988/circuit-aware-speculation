"""Task #4 analysis primitives: atlas aggregation and counterfactual oracle.

Fixtures are constructed inputs for code validation only — no experimental
values are asserted as results (AGENTS.md).
"""
import pytest

from cas.analysis import (
    accepted_for_action,
    action_utilities,
    atlas_cells,
    atlas_table,
    bootstrap_rate_ci,
    measured_costs_from_rounds,
    oracle_policy_value,
)


def tok(req, accepted, cats, phase="mid"):
    return {"request_id": req, "accepted": accepted,
            "categories": cats, "phase": phase}


# ---- atlas -------------------------------------------------------------------

def test_atlas_cells_overlapping_categories():
    tokens = [tok("p1", True, ["punctuation", "sentence_boundary"]),
              tok("p1", False, ["number"]),
              tok("p2", True, ["number"])]
    cells = atlas_cells(tokens)
    assert cells[("punctuation", "mid")]["n"] == 1
    assert cells[("sentence_boundary", "mid")]["n"] == 1  # same token, both cells
    num = cells[("number", "mid")]
    assert num["n"] == 2 and num["accepted"] == 1 and num["rate"] == 0.5
    assert set(num["prompts"]) == {"p1", "p2"}


def test_atlas_empty_categories_bucketed():
    cells = atlas_cells([tok("p1", True, [])])
    assert ("uncategorized", "mid") in cells


def test_bootstrap_ci_deterministic_and_ordered():
    prompts = {f"p{i}": [10, 5 + (i % 3)] for i in range(20)}
    a = bootstrap_rate_ci(prompts, n_boot=300, seed=7)
    b = bootstrap_rate_ci(prompts, n_boot=300, seed=7)
    assert a == b                     # seeded determinism
    assert 0.0 <= a[0] <= a[1] <= 1.0
    point = sum(v[1] for v in prompts.values()) / sum(v[0] for v in prompts.values())
    assert a[0] <= point <= a[1]


def test_atlas_table_sorted_with_cis():
    tokens = ([tok("p1", False, ["number"]) for _ in range(30)]
              + [tok("p2", False, ["number"]) for _ in range(30)]
              + [tok("p1", True, ["punctuation"]) for _ in range(30)]
              + [tok("p2", True, ["punctuation"]) for _ in range(30)])
    rows = atlas_table(tokens, min_n=50, n_boot=100)
    assert [r["category"] for r in rows] == ["number", "punctuation"]
    assert rows[0]["rate"] == 0.0 and rows[1]["rate"] == 1.0
    assert rows[0]["n_prompts"] == 2


# ---- oracle ------------------------------------------------------------------

COSTS = {0: 10.0, 1: 12.0, 2: 14.0, 3: 16.0, 4: 18.0, 6: 22.0, 8: 26.0}


def test_accepted_for_action_prefix_semantics():
    match = [True, True, False, True]  # position 3 matches counterfactually
    assert accepted_for_action(match, 0) == 0
    assert accepted_for_action(match, 1) == 1
    assert accepted_for_action(match, 2) == 2
    assert accepted_for_action(match, 3) == 2  # stops at first False
    assert accepted_for_action(match, 4) == 2  # post-rejection match ignored


def test_action_utilities_censoring():
    utils = action_utilities([True, True], COSTS)   # realized length 2
    assert set(utils) == {0, 1, 2}                  # 3,4,6,8 censored
    assert utils[2][0] == 3                         # 2 accepted + correction/bonus


def test_oracle_beats_fixed_on_mixed_rounds():
    easy = [True] * 8      # long drafts pay off
    hard = [False] * 8     # skip pays off
    res = oracle_policy_value([easy, hard] * 50, COSTS)
    assert res["headroom"] > 0                      # adaptivity has value here
    assert res["oracle"] >= res["best_fixed"][1]
    assert res["n_rounds"] == 100 and res["censored"] == 0


def test_oracle_no_headroom_when_uniform():
    res = oracle_policy_value([[True] * 8] * 20, COSTS)
    assert res["headroom"] == pytest.approx(0.0)    # oracle == best fixed (L=8)
    assert res["best_fixed"][0] == 8


def test_measured_costs_from_round_rows():
    rows = [{"requested_action": 4,
             "latency_ns": '{"draft": 100, "verify": 200, "controller": 5}'},
            {"requested_action": 4,
             "latency_ns": {"draft": 300, "verify": 400, "controller": 5}}]
    costs = measured_costs_from_rounds(rows)
    assert costs == {4: (305 + 705) / 2}


def test_bootstrap_delta_ci_separated_pools():
    from cas.analysis.atlas import bootstrap_delta_ci

    # 30 prompts; pool A accepts 9/10 per prompt, pool B 2/10 -> clean contrast
    pa = {f"p{i}": [10, 9] for i in range(30)}
    pb = {f"p{i}": [10, 2] for i in range(30)}
    d = bootstrap_delta_ci(pa, pb, n_boot=300, seed=0)
    assert d["delta"] == pytest.approx(0.7)
    assert d["lo"] > 0.0                      # CI excludes 0
    assert d["p_delta_le_0"] == 0.0
    assert d["n_boot_effective"] == 300


def test_bootstrap_delta_ci_null_contrast_straddles_zero():
    from cas.analysis.atlas import bootstrap_delta_ci

    # identical per-prompt counts, disjoint prompt ids -> exact null contrast
    pa = {f"p{i}": [10, 5 + (i % 3)] for i in range(30)}
    pb = {f"q{i}": [10, 5 + (i % 3)] for i in range(30)}
    d = bootstrap_delta_ci(pa, pb, n_boot=300, seed=0)
    assert d["delta"] == pytest.approx(0.0)
    assert d["lo"] <= 0.0 <= d["hi"]          # no real contrast


def test_bootstrap_delta_ci_deterministic_and_paired():
    from cas.analysis.atlas import bootstrap_delta_ci

    # shared prompts between pools (the category-pool case) + determinism
    pa = {f"p{i}": [8, 6] for i in range(20)}
    pb = {f"p{i}": [12, 5] for i in range(20)}
    d1 = bootstrap_delta_ci(pa, pb, n_boot=200, seed=3)
    d2 = bootstrap_delta_ci(pa, pb, n_boot=200, seed=3)
    assert d1 == d2
    assert d1["lo"] <= d1["delta"] <= d1["hi"]


def test_bootstrap_delta_ci_empty_pool():
    from cas.analysis.atlas import bootstrap_delta_ci

    d = bootstrap_delta_ci({}, {"p1": [5, 3]}, n_boot=50, seed=0)
    assert d["n_boot_effective"] == 0 and d["delta"] == 0.0
