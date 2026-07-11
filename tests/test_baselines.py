"""Surface-baseline feature construction (Task #4 / I13): leakage rules.

The point of these tests is to lock the anti-leakage guarantees (D018): history
uses only prior rounds; pre-round target signals come from the PREVIOUS verify,
never the current round's outcome. Fixtures are constructed inputs; no
experimental values are asserted as results (AGENTS.md).
"""
from cas.analysis.baselines import FEATURE_SETS, feature_rows


def rnd(request_id, round_id, start, proposed, targets, accepted,
        ents, margs, front_e, front_m):
    return {"request_id": request_id, "round_id": round_id,
            "start_output_pos": start, "proposed_token_ids": proposed,
            "target_argmax_ids": targets, "accepted_prefix_len": accepted,
            "realized_draft_len": len(proposed),
            "draft_entropy": ents, "draft_top1_margin": margs,
            "target_entropy_frontier": front_e,
            "target_margin_frontier": front_m}


def test_history_uses_only_prior_rounds():
    rounds = [
        rnd("p", 0, 1, [5, 6], [5, 9, 0], 1, [0.1, 0.2], [0.9, 0.8], 1.5, 0.4),
        rnd("p", 1, 3, [7, 8], [7, 8, 0], 2, [0.3, 0.4], [0.7, 0.6], 1.1, 0.5),
    ]
    rows = feature_rows(rounds, {"p": "code"})
    # round 0's tokens see NO history (None) — nothing precedes them
    r0 = [r for r in rows if r["round_id"] == 0]
    assert all(r["history_ema"] is None for r in r0)
    # round 1's tokens see history from round 0 only (a real float now)
    r1 = [r for r in rows if r["round_id"] == 1]
    assert all(r["history_ema"] is not None for r in r1)


def test_preround_target_signal_is_from_previous_round():
    rounds = [
        rnd("p", 0, 1, [5], [5, 0], 1, [0.1], [0.9], 1.5, 0.4),
        rnd("p", 1, 2, [7], [7, 0], 1, [0.3], [0.7], 2.2, 0.6),
    ]
    rows = feature_rows(rounds, {"p": "math"})
    r0 = [r for r in rows if r["round_id"] == 0][0]
    r1 = [r for r in rows if r["round_id"] == 1][0]
    # round 0 has no previous verify -> None; round 1 carries round 0's frontier
    assert r0["prev_target_entropy"] is None
    assert r1["prev_target_entropy"] == 1.5   # round 0's value, NOT 2.2
    assert r1["prev_target_margin"] == 0.4


def test_label_is_committed_acceptance_not_counterfactual():
    # proposed[1] matches target counterfactually but sits after the rejection
    r = rnd("p", 0, 1, [5, 6], [5, 6, 0], 1, [0.1, 0.2], [0.9, 0.8], 1.0, 0.5)
    rows = feature_rows([r], {"p": "chat"})
    assert [x["accepted"] for x in rows] == [True, False]
    assert [x["target_match"] for x in rows] == [True, True]  # both match


def test_no_same_round_outcome_leaks_into_features():
    r = rnd("p", 0, 1, [5], [5, 0], 1, [0.4], [0.6], 1.9, 0.3)
    row = feature_rows([r], {"p": "code"})[0]
    feature_keys = {k for _, cols in FEATURE_SETS for k in cols}
    # the current round's frontier entropy/margin (an outcome) is never a feature
    assert "target_entropy_frontier" not in feature_keys
    assert "target_margin_frontier" not in feature_keys
    # only prev_* target signals are exposed, and this first round's is None
    assert row["prev_target_entropy"] is None


def test_feature_sets_reference_only_available_columns():
    r = rnd("p", 0, 1, [5], [5, 0], 1, [0.4], [0.6], 1.9, 0.3)
    row = feature_rows([r], {"p": "code"})[0]
    for _, cols in FEATURE_SETS:
        for c in cols:
            assert c in row, f"feature set references missing column {c}"
