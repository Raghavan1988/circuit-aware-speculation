"""Counterfactual per-round oracle and headroom (D018.3; Codex's go/no-go).

Every traced round's `target_argmax_ids` yields per-position draft/target
matches; for any action length H <= the round's realized length, the accepted
length under H is the all-True prefix of the first H matches (exact-greedy
structure, valid per round — D018.3). With a measured per-action cost profile
this gives best-action-in-hindsight utilities and the oracle headroom over
every fixed action: the number that decides whether controller work proceeds
(D018 tripwire: headroom below ~5 percent stops it).

Rounds from the fixed_8 run label ALL actions; shorter runs label actions up
to their realized length (censored — callers should prefer fixed_8 traces).
Pure stdlib; costs are injected, never invented.
"""
from __future__ import annotations

ACTIONS = (0, 1, 2, 3, 4, 6, 8)


def accepted_for_action(match: list[bool] | tuple[bool, ...], length: int) -> int:
    """Accepted length if only the first `length` proposals had been drafted."""
    k = 0
    for m in match[:length]:
        if not m:
            break
        k += 1
    return k


def action_utilities(match, costs: dict, actions=ACTIONS) -> dict:
    """Per-action (emitted, cost, tokens_per_cost) for one round.

    Actions longer than the observed match vector are censored (absent from
    the result) — except skip(0), which is always evaluable.
    """
    out = {}
    for length in actions:
        if length > len(match):
            continue
        emitted = accepted_for_action(match, length) + 1
        cost = costs[length]
        if cost <= 0:
            raise ValueError(f"non-positive cost for action {length}")
        out[length] = (emitted, cost, emitted / cost)
    return out


def oracle_policy_value(rounds, costs: dict, actions=ACTIONS) -> dict:
    """Aggregate tokens-per-cost of the per-round best action vs every fixed
    action, over an iterable of per-round match vectors.

    Returns {"oracle": tpc, "fixed": {L: tpc}, "best_fixed": (L, tpc),
             "headroom": relative oracle gain over the best fixed action,
             "n_rounds": ..., "censored": rounds lacking full action coverage}.
    """
    tot_or_tok = tot_or_cost = 0.0
    fixed_tok = {a: 0.0 for a in actions}
    fixed_cost = {a: 0.0 for a in actions}
    fixed_n = {a: 0 for a in actions}
    n = censored = 0
    for match in rounds:
        n += 1
        utils = action_utilities(match, costs, actions)
        if len(utils) < len(actions):
            censored += 1
        best = max(utils.values(), key=lambda v: v[2])
        tot_or_tok += best[0]
        tot_or_cost += best[1]
        for a, (tok, cost, _) in utils.items():
            fixed_tok[a] += tok
            fixed_cost[a] += cost
            fixed_n[a] += 1
    if n == 0:
        raise ValueError("no rounds")
    oracle = tot_or_tok / tot_or_cost
    fixed = {a: (fixed_tok[a] / fixed_cost[a]) for a in actions if fixed_n[a]}
    best_a = max(fixed, key=lambda a: fixed[a])
    headroom = (oracle - fixed[best_a]) / fixed[best_a]
    return {"oracle": oracle, "fixed": fixed, "best_fixed": (best_a, fixed[best_a]),
            "headroom": headroom, "n_rounds": n, "censored": censored}


def measured_costs_from_rounds(round_rows, actions=ACTIONS) -> dict:
    """Mean measured per-round cost (draft+verify+controller+tracing ns) per
    requested action, from round dicts as read off rounds.parquet. This is the
    injected cost profile for the oracle — measured, never assumed."""
    import json

    tot = {a: 0 for a in actions}
    cnt = {a: 0 for a in actions}
    for r in round_rows:
        a = r["requested_action"]
        if a not in tot:
            continue
        lat = r["latency_ns"]
        if isinstance(lat, str):
            lat = json.loads(lat)
        tot[a] += sum(lat.values())
        cnt[a] += 1
    return {a: tot[a] / cnt[a] for a in actions if cnt[a]}
