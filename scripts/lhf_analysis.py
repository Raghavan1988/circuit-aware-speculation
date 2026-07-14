"""Low-hanging-fruit analyses over sealed fixed_8 traces (all CPU, offline).

Six directions, each computed from the immutable counterfactual labels + recorded
signals in a fixed_8 run. No model, no GPU. Works on any sealed run_dir (Qwen v1,
Llama, v2), so transfer/replication is the same code on different inputs.

  #1 pre_round_gate     - length + skip decided BEFORE drafting, from the PRIOR
                          round's target frontier entropy (the C10 cell).
  #2 calibration_audit  - fine entropy-threshold efficiency curve + the true
                          optimum vs the tuned tau; P(accept | entropy) calibration.
  #3 headroom_attribution - entropy-stop (per-round) vs a category-clairvoyant
                          static length map (upper bound on a content LOOKUP):
                          how much of the win is static structure vs per-round?
  #5 skip_economics     - decompose the entropy controller's rounds into
                          skip / shorten / full and the marginal value of skip.
  #6 block_breaker      - which token CATEGORIES terminate accepted runs (the
                          run-length substrate a block/diffusion drafter needs).
(#4 transfer is run by the Modal driver: apply one run's tuned tau to another.)
"""
from __future__ import annotations

import os

from scripts.eval_length_policies import (ACTIONS, accepted_under,
                                          make_cost_serving)


def load_rounds(run_dir):
    """Per request: ordered rounds with match vector, per-position draft entropy,
    proposed ids, and the PREVIOUS round's target frontier entropy (pre-draft)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(os.path.join(run_dir, "fixed_8", "rounds.parquet")).to_pylist()
    by: dict = {}
    for r in rows:
        p = list(r["proposed_token_ids"] or [])
        t = list(r["target_argmax_ids"] or [])
        if not p:
            continue
        m = tuple(p[i] == t[i] for i in range(min(len(p), len(t))))
        by.setdefault(r["request_id"], []).append(
            {"rid": r["round_id"], "match": m, "ent": list(r["draft_entropy"] or []),
             "proposed": p, "fe": r.get("target_entropy_frontier"),
             "start": r["start_output_pos"]})
    for k in by:
        by[k].sort(key=lambda x: x["rid"])
        pe = None
        for rnd in by[k]:
            rnd["prev_fe"] = pe
            pe = rnd["fe"]
    return by


def split_of(run_dir):
    import json

    import pyarrow.parquet as pq
    summ = pq.read_table(os.path.join(run_dir, "fixed_8",
                                      "request_summaries.parquet")).to_pylist()
    data_root = os.path.dirname(os.path.dirname(run_dir.rstrip("/")))
    with open(os.path.join(data_root, "data", "split_manifest.json")) as f:
        assign = json.load(f)["assignment"]
    return {s["request_id"]: assign.get(s["prompt_hash"], "unknown") for s in summ}


def rounds_for(by, split, want):
    return [rs for rid, rs in sorted(by.items()) if split.get(rid) == want]


def _entropy_stop(rnd, tau):
    L = 0
    for e in rnd["ent"][:8]:
        if e is None or e > tau:
            break
        L += 1
    return L


def _eff(rounds, choose_L, cost):
    emit = fw = drafted = accepted = n = skip = 0
    for rs in rounds:
        for rnd in rs:
            L = choose_L(rnd)
            k = accepted_under(rnd["match"], L)
            emit += k + 1; fw += cost(L); drafted += L; accepted += k; n += 1
            skip += (L == 0)
    return {"eff": emit / fw, "yield": emit / n, "wasted_per_emit": (drafted - accepted) / emit,
            "skip_frac": skip / n, "n": n}


# ---------------- #2 calibration-optimal stopping audit ----------------------
def calibration_audit(rounds_dev, rounds_test, cost):
    taus = [round(0.1 * i, 1) for i in range(1, 61)]  # 0.1 .. 6.0
    dev_curve = [(tau, _eff(rounds_dev, lambda r, t=tau: _entropy_stop(r, t), cost)["eff"])
                 for tau in taus]
    tau_star = max(dev_curve, key=lambda x: x[1])[0]
    # P(accept | entropy bin) over test positions (the calibration the rule reads)
    edges = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    bins = [[0, 0] for _ in range(len(edges) + 1)]
    for rs in rounds_test:
        for rnd in rs:
            for e, m in zip(rnd["ent"], rnd["match"]):
                if e is None:
                    continue
                i = next((j for j, ed in enumerate(edges) if e <= ed), len(edges))
                bins[i][0] += 1; bins[i][1] += bool(m)
    calib = [{"bin": ("<=%.2f" % edges[i]) if i < len(edges) else ">6.0",
              "n": bins[i][0], "p_accept": round(bins[i][1] / bins[i][0], 3) if bins[i][0] else None}
             for i in range(len(bins))]
    # cost-ratio optimality: drafting one more position is worth it iff its accept
    # prob exceeds the marginal cost fraction; ratio = draft/verify cost.
    ratio = cost(1) - cost(0)
    p_threshold = ratio / (1.0 + ratio)  # break-even P(accept) for one extra draft
    e_star = _eff(rounds_test, lambda r: _entropy_stop(r, tau_star), cost)["eff"]
    e_tuned = _eff(rounds_test, lambda r: _entropy_stop(r, 2.0), cost)["eff"]
    return {"tau_star_dev": tau_star, "eff_at_tau_star_test": round(e_star, 4),
            "eff_at_tuned_2.0_test": round(e_tuned, 4),
            "tuned_is_within_pct_of_optimal": round((e_tuned / e_star - 1) * 100, 2),
            "breakeven_p_accept": round(p_threshold, 3),
            "calibration": calib}


# ---------------- #1 pre-round C10 length + skip gate ------------------------
def pre_round_gate(by, split, cost):
    """Decide L per round from the PRIOR round's frontier entropy only (pre-draft).
    Tune a bin->best-L lookup on dev, apply frozen on test. Compare to best fixed
    and to the within-round entropy-stop. Also a pre-round SKIP gate."""
    dev = rounds_for(by, split, "dev")
    test = rounds_for(by, split, "test")
    edges = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

    def fe_bin(v):
        if v is None:
            return len(edges)  # first round of a request: no prior signal
        return next((j for j, ed in enumerate(edges) if v <= ed), len(edges))

    # tune: for each prev-frontier bin, the fixed L maximizing dev efficiency
    best_L = {}
    for b in range(len(edges) + 1):
        sub = [[rnd] for rs in dev for rnd in rs if fe_bin(rnd["prev_fe"]) == b]
        if not sub:
            best_L[b] = 4
            continue
        best_L[b] = max(ACTIONS, key=lambda L: _eff(sub, lambda r, LL=L: LL, cost)["eff"])
    gate = lambda rnd: best_L[fe_bin(rnd["prev_fe"])]
    res = _eff(test, gate, cost)
    # baselines on test
    best_fixed = max(_eff(test, lambda r, LL=L: LL, cost)["eff"] for L in ACTIONS)
    estop = _eff(test, lambda r: _entropy_stop(r, 2.0), cost)["eff"]
    return {"pre_round_map_bestL_by_frontier_bin": best_L,
            "pre_round_eff_test": round(res["eff"], 4),
            "pre_round_skip_frac": round(res["skip_frac"], 3),
            "best_fixed_eff_test": round(best_fixed, 4),
            "entropy_stop_eff_test": round(estop, 4),
            "pre_round_vs_best_fixed_pct": round((res["eff"] / best_fixed - 1) * 100, 2),
            "pre_round_vs_within_round_pct": round((res["eff"] / estop - 1) * 100, 2)}


# ---------------- #5 skip economics -----------------------------------------
def skip_economics(rounds, cost, tau=2.0):
    n = skip = shorten = full = 0
    for rs in rounds:
        for rnd in rs:
            L = _entropy_stop(rnd, tau)
            n += 1
            if L == 0:
                skip += 1
            elif L < 8:
                shorten += 1
            else:
                full += 1
    with_skip = _eff(rounds, lambda r: _entropy_stop(r, tau), cost)["eff"]
    # same rule but never skip (floor L at 1): isolate skip's marginal value
    no_skip = _eff(rounds, lambda r: max(1, _entropy_stop(r, tau)), cost)["eff"]
    return {"skip_pct": round(100 * skip / n, 1), "shorten_pct": round(100 * shorten / n, 1),
            "full_pct": round(100 * full / n, 1),
            "eff_with_skip": round(with_skip, 4), "eff_no_skip": round(no_skip, 4),
            "skip_marginal_gain_pct": round((with_skip / no_skip - 1) * 100, 2)}


# ---------------- #3 + #6 category-based (need the tokenizer) ----------------
def category_analyses(run_dir, by, split, cost, tokenizer_id, tokenizer_rev=None):
    """#6 block_breaker: which categories are the FIRST rejection in a round.
    #3 headroom_attribution: category-clairvoyant static length map vs entropy."""
    from transformers import AutoTokenizer

    from cas.annotate.categories import annotate_categories

    tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_rev or None)
    cat_cache: dict = {}

    def cats(tid):
        c = cat_cache.get(tid)
        if c is None:
            c = sorted(annotate_categories(tok.convert_ids_to_tokens(int(tid)),
                                           token_id=int(tid)))
            cat_cache[tid] = c
        return c

    test = rounds_for(by, split, "test")
    # #6: first-rejection token category tally (block breakers)
    breaker = {}
    total_breaks = 0
    for rs in test:
        for rnd in rs:
            fr = next((i for i, m in enumerate(rnd["match"]) if not m), None)
            if fr is None or fr >= len(rnd["proposed"]):
                continue
            total_breaks += 1
            for c in (cats(rnd["proposed"][fr]) or ["uncategorized"]):
                breaker[c] = breaker.get(c, 0) + 1
    breaker_rate = sorted(((c, round(v / total_breaks, 3), v) for c, v in breaker.items()),
                          key=lambda x: -x[1])
    # #3: category-clairvoyant static length (upper bound on a content LOOKUP).
    # For each round, categorize the FIRST proposed token (available notion of the
    # upcoming content) and use the dev-optimal fixed L for that category.
    dev = rounds_for(by, split, "dev")

    def first_cat(rnd):
        cs = cats(rnd["proposed"][0]) if rnd["proposed"] else ["uncategorized"]
        return cs[0] if cs else "uncategorized"

    by_cat_dev: dict = {}
    for rs in dev:
        for rnd in rs:
            by_cat_dev.setdefault(first_cat(rnd), []).append(rnd)
    cat_bestL = {c: max(ACTIONS, key=lambda L: _eff([[r] for r in rs], lambda x, LL=L: LL, cost)["eff"])
                 for c, rs in by_cat_dev.items()}
    static = _eff(test, lambda r: cat_bestL.get(first_cat(r), 8), cost)["eff"]
    estop = _eff(test, lambda r: _entropy_stop(r, 2.0), cost)["eff"]
    best_fixed = max(_eff(test, lambda r, LL=L: LL, cost)["eff"] for L in ACTIONS)
    return {"block_breakers_top": breaker_rate[:12], "n_breaks": total_breaks,
            "attribution": {
                "best_fixed": round(best_fixed, 4),
                "category_static_lookup": round(static, 4),
                "entropy_stop_per_round": round(estop, 4),
                "static_share_of_gain_pct": round(
                    100 * (static - best_fixed) / (estop - best_fixed), 1)
                if estop > best_fixed else None}}


def run_all(run_dir, ratio=0.1, tokenizer_id=None, tokenizer_rev=None):
    by = load_rounds(run_dir)
    split = split_of(run_dir)
    cost = make_cost_serving(ratio)
    dev = rounds_for(by, split, "dev")
    test = rounds_for(by, split, "test")
    out = {"run_dir": run_dir, "ratio": ratio,
           "n_dev_rounds": sum(len(rs) for rs in dev),
           "n_test_rounds": sum(len(rs) for rs in test),
           "calibration_audit": calibration_audit(dev, test, cost),
           "pre_round_gate": pre_round_gate(by, split, cost),
           "skip_economics": skip_economics(test, cost)}
    if tokenizer_id:
        out["category"] = category_analyses(run_dir, by, split, cost,
                                            tokenizer_id, tokenizer_rev)
    return out
