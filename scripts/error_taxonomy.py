"""T5.4: error taxonomy + candidate-set ablation over the sealed fixed_8 labels.

Roadmap error taxonomy, made concrete for the frozen entropy-stop controller
(tau tuned on dev) using the counterfactual per-position labels:

  * over-drafting  — tokens the controller drafted PAST the first rejection in
    a round (drafted but discarded): magnitude = chosen_L - accepted_k when a
    rejection occurred inside the chosen block.
  * under-drafting — the controller stopped although the match run continued:
    when all chosen tokens were accepted, the number of additional consecutive
    matches available (within the observed 8) that were left on the table.
  * signal miscalibration — P(accept) per draft-entropy bin: the reliability
    curve the threshold rule depends on (should be monotone decreasing).
  * cold-start regret — covered by the T5.3 bandit quartile convergence (see
    eval_length_policies); pointer only here.
  * batch interference — NOT MEASURABLE on this harness (batch size 1); stated
    explicitly rather than silently omitted.

Candidate-set ablation (roadmap A4): oracle + entropy-stop efficiency under a
coarse {0,1,4,8} vs fine {0,1,2,3,4,6,8} action menu.

Pure CPU over sealed traces; every number script-generated.
"""
from __future__ import annotations

import argparse
import json
import os

from scripts.eval_length_policies import (ACTIONS, _domain_of,
                                          _rounds_by_request, _split_of,
                                          accepted_under, entropy_stop_policy,
                                          evaluate, fixed_policy,
                                          make_cost_serving)

ENT_BINS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # right edges; +inf tail


def _menu_oracle(menu, cost):
    def p(rnd, mem):
        best, bestv = 0, -1.0
        for L in menu:
            if L > len(rnd["match"]):
                continue
            v = (accepted_under(rnd["match"], L) + 1) / cost(L)
            if v > bestv:
                best, bestv = L, v
        return best
    return p


def _menu_entropy(tau, menu):
    """Entropy stop, then round DOWN to the largest allowed length <= stop."""
    allowed = sorted(menu)

    def p(rnd, mem):
        L = 0
        for e in rnd["ent"][:8]:
            if e is None or e > tau:
                break
            L += 1
        return max((a for a in allowed if a <= L), default=0)
    return p


def run(run_dir, eval_split="test", draft_ratio=0.1, tau=2.0):
    by_req = _rounds_by_request(run_dir)
    split = _split_of(run_dir)
    dom = _domain_of(run_dir)
    rounds_by_domain: dict = {}
    all_rounds = []
    for rid, rs in sorted(by_req.items()):
        if split.get(rid) != eval_split:
            continue
        all_rounds.append(rs)
        rounds_by_domain.setdefault(dom.get(rid, "unknown"), []).append(rs)
    cost = make_cost_serving(draft_ratio)
    stop = entropy_stop_policy(tau)

    # ---- over/under-drafting per domain (frozen entropy controller) --------
    taxonomy = {}
    for d, rss in sorted(rounds_by_domain.items()):
        n = over_rounds = under_rounds = 0
        over_toks = under_toks = 0
        for rs in rss:
            for rnd in rs:
                L = stop(rnd, {})
                k = accepted_under(rnd["match"], L)
                n += 1
                if k < L:                       # drafted past the rejection
                    over_rounds += 1
                    over_toks += L - k
                elif L < len(rnd["match"]):     # stopped early: count missed run
                    miss = 0
                    for m in rnd["match"][L:]:
                        if not m:
                            break
                        miss += 1
                    if miss:
                        under_rounds += 1
                        under_toks += miss
        taxonomy[d] = {
            "n_rounds": n,
            "over_draft_round_pct": round(100 * over_rounds / n, 1),
            "over_draft_tokens_per_round": round(over_toks / n, 3),
            "under_draft_round_pct": round(100 * under_rounds / n, 1),
            "under_draft_tokens_per_round": round(under_toks / n, 3),
        }

    # ---- signal calibration: P(accept) per entropy bin ---------------------
    bins = [[0, 0] for _ in range(len(ENT_BINS) + 1)]  # [n, accepted]
    for rs in all_rounds:
        for rnd in rs:
            for e, m in zip(rnd["ent"], rnd["match"]):
                if e is None:
                    continue
                i = next((j for j, edge in enumerate(ENT_BINS) if e <= edge),
                         len(ENT_BINS))
                bins[i][0] += 1
                bins[i][1] += bool(m)
    edges = ["<=%.2f" % e for e in ENT_BINS] + [">%.1f" % ENT_BINS[-1]]
    calibration = [{"entropy_bin": edges[i], "n": bins[i][0],
                    "p_accept": round(bins[i][1] / bins[i][0], 3) if bins[i][0] else None}
                   for i in range(len(bins))]
    rates = [c["p_accept"] for c in calibration if c["p_accept"] is not None]
    monotone = all(a >= b for a, b in zip(rates, rates[1:]))

    # ---- candidate-set ablation (A4) ---------------------------------------
    menus = {"coarse{0,1,4,8}": (0, 1, 4, 8), "fine{0,1,2,3,4,6,8}": ACTIONS}
    candidate_set = {}
    for name, menu in menus.items():
        candidate_set[name] = {
            "oracle_eff": round(
                evaluate(all_rounds, _menu_oracle(menu, cost), cost)["eff_serving"], 4),
            "entropy_stop_eff": round(
                evaluate(all_rounds, _menu_entropy(tau, menu), cost)["eff_serving"], 4),
        }
    bf = max((evaluate(all_rounds, fixed_policy(L), cost)["eff_serving"], L)
             for L in ACTIONS)
    return {"eval_split": eval_split, "tau": tau, "draft_ratio": draft_ratio,
            "taxonomy_by_domain": taxonomy,
            "entropy_calibration": calibration,
            "calibration_monotone_decreasing": monotone,
            "candidate_set": candidate_set,
            "best_fixed": {"L": bf[1], "eff": round(bf[0], 4)},
            "cold_start": "see T5.3 bandit stream-quartile convergence",
            "batch_interference": "not measurable: batch-size-1 harness"}


def _print_tax(res):
    print(f"\nT5.4 ERROR TAXONOMY (split={res['eval_split']}, tau={res['tau']})")
    print(f"{'domain':<12}{'rounds':>8}{'over-draft %':>14}{'over tok/rd':>12}"
          f"{'under %':>9}{'under tok/rd':>13}")
    for d, m in res["taxonomy_by_domain"].items():
        print(f"{d:<12}{m['n_rounds']:>8}{m['over_draft_round_pct']:>14}"
              f"{m['over_draft_tokens_per_round']:>12}"
              f"{m['under_draft_round_pct']:>9}"
              f"{m['under_draft_tokens_per_round']:>13}")
    print(f"\nEntropy calibration (P(accept) per bin; monotone decreasing = "
          f"{res['calibration_monotone_decreasing']}):")
    for c in res["entropy_calibration"]:
        if c["n"]:
            print(f"  {c['entropy_bin']:>8}: {c['p_accept']:<6} (n={c['n']})")
    print("\nCandidate-set ablation (serving efficiency):")
    for name, m in res["candidate_set"].items():
        print(f"  {name:<22} oracle={m['oracle_eff']}  entropy_stop={m['entropy_stop_eff']}")
    print(f"  best fixed: L={res['best_fixed']['L']} @ {res['best_fixed']['eff']}")
    print(f"  cold start: {res['cold_start']}")
    print(f"  batch interference: {res['batch_interference']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default="test")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = run(a.run, a.eval, a.ratio, a.tau)
    _print_tax(res)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
