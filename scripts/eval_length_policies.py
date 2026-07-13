"""RQ2 (roadmap Phase 1): does an adaptive draft-length controller beat the best
fixed length? Offline replay over the sealed fixed_8 traces.

Every fixed_8 round recorded the draft's per-position entropy and the
counterfactual per-position match (draft argmax vs target argmax) for all 8
proposals, so we can replay ANY length policy exactly without re-running a model:
a policy picks a realized length L per round, the accepted count is the leading
all-match prefix of the first L proposals (D018.3), and the round emits
accepted+1 tokens (the +1 is the target's bonus token). This is faithful because
the entropies and matches are the draft's own sealed greedy trajectory.

Metrics are latency-independent so the launch-bound draft cost cannot erase a
controller win:
  * yield         = emitted tokens per round (higher is better)
  * wasted/emit   = drafted-but-rejected tokens per emitted token (lower better)
  * efficiency    = emitted tokens per unit compute, under two cost bases:
      - launch  : every forward costs 1 (draft forward approx = verify forward,
                  the measured launch-bound reality on this harness)
      - serving : verify = 1, each draft forward = `draft_ratio` (a serving-grade
                  draft where the small model is genuinely cheap; default 0.1)

Policies: fixed L; entropy-stop (SVIP, tau tuned on dev); recent-acceptance
(content-aware via history EMA); and the per-round oracle (upper bound). Pure CPU.
"""
from __future__ import annotations

import argparse
import json
import os

ACTIONS = (0, 1, 2, 3, 4, 6, 8)


def _read(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _rounds_by_request(run_dir):
    rows = _read(os.path.join(run_dir, "fixed_8", "rounds.parquet"))
    by_req: dict = {}
    for r in rows:
        proposed = list(r["proposed_token_ids"] or ())
        targ = list(r["target_argmax_ids"] or ())
        ents = list(r["draft_entropy"] or ())
        if not proposed:
            continue
        match = tuple(proposed[i] == targ[i] for i in range(min(len(proposed), len(targ))))
        by_req.setdefault(r["request_id"], []).append(
            {"round_id": r["round_id"], "match": match, "ent": ents})
    for rid in by_req:
        by_req[rid].sort(key=lambda x: x["round_id"])
    return by_req


def _split_of(run_dir):
    summ = _read(os.path.join(run_dir, "fixed_8", "request_summaries.parquet"))
    data_root = os.path.dirname(os.path.dirname(run_dir.rstrip("/")))
    with open(os.path.join(data_root, "data", "split_manifest.json")) as f:
        assign = json.load(f)["assignment"]
    return {s["request_id"]: assign.get(s["prompt_hash"], "unknown") for s in summ}


def accepted_under(match, L):
    k = 0
    for m in match[:L]:
        if not m:
            break
        k += 1
    return k


# ---- policies: (round_state) -> realized L; some carry per-request memory ----
def fixed_policy(L):
    def p(rnd, mem):
        return L
    return p


def entropy_stop_policy(tau):
    def p(rnd, mem):
        L = 0
        for e in rnd["ent"][:8]:
            if e is None or e > tau:
                break
            L += 1
        return L
    return p


def history_policy(alpha=0.3, lo=0.55, hi=0.8):
    """Content-aware via recent acceptance EMA: draft long when recent acceptance
    is high, short when low. Deployable pre-round signal (prior rounds only)."""
    def p(rnd, mem):
        ema = mem.get("ema")
        if ema is None:
            return 4  # neutral start
        return 8 if ema >= hi else (4 if ema >= lo else 2)
    return p


def oracle_policy(cost_of):
    def p(rnd, mem):
        best, bestv = 0, -1.0
        for L in ACTIONS:
            if L > len(rnd["match"]):
                continue
            emit = accepted_under(rnd["match"], L) + 1
            v = emit / cost_of(L)
            if v > bestv:
                best, bestv = L, v
        return best
    return p


def cost_launch(L):
    return L + 1.0


def make_cost_serving(ratio):
    def c(L):
        return 1.0 + L * ratio
    return c


def evaluate(rounds, policy, cost_serving, alpha=0.3):
    emit = drafted = accepted = n = 0
    fw_launch = fw_serving = 0.0
    for req_rounds in rounds:
        mem: dict = {}
        for rnd in req_rounds:
            L = policy(rnd, mem)
            k = accepted_under(rnd["match"], L)
            e = k + 1
            emit += e; drafted += L; accepted += k; n += 1
            fw_launch += cost_launch(L)
            fw_serving += cost_serving(L)
            # update recent-acceptance EMA (fraction of drafted accepted this round)
            frac = (k / L) if L > 0 else 1.0
            mem["ema"] = frac if mem.get("ema") is None else (
                alpha * frac + (1 - alpha) * mem["ema"])
    wasted = drafted - accepted
    return {"yield": emit / n, "wasted_per_emit": wasted / emit,
            "eff_launch": emit / fw_launch, "eff_serving": emit / fw_serving,
            "n_rounds": n, "emit": emit, "drafted": drafted, "accepted": accepted,
            "fw_serving": fw_serving,
            "accept_rate": (accepted / drafted) if drafted else 0.0}


def _tune_tau(rounds, cost_serving):
    """Pick the entropy threshold maximizing serving efficiency (dev only)."""
    taus = [round(0.2 * i, 2) for i in range(1, 26)]  # 0.2 .. 5.0
    best_tau, bestv = None, -1.0
    for tau in taus:
        v = evaluate(rounds, entropy_stop_policy(tau), cost_serving)["eff_serving"]
        if v > bestv:
            best_tau, bestv = tau, v
    return best_tau


def run(run_dir, eval_split="dev", draft_ratio=0.1, frozen_tau=None):
    by_req = _rounds_by_request(run_dir)
    split = _split_of(run_dir)
    rounds = [rs for rid, rs in by_req.items() if split.get(rid) == eval_split]
    cost_serving = make_cost_serving(draft_ratio)

    # threshold is ALWAYS tuned on dev; on test we use that frozen value so the
    # reported number is honest held-out generalization (roadmap RQ2 transfer).
    if frozen_tau is None:
        dev_rounds = ([rs for rid, rs in by_req.items() if split.get(rid) == "dev"]
                      if eval_split != "dev" else rounds)
        tau = _tune_tau(dev_rounds, cost_serving)
    else:
        tau = frozen_tau

    results = {}
    for L in ACTIONS:
        results[f"fixed_{L}"] = evaluate(rounds, fixed_policy(L), cost_serving)
    results[f"entropy_stop(tau={tau})"] = evaluate(
        rounds, entropy_stop_policy(tau), cost_serving)
    results["history_ema"] = evaluate(rounds, history_policy(), cost_serving)
    results["ORACLE_launch"] = evaluate(rounds, oracle_policy(cost_launch), cost_serving)
    results["ORACLE_serving"] = evaluate(rounds, oracle_policy(cost_serving), cost_serving)
    return {"eval_split": eval_split, "draft_ratio": draft_ratio, "tau": tau,
            "tau_tuned_on": "dev",
            "n_rounds": results["fixed_8"]["n_rounds"], "policies": results}


def _print(res):
    P = res["policies"]
    print(f"\nRQ2 length-policy evaluation  (split={res['eval_split']}, "
          f"serving draft_ratio={res['draft_ratio']}, rounds={res['n_rounds']})")
    print(f"{'policy':<22}{'yield':>8}{'wasted/emit':>13}{'eff_launch':>12}{'eff_serving':>13}")
    order = [k for k in P if k.startswith('fixed')] + \
            [k for k in P if k.startswith('entropy')] + \
            ['history_ema', 'ORACLE_launch', 'ORACLE_serving']
    for k in order:
        m = P[k]
        print(f"{k:<22}{m['yield']:>8.3f}{m['wasted_per_emit']:>13.3f}"
              f"{m['eff_launch']:>12.4f}{m['eff_serving']:>13.4f}")
    # headline comparisons
    best_fixed_serv = max((k for k in P if k.startswith('fixed')),
                          key=lambda k: P[k]['eff_serving'])
    ctrl = next(k for k in P if k.startswith('entropy'))
    orc = P['ORACLE_serving']['eff_serving']
    bf = P[best_fixed_serv]['eff_serving']
    ce = P[ctrl]['eff_serving']
    print(f"\n[serving-cost basis] best fixed = {best_fixed_serv} @ {bf:.4f}")
    print(f"  entropy-stop controller @ {ce:.4f}  "
          f"({(ce/bf-1)*100:+.1f}% vs best fixed)")
    print(f"  oracle ceiling @ {orc:.4f}  "
          f"(controller captures {(ce-bf)/(orc-bf)*100:.0f}% of the oracle gap "
          f"over best fixed)" if orc > bf else "")
    print(f"[wasted tokens] best fixed {best_fixed_serv}: "
          f"{P[best_fixed_serv]['wasted_per_emit']:.3f}/emit  vs  "
          f"entropy-stop: {P[ctrl]['wasted_per_emit']:.3f}/emit")


def _domain_of(run_dir):
    summ = _read(os.path.join(run_dir, "fixed_8", "request_summaries.parquet"))
    return {s["request_id"]: s["domain"] for s in summ}


def routing_opportunity(run_dir, draft_ratio=0.1):
    """Scopes RQ3 (does routing help?) WITHOUT a draft pool, from the single
    draft's per-domain behavior:
      (a) per-domain acceptance rate — where the general draft is weak, a
          specialized draft would help most (the draft-routing ceiling proxy);
      (b) per-domain optimal length — if it differs, routing the LENGTH by domain
          already beats one global length (content-awareness premise), and the
          gain is a lower bound on content-aware routing value.
    """
    by_req = _rounds_by_request(run_dir)
    dom = _domain_of(run_dir)
    cost_serving = make_cost_serving(draft_ratio)
    domains: dict = {}
    for rid, rs in by_req.items():
        domains.setdefault(dom.get(rid, "unknown"), []).append(rs)

    per = {}
    for d, rounds in domains.items():
        res = {L: evaluate(rounds, fixed_policy(L), cost_serving) for L in ACTIONS}
        bestL = max(ACTIONS, key=lambda L: res[L]["eff_serving"])
        est = evaluate(rounds, entropy_stop_policy(2.0), cost_serving)
        per[d] = {"n_rounds": res[8]["n_rounds"], "accept_rate": res[8]["accept_rate"],
                  "best_L": bestL, "eff_best_fixed": res[bestL]["eff_serving"],
                  "eff_entropy_stop": est["eff_serving"],
                  "_emit_bestL": res[bestL]["emit"], "_fw_bestL": res[bestL]["fw_serving"]}
    all_rounds = [rs for rss in domains.values() for rs in rss]
    g = {L: evaluate(all_rounds, fixed_policy(L), cost_serving) for L in ACTIONS}
    gL = max(ACTIONS, key=lambda L: g[L]["eff_serving"])
    global_eff = g[gL]["eff_serving"]
    routed_eff = (sum(per[d]["_emit_bestL"] for d in per)
                  / sum(per[d]["_fw_bestL"] for d in per))
    return {"draft_ratio": draft_ratio, "global_best_L": gL, "global_eff": global_eff,
            "domain_routed_length_eff": routed_eff,
            "length_routing_gain_pct": (routed_eff / global_eff - 1) * 100,
            "per_domain": {d: {k: v for k, v in per[d].items() if not k.startswith("_")}
                           for d in per}}


def _print_routing(r):
    print(f"\nRQ3 routing-OPPORTUNITY scoping (serving draft_ratio={r['draft_ratio']})")
    print(f"{'domain':<16}{'rounds':>8}{'accept_rate':>12}{'best_L':>8}"
          f"{'eff_fixed':>11}{'eff_estop':>11}")
    for d, m in sorted(r["per_domain"].items(), key=lambda kv: kv[1]["accept_rate"]):
        print(f"{d:<16}{m['n_rounds']:>8}{m['accept_rate']:>12.3f}{m['best_L']:>8}"
              f"{m['eff_best_fixed']:>11.4f}{m['eff_entropy_stop']:>11.4f}")
    print(f"\nglobal best fixed length = L{r['global_best_L']} @ {r['global_eff']:.4f}")
    print(f"route length by domain (own best L) @ {r['domain_routed_length_eff']:.4f}  "
          f"({r['length_routing_gain_pct']:+.1f}% over one global length)")
    ars = [m["accept_rate"] for m in r["per_domain"].values()]
    bls = {m["best_L"] for m in r["per_domain"].values()}
    print(f"acceptance spread across domains: {min(ars):.3f}..{max(ars):.3f}  "
          f"(gap {max(ars)-min(ars):.3f}); per-domain optimal length set = {sorted(bls)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default="dev")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--routing", action="store_true", help="RQ3 opportunity scoping")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.routing:
        res = routing_opportunity(a.run, a.ratio)
        _print_routing(res)
    else:
        res = run(a.run, a.eval, a.ratio)
        _print(res)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
