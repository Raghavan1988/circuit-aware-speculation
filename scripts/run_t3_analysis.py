"""T3 measurement analysis over a sealed I07 sweep (all three tasks, CPU-only).

Runs against the immutable sealed Parquet corpus and emits one script-generated
JSON report — no hand-typed numbers (AGENTS.md). Three sub-analyses:

  T3.1 Oracle headroom (D018.3 / Codex go/no-go):
        match vectors come from the fixed_8 run (labels ALL actions
        counterfactually); the per-action cost profile is MEASURED from each
        action's own policy run (skip -> 0, fixed_L -> L). Reports oracle
        tokens-per-cost vs every fixed action and the relative headroom over the
        best fixed action. D018 tripwire: headroom < ~5% stops controller work.

  T3.2 Surface-baseline strength ladder (I13 / hardened C10):
        the cheap logistic baseline every hidden-state probe must beat, fit with
        GroupKFold on prompt id over the dev split (reuses scripts/fit_baselines).

  T3.3 Acceptance atlas (C04 / I18):
        per-token counterfactual acceptance (target_match) by overlapping token
        category x generation phase, with a prompt-grouped bootstrap CI. Tokens
        are annotated at analysis time (the sweep wrote no annotations) using the
        run's own tokenizer + cas.annotate.

Usage (CPU Modal container; deps pinned in the image):
    python scripts/run_t3_analysis.py --run /artifacts/traces/<run_id> \
        --out /artifacts/analysis/<run_id>/t3_report.json
"""
from __future__ import annotations

import argparse
import json
import os

# policy dir -> action length it measures the cost of
POLICY_ACTION = {
    "skip": 0, "fixed_1": 1, "fixed_2": 2, "fixed_3": 3,
    "fixed_4": 4, "fixed_6": 6, "fixed_8": 8,
}
LABEL_POLICY = "fixed_8"  # rounds here label every action (realized len up to 8)


def _read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _match_vectors(rounds):
    """Per-round draft/target agreement vectors from a labelling policy run."""
    matches = []
    for r in rounds:
        proposed = r["proposed_token_ids"] or []
        targ = r["target_argmax_ids"] or []
        matches.append(tuple(p == t for p, t in zip(proposed, targ)))
    return [m for m in matches if m]  # drop skip/empty rounds (no drafted token)


def _mean_components(rounds):
    """Mean of each latency component (ns) over a policy's rounds."""
    tot: dict[str, float] = {}
    n = 0
    for r in rounds:
        lat = r["latency_ns"]
        if isinstance(lat, str):
            lat = json.loads(lat)
        n += 1
        for k, v in lat.items():
            tot[k] = tot.get(k, 0.0) + v
    return {k: v / n for k, v in tot.items()} if n else {}


def _oracle_under(matches, costs):
    from cas.analysis.oracle import oracle_policy_value
    res = oracle_policy_value(matches, costs, actions=tuple(sorted(costs)))
    res["best_fixed"] = list(res["best_fixed"])
    res["costs_ns"] = {int(k): v for k, v in costs.items()}
    return res


def oracle_headroom(run_dir):
    """Oracle headroom under two cost bases, because the routing decision hinges
    on which cost the controller actually pays:

      * full  — every measured component (draft+verify+controller+tracing).
        This is the AGENTS.md end-to-end basis, but it charges longer actions
        for per-position TRACING that a deployed controller would not emit.
      * compute — draft+verify only (the GPU work a deployed router pays).
        A DEPLOYMENT ESTIMATE, explicitly NOT an end-to-end latency claim.

    D018 tripwire (~5%) is read against the compute basis for the go/no-go, with
    the full basis reported alongside for honesty."""
    comps = {}          # per action: mean component dict
    for pol, L in POLICY_ACTION.items():
        rr = _read_parquet(os.path.join(run_dir, pol, "rounds.parquet"))
        comps[L] = _mean_components(rr)
    costs_full = {L: sum(c.values()) for L, c in comps.items() if c}
    costs_compute = {L: (c.get("draft", 0.0) + c.get("verify", 0.0))
                     for L, c in comps.items() if c}

    label_rounds = _read_parquet(os.path.join(run_dir, LABEL_POLICY, "rounds.parquet"))
    matches = _match_vectors(label_rounds)
    return {
        "label_policy": LABEL_POLICY,
        "mean_components_ns": {int(L): comps[L] for L in comps},
        "full_basis": _oracle_under(matches, costs_full),
        "compute_basis": _oracle_under(matches, costs_compute),
    }


def _split_by_request(run_dir, summaries, data_dir="data"):
    """Recover the prompt-grouped split per request.

    The sweep stamped `RequestSummary.split` via assignment.get(prompt_id) but
    the split manifest is keyed by prompt_hash, so the stored column is all
    "unknown" (see analysis note; corpus otherwise valid). Recover it here by
    joining request -> prompt_hash -> split through the frozen manifest, which
    is deterministic from the immutable inputs (no re-run needed).

    `data_dir` selects the corpus manifest ("data" for v1, "data_v2" for the
    v2 corpus), mirroring the D025 capture threading."""
    data_root = os.path.dirname(os.path.dirname(run_dir.rstrip("/")))  # .../ artifacts
    manifest_path = os.path.join(data_root, data_dir, "split_manifest.json")
    with open(manifest_path) as f:
        assignment = json.load(f)["assignment"]  # {prompt_hash: dev|test}
    return {s["request_id"]: assignment.get(s["prompt_hash"], "unknown")
            for s in summaries}


def cost_sensitivity(run_dir, draft_ms_per_token=(27.2, 15.0, 8.0, 4.0, 2.0)):
    """(a) CPU-only what-if: hold the sealed match vectors fixed and replace the
    (confounded) measured draft cost with a hypothetical per-token draft latency,
    then recompute oracle headroom + best fixed action. verify/controller/tracing
    stay at their measured per-action values (one target forward per round).

    Also emits confound checks so a reader can see what the profile imports."""
    from cas.analysis.oracle import oracle_policy_value

    comps = {}
    for pol, L in POLICY_ACTION.items():
        rr = _read_parquet(os.path.join(run_dir, pol, "rounds.parquet"))
        comps[L] = _mean_components(rr)
    # non-draft (target verify + controller + tracing) per action, measured
    non_draft = {L: (c.get("verify", 0.0) + c.get("controller", 0.0)
                     + c.get("tracing", 0.0)) for L, c in comps.items()}
    measured_draft = {L: c.get("draft", 0.0) for L, c in comps.items()}

    label_rounds = _read_parquet(os.path.join(run_dir, LABEL_POLICY, "rounds.parquet"))
    matches = _match_vectors(label_rounds)

    # --- confound checks -------------------------------------------------
    verify_ms = {L: comps[L].get("verify", 0.0) / 1e6 for L in sorted(comps)}
    lens = [len(m) for m in matches]
    confounds = {
        # verify should be ~monotone non-decreasing in block size (more
        # positions per forward). Non-monotonicity => scheduling/hardware jitter
        # imported into the cost profile.
        "verify_ms_by_action": verify_ms,
        "verify_monotonic_in_L": all(
            verify_ms[a] <= verify_ms[b] + 1.0  # 1ms slack
            for a, b in zip(sorted(verify_ms), sorted(verify_ms)[1:])),
        "measured_draft_ms_per_token": {
            L: (measured_draft[L] / 1e6 / L if L else 0.0) for L in sorted(comps)},
        # all fixed_8 match vectors should be length 8 (draft always proposes 8;
        # eos/max_new truncate at COMMIT, not draft). Short vectors => a real
        # realized-length surprise worth investigating.
        "match_len_min": min(lens), "match_len_max": max(lens),
        "match_len_all_8": all(x == 8 for x in lens), "n_matches": len(lens),
    }

    rows = []
    for dms in draft_ms_per_token:
        draft_ns = dms * 1e6
        costs = {L: non_draft[L] + L * draft_ns for L in non_draft}
        r = oracle_policy_value(matches, costs, actions=tuple(sorted(costs)))
        rows.append({
            "draft_ms_per_token": dms,
            "best_fixed_L": r["best_fixed"][0],
            "headroom_pct": round(r["headroom"] * 100, 2),
            "throughput_by_L": {int(L): r["fixed"][L] for L in r["fixed"]},
        })
    return {"label_policy": LABEL_POLICY, "confounds": confounds, "sweep": rows}


def surface_baselines(run_dir, eval_split="dev"):
    from cas.analysis.baselines import FEATURE_SETS, feature_rows
    from scripts.fit_baselines import fit_all

    rounds = _read_parquet(os.path.join(run_dir, LABEL_POLICY, "rounds.parquet"))
    summaries = _read_parquet(
        os.path.join(run_dir, LABEL_POLICY, "request_summaries.parquet"))
    domain = {s["request_id"]: s["domain"] for s in summaries}
    split = _split_by_request(run_dir, summaries)
    rows = [r for r in feature_rows(rounds, domain)
            if split.get(r["request_id"]) == eval_split]
    return {"policy": LABEL_POLICY, "eval_split": eval_split,
            "n_rows": len(rows), "split_recovered_via": "prompt_hash+manifest",
            "results": fit_all(rows, FEATURE_SETS)}


def _load_tokenizer(run_dir):
    """Load the run's own tokenizer from run_metadata.json (numerical identity)."""
    from transformers import AutoTokenizer
    with open(os.path.join(run_dir, LABEL_POLICY, "run_metadata.json")) as f:
        meta = json.load(f)
    return AutoTokenizer.from_pretrained(
        meta["tokenizer_id"], revision=meta.get("tokenizer_revision") or None)


def atlas(run_dir, min_n=50, n_boot=1000):
    from cas.annotate.categories import annotate_categories
    from cas.annotate.phases import annotate_phase
    from cas.analysis.atlas import atlas_table

    tok = _load_tokenizer(run_dir)
    tokens = _read_parquet(os.path.join(run_dir, LABEL_POLICY, "tokens.parquet"))

    # cache piece + categories per draft token id (tokenizer-light, id-stable)
    cat_cache: dict[int, list[str]] = {}

    def cats_for(tid):
        c = cat_cache.get(tid)
        if c is None:
            piece = tok.convert_ids_to_tokens(int(tid))
            c = sorted(annotate_categories(piece, token_id=int(tid)))
            cat_cache[tid] = c
        return c

    rows = []
    for t in tokens:
        rows.append({
            "request_id": t["request_id"],
            # counterfactual per-position acceptance: unbiased across positions
            # (committed `accepted` censors everything after the first rejection)
            "accepted": bool(t["target_match"]),
            "categories": cats_for(t["draft_token_id"]),
            "phase": annotate_phase(int(t["token_position"])),
        })
    return {"policy": LABEL_POLICY, "acceptance_label": "target_match",
            "n_tokens": len(rows), "min_n": min_n,
            "table": atlas_table(rows, min_n=min_n, n_boot=n_boot)}


# C04 contrast pools, fixed from the recorded Qwen-v1 atlas pattern (M3 note,
# claims ledger 2026-07-12): structural/code-adjacent tokens accept high,
# entity/reasoning tokens accept low. The pools are part of the pre-registered
# C04 claim; the test-split pass evaluates exactly these.
C04_HIGH = ("code_delimiter", "operator", "number")
C04_LOW = ("named_entity", "reasoning_transition")


def _pool_prompts(rows, cats):
    """{request_id: [n_tokens, n_accepted]} over rows carrying ANY category in
    `cats` (overlapping labels: a token joins the pool once even if it carries
    several pool categories)."""
    cs = set(cats)
    pool = {}
    for r in rows:
        if cs.intersection(r["categories"]):
            p = pool.setdefault(r["request_id"], [0, 0])
            p[0] += 1
            p[1] += bool(r["accepted"])
    return pool


def _marginal_table(rows, key_fn, min_n, n_boot):
    """Rate + prompt-grouped CI per key (domain / phase / category marginal)."""
    from cas.analysis.atlas import bootstrap_rate_ci

    cells = {}
    for r in rows:
        for key in key_fn(r):
            c = cells.setdefault(key, {"n": 0, "k": 0, "prompts": {}})
            c["n"] += 1
            c["k"] += bool(r["accepted"])
            p = c["prompts"].setdefault(r["request_id"], [0, 0])
            p[0] += 1
            p[1] += bool(r["accepted"])
    out = []
    for key, c in sorted(cells.items()):
        if c["n"] < min_n:
            continue
        lo, hi = bootstrap_rate_ci(c["prompts"], n_boot=n_boot)
        out.append({"key": key, "n": c["n"], "rate": c["k"] / c["n"],
                    "ci_lo": lo, "ci_hi": hi, "n_prompts": len(c["prompts"])})
    out.sort(key=lambda r: r["rate"])
    return out


def atlas_c04(run_dir, eval_split, data_dir="data", min_n=50, n_boot=1000):
    """C04 acceptance atlas on ONE split, domain-controlled (I18 / C04).

    Extends the T3.3 atlas with what the C04 claim needs (landscape §C04
    positioning): (a) split filtering (the T3.3 atlas pooled dev+test); (b) the
    domain-marginal control table (reproducing the domain-grain axis of prior
    work, arXiv:2604.14682); (c) per-category tables WITHIN each domain, so
    category heterogeneity is shown beyond domain identity; (d) the
    pre-registered structural-vs-entity contrast (C04_HIGH vs C04_LOW pools)
    overall and within every domain, with paired prompt-grouped bootstrap CIs.
    Labels are counterfactual per-position `target_match` (unbiased across
    positions). Every number is script-generated from sealed traces."""
    from cas.analysis.atlas import atlas_table, bootstrap_delta_ci
    from cas.annotate.categories import CATEGORY_SET_VERSION, annotate_categories
    from cas.annotate.phases import PHASE_SET_VERSION, annotate_phase

    tok = _load_tokenizer(run_dir)
    tokens = _read_parquet(os.path.join(run_dir, LABEL_POLICY, "tokens.parquet"))
    summaries = _read_parquet(
        os.path.join(run_dir, LABEL_POLICY, "request_summaries.parquet"))
    domain = {s["request_id"]: s["domain"] for s in summaries}
    split = _split_by_request(run_dir, summaries, data_dir=data_dir)

    cat_cache: dict[int, list[str]] = {}

    def cats_for(tid):
        c = cat_cache.get(tid)
        if c is None:
            piece = tok.convert_ids_to_tokens(int(tid))
            c = sorted(annotate_categories(piece, token_id=int(tid)))
            cat_cache[tid] = c
        return c

    rows = []
    for t in tokens:
        rid = t["request_id"]
        if split.get(rid) != eval_split:
            continue
        rows.append({
            "request_id": rid,
            "domain": domain.get(rid, "unknown"),
            "accepted": bool(t["target_match"]),
            "categories": cats_for(t["draft_token_id"]),
            "phase": annotate_phase(int(t["token_position"])),
        })

    def _contrast(sub_rows):
        pa = _pool_prompts(sub_rows, C04_HIGH)
        pb = _pool_prompts(sub_rows, C04_LOW)
        na = sum(v[0] for v in pa.values())
        nb = sum(v[0] for v in pb.values())
        if na < min_n or nb < min_n:
            return {"note": f"insufficient pool ({na} high / {nb} low)",
                    "n_high": na, "n_low": nb}
        d = bootstrap_delta_ci(pa, pb, n_boot=n_boot)
        d.update({"n_high": na, "n_low": nb,
                  "rate_high": sum(v[1] for v in pa.values()) / na,
                  "rate_low": sum(v[1] for v in pb.values()) / nb})
        return d

    domains = sorted({r["domain"] for r in rows})
    report = {
        "run_dir": run_dir, "eval_split": eval_split, "data_dir": data_dir,
        "policy": LABEL_POLICY, "acceptance_label": "target_match",
        "category_set_version": CATEGORY_SET_VERSION,
        "phase_set_version": PHASE_SET_VERSION,
        "min_n": min_n, "n_boot": n_boot,
        "n_tokens": len(rows),
        "n_requests": len({r["request_id"] for r in rows}),
        "contrast_pools": {"high": list(C04_HIGH), "low": list(C04_LOW)},
        "domain_marginal": _marginal_table(
            rows, lambda r: [r["domain"]], min_n, n_boot),
        "phase_marginal": _marginal_table(
            rows, lambda r: [r["phase"]], min_n, n_boot),
        "category_marginal": _marginal_table(
            rows, lambda r: r["categories"] or ["uncategorized"], min_n, n_boot),
        "category_by_phase": atlas_table(rows, min_n=min_n, n_boot=n_boot),
        "within_domain_category": {
            d: _marginal_table([r for r in rows if r["domain"] == d],
                               lambda r: r["categories"] or ["uncategorized"],
                               min_n, n_boot)
            for d in domains},
        "contrast_overall": _contrast(rows),
        "contrast_by_domain": {
            d: _contrast([r for r in rows if r["domain"] == d])
            for d in domains},
    }
    return report


def run(run_dir, out_path=None, eval_split="dev"):
    report = {
        "run_dir": run_dir,
        "t3_1_oracle_headroom": oracle_headroom(run_dir),
        "t3_2_surface_baselines": surface_baselines(run_dir, eval_split),
        "t3_3_acceptance_atlas": atlas(run_dir),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return report


def _print_summary(rep):
    oc = rep["t3_1_oracle_headroom"]
    print("\n=== T3.1 ORACLE HEADROOM (M3 decision gate) ===")
    print("  mean latency components (ns) per action:")
    for L in sorted(oc["mean_components_ns"], key=int):
        c = oc["mean_components_ns"][L]
        parts = " ".join(f"{k}={v/1e6:.2f}ms" for k, v in sorted(c.items()))
        print(f"    L={L}: {parts}")
    for basis in ("compute_basis", "full_basis"):
        o = oc[basis]
        gate = "PROCEED" if o["headroom"] >= 0.05 else "STOP controller work"
        print(f"  [{basis}]  best fixed L={o['best_fixed'][0]}  "
              f"HEADROOM={o['headroom']*100:.2f}%  "
              f"({'tripwire ~5%: ' + gate})")
        for a in sorted(o["fixed"], key=lambda k: int(k)):
            print(f"      L={a}: {o['fixed'][a]:.6g}")
        print(f"      n_rounds={o['n_rounds']} censored={o['censored']}")

    b = rep["t3_2_surface_baselines"]
    print(f"\n=== T3.2 SURFACE BASELINES (split={b['eval_split']}, n={b['n_rows']}) ===")
    for r in b["results"]:
        if "auroc" in r:
            print(f"  {r['features']:<24} AUROC={r['auroc']:.4f} "
                  f"AUPRC={r['auprc']:.4f} Brier={r['brier']:.4f} ECE={r['ece']:.4f}")
        else:
            print(f"  {r['features']:<24} {r.get('note','')}")

    a = rep["t3_3_acceptance_atlas"]
    print(f"\n=== T3.3 ACCEPTANCE ATLAS (n_tokens={a['n_tokens']}, "
          f"label={a['acceptance_label']}, cells>= {a['min_n']}) ===")
    tbl = a["table"]
    print(f"  {len(tbl)} cells. Lowest-acceptance:")
    for r in tbl[:6]:
        print(f"    {r['category']:<20} {r['phase']:<7} rate={r['rate']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] n={r['n']}")
    print("  Highest-acceptance:")
    for r in tbl[-6:]:
        print(f"    {r['category']:<20} {r['phase']:<7} rate={r['rate']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] n={r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="sealed run dir (…/<run_id>)")
    ap.add_argument("--out", default="", help="JSON report path")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    rep = run(args.run, args.out or None, args.eval)
    _print_summary(rep)
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
