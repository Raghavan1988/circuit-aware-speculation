"""Score one pre-round candidate signal against the frozen baseline (I23/I13, D023).

Integration glue for the generator-critic loop (docs/generator_critic.md). Given a
FeatureSpec, this:

  1. loads the target-frontier activation artifact for a sealed run
     (/artifacts/probes/<run_id>/frontier/, written by capture_frontier_activations);
  2. builds the candidate feature matrix via cas.autoresearch.features.build_features;
  3. builds the FROZEN pre-round baseline (`preround_hardened`: recent-acceptance
     EMA + previous-round target frontier entropy/margin) at ROUND granularity,
     reusing the leakage-safe walk in cas.analysis.baselines.feature_rows;
  4. scores incremental lift + equal-capacity controls + decision-regret via
     cas.autoresearch.eval.incremental_lift, under prompt-grouped GroupKFold OOF.

Every number printed is script-computed from immutable sealed artifacts (AGENTS.md).
Selection is dev-only; the test split is never used for tuning. The frozen bar is
`preround_hardened` (~0.73 AUROC on Qwen-v1); a candidate is interesting only if it
beats that bar AND its equal-dimensional norm-matched/random controls.

Requires numpy + scikit-learn + pyarrow (pinned in the Modal image); NOT importable
in the local base env, so run on a CPU Modal container or a pinned machine.

Usage:
    PYTHONPATH=src python scripts/fit_autoresearch.py \
        --run sweep-2026-07-11T203836 --eval dev \
        --spec-json '{"name":"raw_L18","family":"raw","layers":[18],"params":{}}' \
        --out /artifacts/analysis/autoresearch/raw_L18.json

    # or evaluate the whole default seed library:
    PYTHONPATH=src python scripts/fit_autoresearch.py --run <run_id> --eval dev --all
"""
from __future__ import annotations

import argparse
import json
import os

from cas.autoresearch.types import (FRONTIER_SUBDIR, FeatureSpec,
                                     frontier_acts_filename)


def _load_frontier(run_dir: str, layers):
    """Return (acts_by_layer {L: ndarray(n, d)}, meta_rows list) in artifact order."""
    import numpy as np
    import pyarrow.parquet as pq

    fdir = os.path.join(run_dir, FRONTIER_SUBDIR)
    meta = pq.read_table(os.path.join(fdir, "frontier_metadata.parquet")).to_pylist()
    acts = {}
    for L in layers:
        path = os.path.join(fdir, frontier_acts_filename(L))
        acts[L] = np.load(path)
        if acts[L].shape[0] != len(meta):
            raise ValueError(
                f"layer {L} acts rows {acts[L].shape[0]} != metadata rows {len(meta)}")
    return acts, meta


def _baseline_by_round(run_dir: str):
    """Frozen `preround_hardened` features per (request_id, round_id).

    Reuses cas.analysis.baselines.feature_rows (the leakage-safe walk: history EMA
    over strictly prior rounds; prev-round target frontier entropy/margin) and
    collapses its per-drafted-token rows to one record per round (these fields are
    round-constant), keyed by (request_id, round_id).
    """
    import pyarrow.parquet as pq

    from cas.analysis.baselines import feature_rows

    base = os.path.join(run_dir, "fixed_8")
    rounds = pq.read_table(os.path.join(base, "rounds.parquet")).to_pylist()
    summ = pq.read_table(
        os.path.join(base, "request_summaries.parquet")).to_pylist()
    domain = {s["request_id"]: s["domain"] for s in summ}
    by_key = {}
    for r in feature_rows(rounds, domain):
        key = (r["request_id"], r["round_id"])
        if key not in by_key:  # first (offset 0) row carries the round-level values
            by_key[key] = {
                "history_ema": r["history_ema"],
                "prev_target_entropy": r["prev_target_entropy"],
                "prev_target_margin": r["prev_target_margin"],
            }
    return by_key


# Frozen baseline columns (the PREROUND_BASELINE = "preround_hardened" stack).
_BASE_COLS = ("history_ema", "prev_target_entropy", "prev_target_margin")


def _baseline_design(meta_rows, base_by_key):
    """X_base for meta_rows (same order), mean-imputed + missing-flag per column
    (mirrors scripts/fit_baselines._design)."""
    import numpy as np

    vals = {c: [] for c in _BASE_COLS}
    recs = []
    for m in meta_rows:
        rec = base_by_key.get((m["request_id"], m["round_id"]), {})
        recs.append(rec)
        for c in _BASE_COLS:
            v = rec.get(c)
            if v is not None:
                vals[c].append(float(v))
    means = {c: (sum(vs) / len(vs) if vs else 0.0) for c, vs in vals.items()}
    X = []
    for rec in recs:
        feat = []
        for c in _BASE_COLS:
            v = rec.get(c)
            feat.append(means[c] if v is None else float(v))
            feat.append(1.0 if v is None else 0.0)  # missing flag
        X.append(feat)
    return np.array(X, dtype="float64")


def _subset(acts, meta, eval_split):
    """Keep rows whose split == eval_split; subset acts + meta consistently."""
    keep = [i for i, m in enumerate(meta) if m.get("split") == eval_split]
    acts_s = {L: A[keep] for L, A in acts.items()}
    meta_s = [meta[i] for i in keep]
    return acts_s, meta_s


def score_spec(spec: FeatureSpec, acts, meta, base_by_key, eval_split, seed=0):
    import numpy as np

    from cas.autoresearch.eval import incremental_lift
    from cas.autoresearch.features import build_features

    acts_s, meta_s = _subset(acts, meta, eval_split)
    if not meta_s:
        return {"spec": spec.name, "note": f"no rows on split={eval_split}"}
    X_cand = build_features(spec, acts_s, meta_s)
    X_base = _baseline_design(meta_s, base_by_key)
    y = np.array([1 if m["accept"] else 0 for m in meta_s])
    groups = np.array([m["request_id"] for m in meta_s])
    res = incremental_lift(X_base, X_cand, y, groups, seed=seed)
    res["spec"] = {"name": spec.name, "family": spec.family,
                   "layers": list(spec.layers), "params": spec.params}
    res["eval_split"] = eval_split
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trace run id (…/probes/<run>)")
    ap.add_argument("--artifacts-root", default="/artifacts/probes")
    ap.add_argument("--traces-root", default="/artifacts/traces")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"],
                    help="split to fit/evaluate; test is frozen (dev by default)")
    ap.add_argument("--layers", default="6,12,18,24")
    ap.add_argument("--spec-json", default="",
                    help='one FeatureSpec as JSON: {"name","family","layers","params"}')
    ap.add_argument("--all", action="store_true",
                    help="evaluate the default seed library instead of --spec-json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.layers.split(","))
    run_dir = os.path.join(args.artifacts_root, args.run)
    traces_dir = os.path.join(args.traces_root, args.run)

    acts, meta = _load_frontier(run_dir, layers)
    base_by_key = _baseline_by_round(traces_dir)

    if args.all:
        from cas.autoresearch.features import default_seed_specs
        specs = default_seed_specs(layers)
    elif args.spec_json:
        d = json.loads(args.spec_json)
        specs = [FeatureSpec(name=d["name"], family=d["family"],
                             layers=tuple(d["layers"]), params=d.get("params", {}))]
    else:
        ap.error("provide --spec-json '<json>' or --all")

    results = []
    for spec in specs:
        res = score_spec(spec, acts, meta, base_by_key, args.eval, seed=args.seed)
        results.append(res)
        print(json.dumps(res))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"run": args.run, "eval": args.eval, "results": results},
                      f, indent=2)


if __name__ == "__main__":
    main()
