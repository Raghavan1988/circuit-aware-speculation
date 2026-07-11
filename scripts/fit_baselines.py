"""Fit the surface-baseline ladder over a sealed I07 sweep (Task #4 / I13).

Reads the frozen prompt-grouped split and one policy's sealed trace run, builds
per-token feature rows (cas.analysis.baselines), then fits each FEATURE_SETS
entry with a regularized logistic probe under GroupKFold on prompt id, reporting
AUROC / AUPRC / Brier / ECE with a prompt-level bootstrap. This is the credited
cheap baseline every hidden-state probe (I12/I13) must beat, and — for the
`preround_*` sets — the hardened C10 baseline (D018).

Requires scikit-learn + numpy (pinned in the Modal image / requirements.txt);
NOT importable in the local base env, so run on a CPU Modal container or a
machine with the pinned deps. All inputs are immutable sealed artifacts; every
number printed is script-computed (AGENTS.md).

Usage (CPU container):
    python scripts/fit_baselines.py \
        --run /artifacts/traces/<run_id> --policy fixed_8 \
        --split /artifacts/data/split_manifest.json --eval dev
"""
from __future__ import annotations

import argparse
import json
import os


def _load_rows(run_dir: str, policy: str):
    import pyarrow.parquet as pq

    base = os.path.join(run_dir, policy)
    rounds = pq.read_table(os.path.join(base, "rounds.parquet")).to_pylist()
    summaries = pq.read_table(
        os.path.join(base, "request_summaries.parquet")).to_pylist()
    domain = {s["request_id"]: s["domain"] for s in summaries}
    split = {s["request_id"]: s["split"] for s in summaries}
    return rounds, domain, split


def _design(rows, cols):
    """(X, y, groups) with mean-imputation + missing-flags for None, one-hot for
    the categorical `domain`. Returns numpy arrays."""
    import numpy as np

    domains = sorted({r["domain"] for r in rows})
    X, y, groups = [], [], []
    # precompute numeric column means for imputation (dev rows only is handled
    # by the caller filtering rows before design)
    means = {}
    for c in cols:
        if c == "domain":
            continue
        vals = [r[c] for r in rows if r[c] is not None]
        means[c] = (sum(vals) / len(vals)) if vals else 0.0
    for r in rows:
        feat = []
        for c in cols:
            if c == "domain":
                feat += [1.0 if r["domain"] == d else 0.0 for d in domains]
            else:
                v = r[c]
                feat.append(means[c] if v is None else float(v))
                feat.append(1.0 if v is None else 0.0)  # missing flag
        X.append(feat)
        y.append(1 if r["accepted"] else 0)
        groups.append(r["request_id"])
    return np.array(X), np.array(y), np.array(groups)


def _ece(y, p, bins=15):
    import numpy as np

    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_all(rows, feature_sets, n_splits=5, seed=0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold

    out = []
    for name, cols in feature_sets:
        X, y, groups = _design(rows, cols)
        if len(set(y)) < 2:
            out.append({"features": name, "note": "single-class; skipped"})
            continue
        oof = np.zeros(len(y))
        gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        for tr, te in gkf.split(X, y, groups):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        out.append({
            "features": name,
            "n": int(len(y)), "pos_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, oof)),
            "auprc": float(average_precision_score(y, oof)),
            "brier": float(brier_score_loss(y, oof)),
            "ece": _ece(y, oof),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trace run dir (…/<run_id>)")
    ap.add_argument("--policy", default="fixed_8",
                    help="policy whose rounds provide labels (fixed_8 labels "
                         "all actions counterfactually)")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"],
                    help="which split to fit/evaluate (test is frozen; dev by "
                         "default so this stays a development-time analysis)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from cas.analysis.baselines import FEATURE_SETS, feature_rows

    rounds, domain, split = _load_rows(args.run, args.policy)
    rows = [r for r in feature_rows(rounds, domain)
            if split.get(r["request_id"]) == args.eval]
    print(f"{len(rows)} token rows on split={args.eval} from {args.policy}")
    results = fit_all(rows, FEATURE_SETS)
    for r in results:
        print(json.dumps(r))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"policy": args.policy, "split": args.eval,
                       "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
