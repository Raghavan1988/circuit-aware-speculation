"""Layerwise acceptance probes + incremental-information test (I12/I13, C01).

Loads captured draft activations (cas.capture / capture_activations) and the
sealed surface features for the SAME token rows, then fits regularized logistic
probes under prompt-grouped GroupKFold and asks the decisive question:

  * does a hidden-state probe beat the hardened surface baseline (~0.84)?      (C01)
  * does hidden ADD information over the surface stack (hidden ⊕ surface)?     (C01/I13)
  * at which layer does acceptance information peak?                            (C02)

All splits are prompt-grouped; hidden features are standardized; every number is
script-computed. Run on a CPU Modal container (sklearn/pyarrow pinned there).
"""
from __future__ import annotations

import argparse
import json
import os


def _load(probe_dir, run_dir, layers, eval_split, aligned_only=True):
    import numpy as np
    import pyarrow.parquet as pq

    from cas.analysis.baselines import feature_rows

    meta = pq.read_table(os.path.join(probe_dir, "metadata.parquet")).to_pylist()
    # surface features for the same rows, keyed by (request_id, round_id, offset)
    rounds = pq.read_table(os.path.join(run_dir, "fixed_8", "rounds.parquet")).to_pylist()
    summ = pq.read_table(
        os.path.join(run_dir, "fixed_8", "request_summaries.parquet")).to_pylist()
    domain = {s["request_id"]: s["domain"] for s in summ}
    surf = {(r["request_id"], r["round_id"], r["proposal_offset"]): r
            for r in feature_rows(rounds, domain)}

    keep = [i for i, m in enumerate(meta)
            if m["split"] == eval_split and (m["aligned"] or not aligned_only)
            and (m["request_id"], m["round_id"], m["offset"]) in surf]
    acts = {L: np.load(os.path.join(probe_dir, f"acts_L{L}.npy"))[keep] for L in layers}
    rows = [meta[i] for i in keep]
    y = np.array([m["label"] for m in rows])
    groups = np.array([m["request_id"] for m in rows])
    # surface design (the hardened surface stack) for the same rows
    SURF_COLS = ("draft_entropy", "draft_margin", "history_ema",
                 "proposal_offset", "output_pos")
    S = []
    for m in rows:
        r = surf[(m["request_id"], m["round_id"], m["offset"])]
        feat = []
        for c in SURF_COLS:
            v = r.get(c)
            feat += [0.0 if v is None else float(v), 1.0 if v is None else 0.0]
        S.append(feat)
    return acts, np.array(S), y, groups, len(rows)


def _fit(X, y, groups, seed=0, standardize=True):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return {"auroc": float(roc_auc_score(y, oof)),
            "auprc": float(average_precision_score(y, oof)),
            "brier": float(brier_score_loss(y, oof))}


def run(probe_dir, run_dir, layers, eval_split="dev"):
    import numpy as np

    acts, S, y, groups, n = _load(probe_dir, run_dir, layers, eval_split)
    out = {"n_rows": n, "pos_rate": float(y.mean()), "eval_split": eval_split,
           "surface_only": _fit(S, y, groups), "layers": {}}
    for L in layers:
        H = acts[L].astype("float32")
        hid = _fit(H, y, groups)
        comb = _fit(np.hstack([H, S]), y, groups)
        out["layers"][str(L)] = {"hidden_only": hid, "hidden_plus_surface": comb}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--layers", default="6,12,18,24")
    ap.add_argument("--eval", default="dev")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    res = run(args.probe_dir, args.run_dir, layers, args.eval)

    s = res["surface_only"]["auroc"]
    print(f"\nrows={res['n_rows']}  pos_rate={res['pos_rate']:.3f}  "
          f"split={res['eval_split']}")
    print(f"SURFACE baseline AUROC = {s:.4f}  (the bar to beat)")
    print(f"{'layer':>6} {'hidden':>8} {'hid+surf':>9} {'Δ vs surface':>13}")
    for L, d in res["layers"].items():
        h = d["hidden_only"]["auroc"]
        c = d["hidden_plus_surface"]["auroc"]
        print(f"{L:>6} {h:>8.4f} {c:>9.4f} {c - s:>+13.4f}")
    best = max(res["layers"].items(), key=lambda kv: kv[1]["hidden_only"]["auroc"])
    print(f"\nbest hidden-only: layer {best[0]} @ {best[1]['hidden_only']['auroc']:.4f} "
          f"({'BEATS' if best[1]['hidden_only']['auroc'] > s else 'does NOT beat'} "
          f"surface {s:.4f})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
