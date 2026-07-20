"""Frozen, un-gameable incremental-lift scorer (I13/I23, D023).

The single arbiter of whether a candidate pre-round feature adds *real*
predictive information over the frozen surface baseline (`PREROUND_BASELINE` in
`cas.autoresearch.types`). Everything here is script-computed and deterministic;
no number is ever hand-entered (AGENTS.md).

The scorer is deliberately hard to game:

  * **Prompt-grouped OOF only.** All fits use ``GroupKFold`` on prompt id; rows
    within a prompt are dependent, so token-level random splits are prohibited
    (AGENTS.md). Standardization is fit on the train fold only.
  * **Equal-capacity controls.** A candidate does not merely have to beat the
    baseline AUROC; it must beat two controls that add *exactly the same number
    of columns* as the candidate but carry no information: ``control_random``
    (i.i.d. N(0,1)) and ``control_norm`` (the same noise rescaled to the
    candidate's per-column std, so it matches capacity *and* scale). A "lift"
    over a random feature of equal dimensionality is not a lift.
  * **Prompt-grouped bootstrap CI.** The (combined - base) AUROC delta gets a
    95% CI by resampling *groups* (not rows) with replacement, plus P(delta<=0).

This module imports ONLY numpy, sklearn, stdlib, and `cas.autoresearch.types`
(sklearn lazily, so the module stays importable without it). It never imports
`cas.autoresearch.features` — the loop's pieces stay decoupled.

Mechanistic/"circuit" language stays G2-gated: a survivor here is a "diagnostic
signal", never a "circuit", until interventions pass (D020).
"""
from __future__ import annotations

import numpy as np

# Re-exported for callers that want the name of the frozen bar these X_base
# columns are built from (the scorer itself operates on arrays, not names).
from cas.autoresearch.types import PREROUND_BASELINE  # noqa: F401


def _ece(y, p, bins=15):
    """Expected calibration error (equal-width bins). Mirrors
    ``scripts/fit_baselines.py`` exactly so probes and candidates are scored on
    the identical calibration metric."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def decision_regret(y, p, cost_draft=0.1, reward_accept=1.0, tau=None) -> float:
    """Coarse economic regret of a binary pre-round accept/skip decision.

    This is a *decision proxy*, NOT the authoritative systems metric. The
    authoritative efficiency claim is always wall-clock end-to-end latency (G3,
    which includes controller/tracing/sync/routing overhead). ``decision_regret``
    only asks: if we had to make a one-shot draft-or-skip call per position using
    probability ``p`` and a fixed threshold, how much utility do we leave on the
    table versus an oracle that sees the label?

    Economic model (docs/CLAIMS_LEDGER.md): a draft forward pass is priced at
    ~0.1x a verify (``cost_draft``); an accepted draft position is worth
    ``reward_accept`` (=1.0). The per-position economic breakeven probability is
    ``cost_draft/(cost_draft+reward_accept)`` = 0.1/1.1 ~= 0.091, used as the
    default decision threshold ``tau``.

    Action: draft iff ``p >= tau`` else skip.
    Realized utility per row: draft -> (reward_accept if y==1 else -cost_draft);
                              skip  -> 0.
    Oracle utility per row:   reward_accept * y  (draft only when y==1, else 0).
    regret = mean(oracle_util - realized_util) >= 0 (0 == oracle-optimal).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if tau is None:
        tau = cost_draft / (cost_draft + reward_accept)
    draft = p >= tau
    realized = np.where(draft, np.where(y >= 0.5, reward_accept, -cost_draft), 0.0)
    oracle = reward_accept * y  # draft only when it would be accepted, else skip
    return float(np.mean(oracle - realized))


def _fit_oof(X, y, groups, seed=0, n_splits=5) -> np.ndarray:
    """Prompt-grouped out-of-fold acceptance probabilities.

    Mirrors the proven pattern in ``scripts/fit_probes.py`` /
    ``scripts/fit_baselines.py``: ``GroupKFold``, ``StandardScaler`` fit on the
    train fold only, ``LogisticRegression(max_iter=2000, C=1.0)``. ``n_splits``
    is clamped to the number of distinct groups. A train fold that is
    single-class cannot fit a classifier; that fold falls back to the constant
    train-prior probability (graceful, deterministic).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    oof = np.zeros(len(y), dtype=float)
    n_groups = len(np.unique(groups))
    splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=splits)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            # degenerate train fold: no class contrast -> predict the prior.
            oof[te] = float(np.mean(y[tr])) if len(tr) else 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def _metrics(y, oof) -> dict:
    """Per-model score card: {auroc, auprc, brier, ece, regret}. AUROC/AUPRC are
    None if ``y`` is single-class (undefined); Brier/ECE/regret are always
    defined."""
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)

    y = np.asarray(y)
    oof = np.asarray(oof, dtype=float)
    two_class = len(np.unique(y)) >= 2
    return {
        "auroc": float(roc_auc_score(y, oof)) if two_class else None,
        "auprc": float(average_precision_score(y, oof)) if two_class else None,
        "brier": float(brier_score_loss(y, oof, pos_label=1)) if two_class
        else float(np.mean((oof - y) ** 2)),
        "ece": _ece(y, oof),
        "regret": decision_regret(y, oof),
    }


def _group_bootstrap_delta(y, oof_base, oof_comb, groups, seed=0, n_boot=1000):
    """Prompt-grouped bootstrap of the (combined - base) AUROC delta.

    Resamples GROUPS (prompt ids) with replacement — never rows, because rows
    within a prompt are dependent (AGENTS.md). For each resample, AUROC is
    recomputed on the SAME resampled row set for both the base OOF and the
    combined OOF predictions, and their difference recorded. Returns the delta
    distribution's 2.5/97.5 percentiles, P(delta<=0), and the effective bootstrap
    count (resamples that happened to be single-class are skipped). Deterministic
    via ``np.random.default_rng(seed)``.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    oof_base = np.asarray(oof_base, dtype=float)
    oof_comb = np.asarray(oof_comb, dtype=float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        yb = y[rows]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(
            roc_auc_score(yb, oof_comb[rows]) - roc_auc_score(yb, oof_base[rows]))
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return {"lo": None, "hi": None, "p_delta_le_0": None, "n_boot": 0}
    return {
        "lo": float(np.percentile(deltas, 2.5)),
        "hi": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float(np.mean(deltas <= 0.0)),
        "n_boot": int(deltas.size),
    }


def incremental_lift(X_base, X_cand, y, groups, seed=0, n_splits=5,
                     n_boot=1000) -> dict:
    """Score a candidate pre-round feature's lift over the frozen baseline.

    Fits, under GroupKFold(n_splits) OOF with within-fold standardization:
      base           : X_base
      combined       : hstack([X_base, X_cand])
      control_random : hstack([X_base, R]) where R ~ N(0,1), R.shape == X_cand.shape
      control_norm   : hstack([X_base, Rn]) where Rn is R rescaled to X_cand's
                       per-column std (equal capacity AND matched scale)

    All four designs share the SAME folds (GroupKFold partitions on ``groups``
    only), so deltas are fold-matched. R (and hence Rn) is seeded off ``seed``.

    Returns a dict with per-model {auroc, auprc, brier, ece, regret} under keys
    ``base``/``combined``/``control_random``/``control_norm``, plus:
      deltas         : {auroc, auprc, brier} of combined - base
      beats_baseline : combined.auroc > base.auroc + 1e-4
      beats_controls : combined.auroc > max(ctrl_random.auroc, ctrl_norm.auroc) + 1e-4
      delta_auroc_ci : prompt-grouped bootstrap {lo, hi, p_delta_le_0, n_boot}
      n, pos_rate, n_features_base, n_features_cand

    Note: after within-fold standardization ``control_random`` and
    ``control_norm`` become numerically identical (standardization removes the
    per-column scale that distinguishes them); both are still reported so the
    equal-capacity comparison is explicit and the metric survives a future
    non-standardized configuration.
    """
    X_base = np.asarray(X_base, dtype=float)
    X_cand = np.asarray(X_cand, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X_base.ndim == 1:
        X_base = X_base.reshape(-1, 1)
    if X_cand.ndim == 1:
        X_cand = X_cand.reshape(-1, 1)

    n = len(y)
    n_feat_base = X_base.shape[1]
    n_feat_cand = X_cand.shape[1]

    # Equal-capacity controls: exactly X_cand's shape, seeded/deterministic.
    rng = np.random.default_rng(seed)
    R = rng.standard_normal(size=X_cand.shape)
    cand_std = X_cand.std(axis=0)                       # per-column target scale
    r_std = R.std(axis=0)
    r_std_safe = np.where(r_std == 0.0, 1.0, r_std)
    Rn = R / r_std_safe * cand_std                      # exact per-column std match

    designs = {
        "base": X_base,
        "combined": np.hstack([X_base, X_cand]),
        "control_random": np.hstack([X_base, R]),
        "control_norm": np.hstack([X_base, Rn]),
    }
    oof = {k: _fit_oof(v, y, groups, seed=seed, n_splits=n_splits)
           for k, v in designs.items()}
    models = {k: _metrics(y, v) for k, v in oof.items()}

    def _delta(metric):
        a, b = models["combined"][metric], models["base"][metric]
        return None if (a is None or b is None) else float(a - b)

    a_comb = models["combined"]["auroc"]
    a_base = models["base"]["auroc"]
    a_ctrl = [models["control_random"]["auroc"], models["control_norm"]["auroc"]]
    beats_baseline = (a_comb is not None and a_base is not None
                      and a_comb > a_base + 1e-4)
    ctrl_defined = [a for a in a_ctrl if a is not None]
    beats_controls = (a_comb is not None and len(ctrl_defined) > 0
                      and a_comb > max(ctrl_defined) + 1e-4)

    out = dict(models)
    out.update({
        "deltas": {"auroc": _delta("auroc"), "auprc": _delta("auprc"),
                   "brier": _delta("brier")},
        "beats_baseline": bool(beats_baseline),
        "beats_controls": bool(beats_controls),
        "delta_auroc_ci": _group_bootstrap_delta(
            y, oof["base"], oof["combined"], groups, seed=seed, n_boot=n_boot),
        "n": int(n),
        "pos_rate": float(np.mean(y)),
        "n_features_base": int(n_feat_base),
        "n_features_cand": int(n_feat_cand),
    })
    return out
