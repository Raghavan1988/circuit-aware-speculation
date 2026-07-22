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


# Draft-cost grid for the regret sweep: cost of one wasted draft token relative to
# reward_accept=1.0. 0.1 is the harness price (draft ~0.1x verify); the grid spans
# cheap->expensive so we can see whether a candidate's ranking lift EVER converts
# to a decision advantage as the skip/draft threshold tau=cost/(cost+1) moves off
# the degenerate "always draft" regime (D023; CLAIMS_LEDGER cost-sensitivity note).
COST_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 9.0)


def regret_cost_sweep(y, base_p, comb_p, costs=COST_GRID) -> list:
    """Calibrated decision-regret of base vs combined across draft costs.

    Pure post-hoc on already-calibrated probabilities (no refitting). For each
    ``cost_draft`` the threshold is ``tau = cost/(cost+1)``. Returns per cost:
    {cost_draft, tau, base_regret, combined_regret, delta_regret (comb-base),
    helps (delta < -1e-6)}. Still a COARSE proxy — wall-clock is authoritative
    (G3) — but it answers the decisive systems question: does the candidate's
    ranking lift reduce decision regret at ANY cost ratio, or only in a regime
    where every model makes the same call?
    """
    out = []
    for c in costs:
        br = decision_regret(y, base_p, cost_draft=c)
        cr = decision_regret(y, comb_p, cost_draft=c)
        out.append({"cost_draft": float(c), "tau": float(c / (c + 1.0)),
                    "base_regret": br, "combined_regret": cr,
                    "delta_regret": float(cr - br),
                    "helps": bool(cr < br - 1e-6)})
    return out


def _regret_vec(y, p, costs) -> np.ndarray:
    """Vectorized ``decision_regret`` over a cost array for one (y, p).

    Numerically identical to ``decision_regret`` computed per cost (draft iff
    p >= tau=cost/(cost+1); accept:+1, reject:-cost; skip:0; regret = mean(y -
    realized)), but evaluated for all costs at once — used by the bootstrap so the
    grouped resampling stays cheap. Returns an array aligned to ``costs``.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    costs = np.asarray(costs, dtype=float)
    tau = costs / (costs + 1.0)                              # (C,)
    draft = p[:, None] >= tau[None, :]                       # (n, C)
    gain = np.where(y[:, None] >= 0.5, 1.0, -costs[None, :])  # accept:+1, reject:-cost
    realized = np.where(draft, gain, 0.0)                    # skip -> 0
    return (y[:, None] - realized).mean(axis=0)              # oracle(=y) - realized


def _regret_sweep_ci(y, base_p, comb_p, groups, costs=COST_GRID, seed=0,
                     n_boot=1000) -> list:
    """Prompt-grouped bootstrap CI on the (combined - base) regret delta per cost.

    Resamples GROUPS (prompt ids) with replacement — never rows, which are
    dependent within a prompt (AGENTS.md) — and recomputes both regrets on the
    SAME resampled rows, holding the calibrated probabilities fixed. This mirrors
    the AUROC bootstrap exactly: it captures evaluation-set / prompt sampling
    variability, answering "is the regret reduction robust to which prompts we
    scored?" (It does NOT capture fit/calibration variability — neither does the
    AUROC bootstrap; the shared global calibrator largely cancels in the delta.)

    Returns per cost: {cost_draft, delta_ci_lo, delta_ci_hi, p_delta_ge_0 (the
    probability of NO help — Δ>=0), helps_ci (95% CI entirely below 0), n_boot}.
    Deterministic via ``np.random.default_rng(seed)``.
    """
    y = np.asarray(y)
    base_p = np.asarray(base_p, dtype=float)
    comb_p = np.asarray(comb_p, dtype=float)
    groups = np.asarray(groups)
    costs = np.asarray(costs, dtype=float)

    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        yb = y[rows]
        deltas.append(_regret_vec(yb, comb_p[rows], costs)
                      - _regret_vec(yb, base_p[rows], costs))
    D = np.asarray(deltas, dtype=float)                      # (n_boot, C)
    lo = np.percentile(D, 2.5, axis=0)
    hi = np.percentile(D, 97.5, axis=0)
    pge0 = np.mean(D >= 0.0, axis=0)
    return [{"cost_draft": float(costs[i]),
             "delta_ci_lo": float(lo[i]), "delta_ci_hi": float(hi[i]),
             "p_delta_ge_0": float(pge0[i]),
             "helps_ci": bool(hi[i] < 0.0),
             "n_boot": int(D.shape[0])} for i in range(len(costs))]


def _fit_oof(X, y, groups, seed=0, n_splits=5, c_reg=1.0, max_iter=5000) -> np.ndarray:
    """Prompt-grouped out-of-fold acceptance probabilities.

    Mirrors the proven pattern in ``scripts/fit_probes.py`` /
    ``scripts/fit_baselines.py``: ``GroupKFold``, ``StandardScaler`` fit on the
    train fold only, ``LogisticRegression``. ``n_splits`` is clamped to the number
    of distinct groups. A single-class train fold falls back to the constant
    train-prior probability (graceful, deterministic).

    ``c_reg`` is the inverse-L2 strength (sklearn ``C``). For a high-dimensional
    probe (raw = 14336 features over ~4k train rows) ``C=1.0`` is under-regularized:
    lbfgs may not converge within ``max_iter``, and a non-converged iterate depends
    on BLAS thread-order, so the AUROC wobbles run-to-run (docs/autoresearch_outcomes.md
    §6.2). A smaller ``c_reg`` (stronger L2) both regularizes appropriately AND makes
    the strictly-convex objective converge to its unique optimum -> the result is
    thread-independent and reproducible. Prefer ``c_reg<=0.1`` for the stabilized runs.
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
        clf = LogisticRegression(max_iter=max_iter, C=c_reg, random_state=seed)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def _fit_platt(scores, y, seed=0):
    """Fit the 2-parameter global Platt map on the LOG-ODDS of ``scores``.

    Returns the fitted ``LogisticRegression`` (apply with ``_apply_platt``), or
    ``None`` when ``y`` is single-class — callers fall back to the constant
    train-prior probability. Split out of ``_calibrate_oof`` so the frozen
    dev->test pass can fit the calibrator on DEV scores and apply it, frozen, to
    TEST scores (``frozen_transfer_lift``)."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(s / (1.0 - s)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(logit, y)
    return lr


def _apply_platt(cal, scores, prior=0.5) -> np.ndarray:
    """Apply a ``_fit_platt`` map to ``scores`` (same clip -> log-odds frame);
    constant ``prior`` when ``cal`` is None (single-class fit)."""
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1.0 - 1e-6)
    if cal is None:
        return np.full(len(s), float(prior))
    logit = np.log(s / (1.0 - s)).reshape(-1, 1)
    return cal.predict_proba(logit)[:, 1]


def _calibrate_oof(oof, y, seed=0) -> np.ndarray:
    """Global Platt recalibration of out-of-fold acceptance probabilities.

    Fits a single 2-parameter logistic (Platt) on the LOG-ODDS of the raw OOF
    scores and applies it to the same scores. The scores are already
    prompt-grouped out-of-fold (D019); a single GLOBAL monotonic map preserves
    AUROC/AUPRC *exactly* — only Brier/ECE/regret change — which is exactly what
    lets us isolate whether a candidate's ranking lift is decision-usable or
    merely a miscalibration artifact of a high-dim probe. (A per-fold calibrator
    would apply different monotone maps per fold and perturb AUROC, defeating the
    purpose.) The calibrator is 2 parameters over thousands of rows, so any
    in-sample optimism is negligible AND is applied identically to base and
    combined, so it cancels in the (combined - base) decision-metric delta we
    compare. Single-class y -> constant prior.
    """
    y = np.asarray(y)
    cal = _fit_platt(oof, y, seed=seed)
    if cal is None:
        return np.full(len(y), float(np.mean(y)) if len(y) else 0.5)
    return _apply_platt(cal, oof)


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
                     n_boot=1000, c_reg=1.0) -> dict:
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
    oof = {k: _fit_oof(v, y, groups, seed=seed, n_splits=n_splits, c_reg=c_reg)
           for k, v in designs.items()}
    models = {k: _metrics(y, v) for k, v in oof.items()}

    # Global Platt recalibration of the (already prompt-grouped OOF) base +
    # combined scores. A single monotonic map preserves AUROC/AUPRC and only moves
    # Brier/ECE/regret, so we can tell a decision-usable signal from a fixable
    # high-dim miscalibration artifact (D023; RESEARCH_SPEC controller /
    # mechanism-systems trade-off). Controls are AUROC-only checks -> not recalibrated.
    cal = {k: _calibrate_oof(oof[k], y, seed=seed)
           for k in ("base", "combined")}
    cal_models = {k: _metrics(y, v) for k, v in cal.items()}
    # Decisive systems check: does the ranking lift reduce decision regret at ANY
    # draft-cost ratio, or only where every model makes the same "always draft" call?
    # Point estimate + prompt-grouped bootstrap CI (so a real reduction is told
    # from threshold noise -- a non-informative candidate must NOT show helps_ci).
    reg_sweep = regret_cost_sweep(y, cal["base"], cal["combined"])
    reg_ci = _regret_sweep_ci(y, cal["base"], cal["combined"], groups,
                              seed=seed, n_boot=n_boot)
    for s, ci in zip(reg_sweep, reg_ci):
        s.update({"delta_ci_lo": ci["delta_ci_lo"], "delta_ci_hi": ci["delta_ci_hi"],
                  "p_delta_ge_0": ci["p_delta_ge_0"], "helps_ci": ci["helps_ci"]})

    def _cdelta(metric):
        a, b = cal_models["combined"][metric], cal_models["base"][metric]
        return None if (a is None or b is None) else float(a - b)

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

    auroc_ci = _group_bootstrap_delta(y, oof["base"], oof["combined"], groups,
                                      seed=seed, n_boot=n_boot)
    n_ci_robust = int(sum(1 for s in reg_sweep if s["helps_ci"]))
    # Credible systems signal: the ranking lift is statistically real (AUROC CI
    # excludes 0) AND the decision-regret reduction is CI-robust over a cost RANGE
    # (>= 2 costs). A single CI-robust cost from a worse-ranking feature (the drift
    # anomaly, docs/autoresearch_outcomes.md §5) is a threshold fluke, not evidence.
    auroc_ci_clean = bool(auroc_ci.get("p_delta_le_0") is not None
                          and auroc_ci["p_delta_le_0"] < 0.05)
    credible_systems = bool(auroc_ci_clean and n_ci_robust >= 2)

    out = dict(models)
    out.update({
        "deltas": {"auroc": _delta("auroc"), "auprc": _delta("auprc"),
                   "brier": _delta("brier")},
        "beats_baseline": bool(beats_baseline),
        "beats_controls": bool(beats_controls),
        "delta_auroc_ci": auroc_ci,
        # Recalibrated (Platt OOF) decision metrics for base vs combined. AUROC/
        # AUPRC in these blocks equal the raw ones (monotonic); Brier/ECE/regret
        # are the post-calibration values. `helps_decision_calibrated` is the
        # decision-usefulness verdict: lower regret than the calibrated baseline.
        "base_calibrated": cal_models["base"],
        "combined_calibrated": cal_models["combined"],
        "deltas_calibrated": {"brier": _cdelta("brier"), "ece": _cdelta("ece"),
                              "regret": _cdelta("regret")},
        "helps_decision_calibrated": bool(
            cal_models["combined"]["regret"] < cal_models["base"]["regret"] - 1e-6),
        "regret_cost_sweep": reg_sweep,
        "helps_at_any_cost": bool(any(s["helps"] for s in reg_sweep)),
        "helps_at_any_cost_ci": bool(any(s["helps_ci"] for s in reg_sweep)),
        "auroc_ci_clean": auroc_ci_clean,
        "n_ci_robust_costs": n_ci_robust,
        "credible_systems": credible_systems,
        "n": int(n),
        "pos_rate": float(np.mean(y)),
        "n_features_base": int(n_feat_base),
        "n_features_cand": int(n_feat_cand),
    })
    return out


def _fit_frozen_scores(X_dev, y_dev, X_test, seed=0, c_reg=1.0,
                       max_iter=5000) -> np.ndarray:
    """Dev-frozen probe scores for test rows.

    ``StandardScaler`` + ``LogisticRegression(C=c_reg)`` fit on ALL dev rows,
    applied exactly once to the test rows. Single-class dev falls back to the
    constant train-prior probability (mirrors ``_fit_oof``'s degenerate-fold
    fallback)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_dev = np.asarray(y_dev)
    if len(np.unique(y_dev)) < 2:
        prior = float(np.mean(y_dev)) if len(y_dev) else 0.0
        return np.full(X_test.shape[0], prior)
    sc = StandardScaler().fit(X_dev)
    clf = LogisticRegression(max_iter=max_iter, C=c_reg, random_state=seed)
    clf.fit(sc.transform(X_dev), y_dev)
    return clf.predict_proba(sc.transform(X_test))[:, 1]


def frozen_transfer_lift(X_base_dev, X_cand_dev, y_dev, groups_dev,
                         X_base_test, X_cand_test, y_test, groups_test,
                         seed=0, n_splits=5, n_boot=1000, c_reg=1.0) -> dict:
    """Strict dev->test transfer scorer: every fitted object is frozen on DEV.

    The one-shot frozen-test companion to ``incremental_lift``. Where
    ``incremental_lift`` on ``eval_split='test'`` runs GroupKFold OOF *within*
    the test rows (nothing is ever selected on test, but each fold model is
    still FIT on test rows), this scorer never fits anything on test:

      * ``StandardScaler`` + ``LogisticRegression(C=c_reg)`` are fit on ALL dev
        rows per design (base / combined / control_random / control_norm); the
        test rows are scored exactly once with the frozen model.
      * The equal-capacity controls are i.i.d. N(0,1) columns drawn
        deterministically from ``seed`` for dev and test rows; the norm-matched
        variant is rescaled to the CANDIDATE'S DEV per-column std (dev-frozen
        scale — no test statistic enters the pipeline).
      * The Platt calibrator is fit on DEV prompt-grouped OOF scores (via
        ``_fit_oof``, so its input distribution is never in-sample-optimistic)
        and applied frozen to the test scores. Caveat: the frozen full-dev model
        is trained on ~n_splits/(n_splits-1) more rows than each OOF fold model,
        so its score distribution is slightly sharper than the calibrator's
        input — AUROC/AUPRC are unaffected (monotone map); calibrated ECE /
        regret carry this honest, deployment-faithful transfer mismatch.
      * The AUROC-delta CI and the regret-sweep CI bootstrap TEST prompts
        (grouped, never rows), exactly as in ``incremental_lift``.

    Returns the ``incremental_lift`` result shape plus
    ``protocol='frozen_dev_to_test'``, ``n_dev``, ``pos_rate_dev``, and
    ``dev_reference`` (dev OOF base/combined AUROC for context)."""
    X_base_dev = np.asarray(X_base_dev, dtype=float)
    X_cand_dev = np.asarray(X_cand_dev, dtype=float)
    X_base_test = np.asarray(X_base_test, dtype=float)
    X_cand_test = np.asarray(X_cand_test, dtype=float)
    y_dev = np.asarray(y_dev)
    y_test = np.asarray(y_test)
    groups_dev = np.asarray(groups_dev)
    groups_test = np.asarray(groups_test)
    if X_base_dev.ndim == 1:
        X_base_dev = X_base_dev.reshape(-1, 1)
    if X_cand_dev.ndim == 1:
        X_cand_dev = X_cand_dev.reshape(-1, 1)
    if X_base_test.ndim == 1:
        X_base_test = X_base_test.reshape(-1, 1)
    if X_cand_test.ndim == 1:
        X_cand_test = X_cand_test.reshape(-1, 1)

    # Equal-capacity controls: dev drawn first, then test, from one seeded
    # stream — deterministic given shapes. Norm-match scale is DEV-frozen.
    rng = np.random.default_rng(seed)
    R_dev = rng.standard_normal(size=X_cand_dev.shape)
    R_test = rng.standard_normal(size=X_cand_test.shape)
    cand_std = X_cand_dev.std(axis=0)

    def _norm_match(R):
        r_std = R.std(axis=0)
        r_std_safe = np.where(r_std == 0.0, 1.0, r_std)
        return R / r_std_safe * cand_std

    designs = {
        "base": (X_base_dev, X_base_test),
        "combined": (np.hstack([X_base_dev, X_cand_dev]),
                     np.hstack([X_base_test, X_cand_test])),
        "control_random": (np.hstack([X_base_dev, R_dev]),
                           np.hstack([X_base_test, R_test])),
        "control_norm": (np.hstack([X_base_dev, _norm_match(R_dev)]),
                         np.hstack([X_base_test, _norm_match(R_test)])),
    }
    scores = {k: _fit_frozen_scores(dv, y_dev, tv, seed=seed, c_reg=c_reg)
              for k, (dv, tv) in designs.items()}
    models = {k: _metrics(y_test, v) for k, v in scores.items()}

    # Dev OOF: context numbers + the (frozen) calibrator's training input.
    oof_dev = {k: _fit_oof(designs[k][0], y_dev, groups_dev, seed=seed,
                           n_splits=n_splits, c_reg=c_reg)
               for k in ("base", "combined")}
    dev_ref = {k: _metrics(y_dev, v) for k, v in oof_dev.items()}

    prior = float(np.mean(y_dev)) if len(y_dev) else 0.5
    cal = {k: _apply_platt(_fit_platt(oof_dev[k], y_dev, seed=seed),
                           scores[k], prior=prior)
           for k in ("base", "combined")}
    cal_models = {k: _metrics(y_test, v) for k, v in cal.items()}
    reg_sweep = regret_cost_sweep(y_test, cal["base"], cal["combined"])
    reg_ci = _regret_sweep_ci(y_test, cal["base"], cal["combined"], groups_test,
                              seed=seed, n_boot=n_boot)
    for s, ci in zip(reg_sweep, reg_ci):
        s.update({"delta_ci_lo": ci["delta_ci_lo"], "delta_ci_hi": ci["delta_ci_hi"],
                  "p_delta_ge_0": ci["p_delta_ge_0"], "helps_ci": ci["helps_ci"]})

    def _cdelta(metric):
        a, b = cal_models["combined"][metric], cal_models["base"][metric]
        return None if (a is None or b is None) else float(a - b)

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

    auroc_ci = _group_bootstrap_delta(y_test, scores["base"], scores["combined"],
                                      groups_test, seed=seed, n_boot=n_boot)
    n_ci_robust = int(sum(1 for s in reg_sweep if s["helps_ci"]))
    auroc_ci_clean = bool(auroc_ci.get("p_delta_le_0") is not None
                          and auroc_ci["p_delta_le_0"] < 0.05)
    credible_systems = bool(auroc_ci_clean and n_ci_robust >= 2)

    out = dict(models)
    out.update({
        "deltas": {"auroc": _delta("auroc"), "auprc": _delta("auprc"),
                   "brier": _delta("brier")},
        "beats_baseline": bool(beats_baseline),
        "beats_controls": bool(beats_controls),
        "delta_auroc_ci": auroc_ci,
        "base_calibrated": cal_models["base"],
        "combined_calibrated": cal_models["combined"],
        "deltas_calibrated": {"brier": _cdelta("brier"), "ece": _cdelta("ece"),
                              "regret": _cdelta("regret")},
        "helps_decision_calibrated": bool(
            cal_models["combined"]["regret"] < cal_models["base"]["regret"] - 1e-6),
        "regret_cost_sweep": reg_sweep,
        "helps_at_any_cost": bool(any(s["helps"] for s in reg_sweep)),
        "helps_at_any_cost_ci": bool(any(s["helps_ci"] for s in reg_sweep)),
        "auroc_ci_clean": auroc_ci_clean,
        "n_ci_robust_costs": n_ci_robust,
        "credible_systems": credible_systems,
        "protocol": "frozen_dev_to_test",
        "n": int(len(y_test)),
        "pos_rate": float(np.mean(y_test)),
        "n_dev": int(len(y_dev)),
        "pos_rate_dev": float(np.mean(y_dev)),
        "dev_reference": {"base_auroc_oof": dev_ref["base"]["auroc"],
                          "combined_auroc_oof": dev_ref["combined"]["auroc"]},
        "n_features_base": int(X_base_dev.shape[1]),
        "n_features_cand": int(X_cand_dev.shape[1]),
    })
    return out


# Accepted-length thresholds for the per-length survival probes. These mirror the
# contract's candidate proposal lengths; k=1 == the binary accept event.
LENGTH_KS = (1, 2, 4, 6, 8)


def length_probe_lift(X_base, X_cand, accepted_len, groups, ks=LENGTH_KS,
                      seed=0, n_splits=5, n_boot=1000, min_class=20,
                      c_reg=1.0) -> dict:
    """Fit one probe per accepted-length threshold k: P(accepted_len >= k | x).

    Under exact-match greedy the accepted prefix is contiguous, so
    ``y_k = (accepted_len >= k)`` is the counterfactual **survival** label for
    drafting k tokens (D018/D019 counterfactual full-information labels): "would
    action L=k be fully accepted?". These K probes are the discrete survival curve
    S_k = P(A >= k | x); the length-aware ("how many tokens") counterpart to the
    single binary accept probe (k=1). For each k the frozen baseline and
    baseline+candidate are fit under prompt-grouped OOF with an equal-capacity
    random control; we report the incremental AUROC lift + prompt-grouped CI +
    calibrated ECE, plus the empirical survival rate P(A>=k). Reuses the same
    leakage-safe machinery as ``incremental_lift`` (no new statistics). k with too
    few of either class (< ``min_class``) are skipped with a note.
    """
    X_base = np.asarray(X_base, dtype=float)
    X_cand = np.asarray(X_cand, dtype=float)
    if X_base.ndim == 1:
        X_base = X_base.reshape(-1, 1)
    if X_cand.ndim == 1:
        X_cand = X_cand.reshape(-1, 1)
    a = np.asarray(accepted_len)
    groups = np.asarray(groups)

    rng = np.random.default_rng(seed)
    R = rng.standard_normal(size=X_cand.shape)          # equal-capacity control
    d_base = X_base
    d_comb = np.hstack([X_base, X_cand])
    d_ctrl = np.hstack([X_base, R])

    per_k, survival, surv_oof = {}, {}, {}
    for k in ks:
        y = (a >= k).astype(int)
        survival[int(k)] = float(y.mean())
        if (len(np.unique(y)) < 2 or int(y.sum()) < min_class
                or int((1 - y).sum()) < min_class):
            per_k[int(k)] = {"note": "insufficient class balance",
                             "pos_rate": float(y.mean())}
            continue
        ob = _fit_oof(d_base, y, groups, seed=seed, n_splits=n_splits, c_reg=c_reg)
        oc = _fit_oof(d_comb, y, groups, seed=seed, n_splits=n_splits, c_reg=c_reg)
        orr = _fit_oof(d_ctrl, y, groups, seed=seed, n_splits=n_splits, c_reg=c_reg)
        mb, mc, mr = _metrics(y, ob), _metrics(y, oc), _metrics(y, orr)
        ci = _group_bootstrap_delta(y, ob, oc, groups, seed=seed, n_boot=n_boot)
        cal_b = _calibrate_oof(ob, y, seed=seed)     # calibrated survival Ŝ_k, base
        cal_c = _calibrate_oof(oc, y, seed=seed)     # calibrated survival Ŝ_k, combined
        surv_oof[int(k)] = {"base": cal_b, "combined": cal_c}   # for length_payoff
        cal_b_ece = _metrics(y, cal_b)["ece"]
        cal_c_ece = _metrics(y, cal_c)["ece"]
        d_auroc = (None if mc["auroc"] is None or mb["auroc"] is None
                   else float(mc["auroc"] - mb["auroc"]))
        per_k[int(k)] = {
            "pos_rate": float(y.mean()),
            "base_auroc": mb["auroc"], "combined_auroc": mc["auroc"],
            "control_random_auroc": mr["auroc"],
            "delta_auroc": d_auroc, "delta_auroc_ci": ci,
            "beats_baseline": bool(mc["auroc"] is not None and mb["auroc"] is not None
                                   and mc["auroc"] > mb["auroc"] + 1e-4),
            "beats_control": bool(mc["auroc"] is not None and mr["auroc"] is not None
                                  and mc["auroc"] > mr["auroc"] + 1e-4),
            "auroc_ci_clean": bool(ci.get("p_delta_le_0") is not None
                                   and ci["p_delta_le_0"] < 0.05),
            "base_cal_ece": cal_b_ece, "combined_cal_ece": cal_c_ece,
        }
    return {"ks": [int(k) for k in ks], "per_k": per_k,
            "empirical_survival": survival, "n": int(len(a)),
            "surv_oof": surv_oof}       # calibrated Ŝ_k arrays for length_payoff (Tier-2)


def _interp_survival(surv_dict, ks, all_k) -> "np.ndarray":
    """Per-row survival interpolated from grid ``ks`` to integer thresholds
    ``all_k``, then enforced monotone non-increasing (the K probes are fit
    independently, so raw calibrated S_k need not be monotone). Returns
    ``(n, len(all_k))``."""
    ks_sorted = sorted(int(k) for k in ks)
    S = np.stack([np.asarray(surv_dict[k], dtype=float) for k in ks_sorted], axis=1)
    ks_arr = np.asarray(ks_sorted, dtype=float)
    out = np.empty((S.shape[0], len(all_k)), dtype=float)
    for j, kk in enumerate(all_k):
        if kk <= ks_arr[0]:
            out[:, j] = S[:, 0]
        elif kk >= ks_arr[-1]:
            out[:, j] = S[:, -1]
        else:
            hi = int(np.searchsorted(ks_arr, kk))
            lo = hi - 1
            w = (kk - ks_arr[lo]) / (ks_arr[hi] - ks_arr[lo])
            out[:, j] = (1.0 - w) * S[:, lo] + w * S[:, hi]
    return np.minimum.accumulate(out, axis=1)      # survival is non-increasing in k


def length_payoff(surv_base, surv_comb, accepted_len, groups, ks=LENGTH_KS,
                  costs=COST_GRID, seed=0, n_boot=1000) -> list:
    """Length-aware controller payoff (Tier-2): choose the proposal length L from a
    predicted survival curve, and measure realized-throughput regret vs a
    clairvoyant oracle across draft costs. Compares a controller driven by the
    CANDIDATE's survival (combined) against one driven by the frozen baseline
    (base) and the best fixed action.

    Model (verify = 1 time unit; a drafted token costs ``cost_draft``): action L
    emits ``1 + min(A, L)`` tokens (accepted prefix + bonus) at cost
    ``1 + cost_draft*L``, so throughput = ``(1 + min(A,L)) / (1 + cost_draft*L)``. A
    controller picks L maximizing EXPECTED throughput from the predicted survival
    (``E[min(A,L)] = sum_{j<=L} S_j``); realized throughput uses the true A. Actions
    = skip(0) + ``ks``. Returns per cost: regret of base / combined / best-fixed vs
    oracle, the (combined - base) regret delta with a prompt-grouped bootstrap CI
    (``helps_ci`` = CI entirely below 0), and the mean chosen length. This is the
    length decision the compute-optimal controller (I14) actually makes; still a
    coarse proxy -- wall-clock is authoritative (G3).
    """
    a = np.asarray(accepted_len, dtype=float)
    groups = np.asarray(groups)
    max_k = int(max(int(k) for k in ks))
    all_k = list(range(1, max_k + 1))
    csum_b = np.cumsum(_interp_survival(surv_base, ks, all_k), axis=1)   # E[min(A,j)] base
    csum_c = np.cumsum(_interp_survival(surv_comb, ks, all_k), axis=1)   # combined
    actions = [0] + sorted(int(k) for k in ks)
    idx = {L: i for i, L in enumerate(actions)}
    ones = np.ones(len(a))

    def _pred_emit(csum, L):
        return ones if L == 0 else 1.0 + csum[:, L - 1]

    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boot_rows = [np.concatenate([idx_by_g[g]
                                 for g in rng.choice(uniq, len(uniq), replace=True)])
                 for _ in range(n_boot)]

    out = []
    for c in costs:
        cost_L = np.array([1.0 + c * L for L in actions])            # (nA,)
        pe_b = np.stack([_pred_emit(csum_b, L) for L in actions], axis=1)
        pe_c = np.stack([_pred_emit(csum_c, L) for L in actions], axis=1)
        Lb = np.array(actions)[np.argmax(pe_b / cost_L[None, :], axis=1)]
        Lc = np.array(actions)[np.argmax(pe_c / cost_L[None, :], axis=1)]
        real_emit = np.stack([1.0 + np.minimum(a, L) for L in actions], axis=1)
        real_tp = real_emit / cost_L[None, :]                        # (n, nA)
        rt_oracle = real_tp.max(axis=1)
        rows = np.arange(len(a))
        rt_b = real_tp[rows, [idx[L] for L in Lb]]
        rt_c = real_tp[rows, [idx[L] for L in Lc]]
        Lfix = actions[int(np.argmax(real_tp.mean(axis=0)))]
        rt_fix = real_tp[:, idx[Lfix]]
        reg_o_b = rt_oracle - rt_b
        reg_o_c = rt_oracle - rt_c
        boot = np.array([float(np.mean(reg_o_c[r]) - np.mean(reg_o_b[r]))
                         for r in boot_rows])
        out.append({
            "cost_draft": float(c), "tau": float(c / (c + 1.0)),
            "regret_base": float(np.mean(reg_o_b)),
            "regret_combined": float(np.mean(reg_o_c)),
            "regret_best_fixed": float(np.mean(rt_oracle - rt_fix)),
            "best_fixed_action": int(Lfix),
            "delta_regret": float(np.mean(reg_o_c) - np.mean(reg_o_b)),
            "delta_ci_lo": float(np.percentile(boot, 2.5)),
            "delta_ci_hi": float(np.percentile(boot, 97.5)),
            "p_delta_ge_0": float(np.mean(boot >= 0.0)),
            "helps_ci": bool(np.percentile(boot, 97.5) < 0.0),
            "mean_len_base": float(np.mean(Lb)),
            "mean_len_combined": float(np.mean(Lc)),
        })
    return out
