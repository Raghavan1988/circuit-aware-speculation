"""I15 intervention harness (G2 causal validation) — pure, tool-agnostic logic.

Turns the correlational first-token acceptance direction (I13/I23,
docs/autoresearch_outcomes.md) into a CAUSAL test per the G2 criteria (D020,
AGENTS.md): an intervention that is **layer-specific, dose-responsive, replicated,
and survives random + norm-matched controls** — PLUS the requirement unique to
this finding: the causal effect on acceptance must exceed the induced ENTROPY
change (a causal analog of "beyond entropy"). If the acceptance effect is fully
entropy-mediated, the direction is causally an entropy knob, not a beyond-entropy
mechanism.

This module holds only the pure numpy logic (direction derivation, controls,
dose-response, entropy-mediation, verdict) so it is unit-tested locally; the GPU
steering (forward-hook / nnsight) lives in modal_app.py::intervene. NOTHING here
upgrades language to "circuit"/"mechanism" — that is a human gate AFTER these
tests pass (D020). D015 specifies nnsight for interventions; the runner scaffolds
with transparent forward hooks (auditable, no image change) and nnsight may be
swapped in.
"""
from __future__ import annotations

import numpy as np


def acceptance_direction(states, accept, normalize=True) -> np.ndarray:
    """First-token acceptance direction at a layer: mean(accepted) − mean(rejected)
    frontier state. Points toward acceptance. Derive on DEV only.

    states: (n, d) float frontier reps. accept: (n,) in {0,1}. Returns (d,).
    """
    states = np.asarray(states, dtype=float)
    accept = np.asarray(accept)
    if len(np.unique(accept)) < 2:
        raise ValueError("need both accepted and rejected states to form a direction")
    d = states[accept == 1].mean(axis=0) - states[accept == 0].mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if (normalize and n > 0) else d


def random_control(direction, seed=0) -> np.ndarray:
    """Norm-matched isotropic random control: iid Gaussian scaled to ||direction||.
    Satisfies 'random AND norm-matched' — the effect can't be a bigger push."""
    d = np.asarray(direction, dtype=float)
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(d.shape)
    rn = np.linalg.norm(r)
    return r / rn * np.linalg.norm(d) if rn > 0 else r


def shuffled_control(direction, seed=0) -> np.ndarray:
    """Norm-PRESERVING structural control: permute the direction's coordinates.
    Same norm and same marginal distribution, but the learned coordinate structure
    is destroyed — isolates structure from magnitude/marginals."""
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, dtype=float).copy()
    rng.shuffle(d)
    return d


def dose_response(alphas, accept_rates) -> dict:
    """Acceptance vs steering dose. Returns OLS slope, monotonicity, and the effect
    size = accept_rate(max α) − accept_rate(min α). `alphas` includes 0 (no-op)."""
    a = np.asarray(alphas, dtype=float)
    y = np.asarray(accept_rates, dtype=float)
    slope = float(np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, y, rcond=None)[0][0])
    order = np.argsort(a)
    ys = y[order]
    diffs = np.diff(ys)
    monotone = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))
    return {"slope": slope, "monotone": monotone, "effect": float(ys[-1] - ys[0])}


def beats_controls(real_effect, control_effects, margin=1e-6) -> bool:
    """Real |dose effect| exceeds every control's |dose effect| (not just push size)."""
    ce = [abs(float(c)) for c in control_effects]
    return bool(abs(float(real_effect)) > (max(ce) if ce else 0.0) + margin)


def entropy_mediation(alpha, accept, entropy) -> dict:
    """The causal 'beyond entropy' test (per-sample). Linear-probability regressions:

        accept ~ alpha            -> total effect  c_total
        accept ~ alpha + entropy  -> direct effect c_direct (α coef | entropy)

    Returns c_total, c_direct, direct_fraction = c_direct/c_total, and
    ``beyond_entropy`` = the direct α effect is same-signed as the total and a
    material fraction of it. direct_fraction ≈ 0 means the acceptance effect is
    fully mediated by the induced entropy change — the direction is causally an
    entropy knob, NOT beyond entropy. (Linear-probability approximation; the GPU
    runner can also report a logistic version.)"""
    a = np.asarray(alpha, dtype=float)
    y = np.asarray(accept, dtype=float)
    e = np.asarray(entropy, dtype=float)
    c_total = float(np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, y, rcond=None)[0][0])
    c_direct = float(np.linalg.lstsq(
        np.vstack([a, e, np.ones_like(a)]).T, y, rcond=None)[0][0])
    direct_fraction = c_direct / c_total if abs(c_total) > 1e-12 else 0.0
    beyond = bool(abs(c_direct) > 1e-6 and c_direct * c_total > 0
                  and abs(direct_fraction) > 0.1)
    return {"c_total": c_total, "c_direct": c_direct,
            "direct_fraction": float(direct_fraction), "beyond_entropy": beyond}


def entropy_stratified_effect(alpha, accept, entropy, n_bins=8) -> dict:
    """Robust, nonparametric complement to ``entropy_mediation``: within narrow
    entropy bins (entropy ~constant), does α still move acceptance? Averages the
    per-bin α→accept slope, weighted by bin count. A non-zero pooled within-bin
    slope is beyond-entropy evidence that does NOT depend on the linear-probability
    model being globally well-specified (it survives a step-shaped accept↔entropy
    relationship, which trips the LPM). Prefer this for the causal verdict."""
    a = np.asarray(alpha, dtype=float)
    y = np.asarray(accept, dtype=float)
    e = np.asarray(entropy, dtype=float)
    edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    slopes, weights = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (e >= lo) & ((e <= hi) if i == n_bins - 1 else (e < hi))
        if m.sum() >= 30 and np.std(a[m]) > 1e-9 and len(np.unique(y[m])) > 1:
            # within-bin, still regress OUT the residual entropy variation (coarse
            # bins leave some), then take alpha's partial coefficient.
            X = np.vstack([a[m], e[m], np.ones(int(m.sum()))]).T
            coef = float(np.linalg.lstsq(X, y[m], rcond=None)[0][0])
            slopes.append(coef)
            weights.append(int(m.sum()))
    if not slopes:
        return {"within_bin_slope": None, "n_bins_used": 0, "beyond_entropy": False}
    pooled = float(np.average(slopes, weights=weights))
    return {"within_bin_slope": pooled, "n_bins_used": len(slopes),
            "beyond_entropy": bool(abs(pooled) > 0.01)}     # magnitude threshold


def g2_verdict(dose_real, dose_controls, mediation) -> dict:
    """Combine one layer's results into a G2 sub-verdict: dose-responsive AND beats
    controls AND beyond-entropy. Layer-specificity and replication are judged ACROSS
    runs (a human gate, D020) — this never emits 'circuit'/'mechanism' language."""
    beats = beats_controls(dose_real["effect"], [d["effect"] for d in dose_controls])
    causal = bool(dose_real["monotone"] and abs(dose_real["effect"]) > 0
                  and beats and mediation["beyond_entropy"])
    return {"dose_effect": dose_real["effect"], "dose_monotone": dose_real["monotone"],
            "beats_controls": beats, "beyond_entropy": mediation["beyond_entropy"],
            "causal_beyond_entropy": causal,
            "language": "diagnostic signal (G2 not tripped until a human confirms "
                        "layer-specificity + replication, D020)"}
